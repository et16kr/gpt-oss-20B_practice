# gpt-oss-20B_practice 계획 문서

## 목표

- `llama3.2-1B_practice`처럼 학습용 inference 프로젝트를 새로 만든다.
- 우선은 `CPU reference`가 정확히 동작하도록 만들고, 이후 각 연산을 `*_gpu()` 커널 과제로 교체할 수 있게 한다.
- 최종적으로는 "프로젝트 뼈대와 CPU 기준 구현은 이미 준비되어 있고, 학생은 CUDA kernel만 채우면 gpt-oss-20b가 돌아가는 상태"를 목표로 한다.

## 이번 프로젝트의 방향

- 기준 템플릿은 [`llama3.2-1B_practice`](/home/et16/aps/llama3.2-1B_practice)로 잡는다.
- 다만 gpt-oss-20b는 Llama보다 구조가 복잡하므로, 그대로 복제하지 않고 아래 차이점을 별도 작업축으로 분리한다.
- 1차 범위는 "짧은 입력, greedy generation, tokenized input 중심"으로 제한한다.
- text chat용 wrapper는 넣되, 커널 학습 흐름을 방해하지 않도록 core inference 안정화 뒤에 붙인다.

## 확인된 gpt-oss-20b 특성

로컬 모델 파일과 공식 공개 구현 기준으로 현재 확인한 내용:

- 24개 transformer block
- `hidden_size = 2880`
- `num_attention_heads = 64`
- `num_key_value_heads = 8`
- `head_dim = 64`
- `q_proj` 출력은 `64 * 64 = 4096`, `k_proj`/`v_proj` 출력은 `8 * 64 = 512`
- layer별 attention 타입이 `sliding_attention` / `full_attention`로 교차
- `sliding_window = 128`
- `num_local_experts = 32`
- `experts_per_token = 4`
- attention에는 bias와 per-head `sinks` 파라미터가 있음
- RoPE는 기본형이 아니라 YaRN 계열 확장 설정을 사용
- 일반 텐서는 주로 BF16
- MoE MLP 가중치는 `tensor.blocks` + `tensor.scales` 형태의 MXFP4 계열 양자화 표현을 사용

## Llama 템플릿에서 그대로 가져갈 부분

- 디렉터리 구조
  - `include/`
  - `src/`
  - `data/`
  - `scripts/`
  - `Makefile`
- 실행 흐름
  - `main`에서 CLI 파싱
  - tokenized input 로딩
  - `forward-only` / `generation` 모드 분리
- 학습 구조
  - CPU 기준 구현 먼저 작성
  - 동일한 함수 시그니처의 `*_gpu()`를 별도로 둠
  - 필요 시 GPU 경로는 초기에는 CPU fallback 또는 TODO 형태로 시작
- 검증 방식
  - 작은 입력에 대해 logits 비교
  - generation 결과 또는 마지막 token top-k 비교

## gpt-oss 전용으로 새로 설계해야 하는 부분

### 1. 체크포인트 로더

- Llama 실습은 단일 `model.safetensors` 기준이지만, 현재 로컬 `gpt-oss-20b`는 root 디렉터리에 shard 3개와 `model.safetensors.index.json`이 있다.
- 따라서 `ShardedSafetensorsLoader`가 필요하다.
- 1차 구현 대상은 root checkpoint인 [`images/gpt-oss-20b`](/home/et16/aps/images/gpt-oss-20b)로 잡는다.
  - 이유: tensor 이름이 `q_proj`, `k_proj`, `v_proj`처럼 Llama 실습 구조와 더 가깝다.
- 다만 공식 reference는 `original/model.safetensors`를 사용하므로, 향후 correctness 비교용 보조 로더는 남겨둘 수 있다.

### 2. attention 구조

- Llama와 달리 `q_proj` 차원이 hidden size와 다르다.
- attention 계산은 GQA 형태다.
- 일부 layer는 full causal attention, 일부 layer는 sliding window causal attention만 적용한다.
- softmax 전에 sink column을 함께 붙이는 로직이 필요하다.
- attention bias도 반영해야 한다.

### 3. MoE MLP 구조

- Llama의 dense SwiGLU MLP가 아니라 top-k routed MoE다.
- 각 token마다 router logits를 계산하고, 상위 4개 expert를 선택한다.
- 선택된 expert weight만 사용해 두 단계 MLP를 수행하고, softmax 가중합으로 합친다.
- MLP 내부 활성화는 일반적인 Llama식 SwiGLU와 동일하게 두지 말고, gpt-oss reference semantics를 그대로 따른다.
  - even/odd split
  - clamp
  - `sigmoid(alpha * x)` 형태
  - linear branch에 `+1` 보정

### 4. MXFP4 expert weight 처리

- 공식 문서 기준으로 MoE projection weight는 packed FP4 blocks와 block scales로 저장된다.
- CPU reference에서 이 가중치를 전부 BF16/FP32로 미리 풀어 메모리에 들고 가면 메모리 낭비가 크다.
- 따라서 1차 설계는 아래 방향이 적합하다.
  - checkpoint에서는 quantized raw tensor를 유지
  - 선택된 expert에 대해서만 on-the-fly dequantize
  - dequant 결과는 작은 scratch buffer에 잠시 올림
- 이 방식이 이후 CUDA 커널 과제와도 더 잘 연결된다.

### 5. harmony 입력 형식

- gpt-oss는 harmony format을 전제로 학습되었다.
- 따라서 text generation wrapper는 일반 chat template가 아니라 harmony 기반 입력 포맷을 써야 한다.
- 다만 커널 연습의 핵심은 tokenizer보다 inference core이므로, 구현 순서는 다음처럼 둔다.
  - 먼저 `token-input -> logits/generation`
  - 그 다음 `Python wrapper -> harmony prompt 생성/결과 decode`

## 제안하는 프로젝트 구조

- `gpt-oss-20B_practice/include/gpt_oss_config.h`
- `gpt-oss-20B_practice/include/tensor.h`
- `gpt-oss-20B_practice/include/layer.h`
- `gpt-oss-20B_practice/include/model.h`
- `gpt-oss-20B_practice/include/app_options.h`
- `gpt-oss-20B_practice/include/generation.h`
- `gpt-oss-20B_practice/include/safetensors_loader.h`
- `gpt-oss-20B_practice/src/config.cpp`
- `gpt-oss-20B_practice/src/safetensors_loader.cpp`
- `gpt-oss-20B_practice/src/tensor.cu`
- `gpt-oss-20B_practice/src/layer.cu`
- `gpt-oss-20B_practice/src/model.cu`
- `gpt-oss-20B_practice/src/generation.cpp`
- `gpt-oss-20B_practice/src/app_options.cpp`
- `gpt-oss-20B_practice/src/main.cpp`
- `gpt-oss-20B_practice/scripts/run_text_generation.py`
- `gpt-oss-20B_practice/scripts/compare_hf_logits.py`
- `gpt-oss-20B_practice/scripts/build_hf_validation_batch.py`
- `gpt-oss-20B_practice/data/`
- `gpt-oss-20B_practice/Makefile`

## 연산 단위 설계 초안

Llama 실습의 함수 분해를 유지하되, 아래처럼 gpt-oss에 맞게 바꾼다.

- `EmbeddingLookup`
- `RMSNorm`
- `LinearBias`
- `SplitQKVHeadsGrouped`
- `ApplyYaRNRoPE`
- `AttentionScoresWithSink`
- `ScaleMaskSoftmax`
- `AttentionContextGrouped`
- `MergeHeads`
- `ResidualAdd`
- `RouterLogits`
- `TopKExperts`
- `DequantizeExpertBlocks`
- `ExpertGateUp`
- `SwiGLUClamp`
- `ExpertDown`
- `WeightedExpertReduce`
- `LMHead`

각 연산마다 `CPU`와 `*_gpu()` 엔트리 둘 다 둔다.

## 단계별 구현 계획

### Phase 0. 뼈대 복제

- `llama3.2-1B_practice`의 프로젝트 구조를 새 디렉터리로 복사하되, 이름과 설명을 `gpt-oss-20B_practice`로 바꾼다.
- `LlamaConfig`를 `GptOssConfig`로 교체한다.
- build, CLI, token batch 포맷, forward-only/generation 모드의 뼈대를 먼저 맞춘다.

완료 기준:

- `make`가 되고
- `./main --help`가 뜨고
- 아직 model load 전이라도 프로젝트 골격이 맞다.

### Phase 1. config + sharded loader

- `config.json` 파서를 gpt-oss 필드에 맞게 작성한다.
- `model.safetensors.index.json`을 읽어 tensor name -> shard path 매핑을 만든다.
- BF16 tensor 로딩을 지원한다.
- expert의 `blocks`, `scales`, `bias`를 raw 형태로 읽을 수 있게 한다.

완료 기준:

- embedding, norm, q/k/v/o, router, expert raw tensor shape를 모두 로드 가능
- shape mismatch 시 명확한 에러 출력

### Phase 2. CPU forward-only

- 먼저 `B=1`로 정확한 forward를 맞춘 뒤, 곧바로 `B=2`, `B=3`, `B=4` 검증 배치로 확장한다.
- attention block과 MoE block을 CPU 기준으로 구현한다.
- 우선은 no-cache 경로로 간다.
- sliding/full attention mask를 layer별로 구분한다.
- Hugging Face `transformers` forward와 같은 입력 배치에 대해 logits를 직접 비교한다.

완료 기준:

- 작은 입력에 대해 logits를 저장할 수 있음
- Hugging Face `transformers`와 마지막 token logits 및 argmax가 batch `2~4` 케이스에서 일치하거나 허용 오차 내
- 길이가 서로 다른 prompt를 섞은 batch에서도 padding/masking이 올바르게 동작

### Phase 3. CPU generation

- 현재 llama 실습처럼 greedy decoding 루프를 만든다.
- 1차는 KV cache 없이 매 step 전체 prompt를 다시 계산한다.
- 짧은 prompt와 작은 `max_new_tokens` 범위에서 동작하도록 한다.
- batch `2~4` 기준으로 여러 요청을 동시에 넣었을 때도 개별 sequence 길이와 stop 처리가 올바르게 유지되는지 확인한다.

완료 기준:

- token input으로 generated token output 생성 가능
- EOS/return stop 처리 가능
- batch `2~4`의 짧은 요청 세트에 대해 Hugging Face greedy 결과와 비교 가능

### Phase 4. harmony wrapper

- Python wrapper에서 harmony 형식 prompt를 만든다.
- line-based input file을 읽어 tokenized batch를 생성한다.
- 생성 결과를 decode해서 text output으로 저장한다.

완료 기준:

- `make run` 또는 `python3 scripts/run_text_generation.py ...`로 실행 가능
- 최소 1개의 짧은 요청에 대해 end-to-end 실행 가능

### Phase 5. CUDA 연습용 분리

- CPU 기준 구현이 맞은 뒤, 각 연산에 `*_gpu()` 경로를 분리한다.
- 초기 GPU 버전은 CPU 결과를 먼저 내는 fallback이어도 괜찮다.
- 이후 학생이 kernel만 바꿔 끼울 수 있도록 op 경계를 안정화한다.

완료 기준:

- TODO 대상 연산 목록이 명확함
- validation 옵션으로 CPU/GPU 결과 비교 가능

## 1차 범위에서 의도적으로 제외할 것

- 긴 컨텍스트 최적화
- KV cache
- speculative decoding
- sampling temperature/top-p
- multi-GPU
- production 급 최적화
- full harmony tool-use 루프

이 항목들은 정확한 CPU 기준 경로와 커널 학습용 인터페이스가 먼저 안정화된 뒤에 다룬다.

## 주요 리스크

### 리스크 1. MXFP4 해석 오류

- 가장 위험한 부분이다.
- expert weight dequant 로직이 조금만 틀려도 logits 전체가 무너진다.
- 가장 먼저 작은 tensor 단위 dequant test를 따로 두는 것이 좋다.

### 리스크 2. sink attention semantics 누락

- 일반 causal attention만 구현하면 결과가 틀릴 가능성이 높다.
- sink column을 softmax에 포함시키는 reference 동작을 그대로 따라야 한다.

### 리스크 3. harmony 없이 text 품질 확인

- token-level forward는 맞더라도 text chat 품질이 이상할 수 있다.
- wrapper 단계에서 harmony 적용 여부를 반드시 분리 검증해야 한다.

### 리스크 4. CPU 메모리/속도

- 이 프로젝트는 학습용이라 느려도 괜찮지만, 무작정 모든 expert weight를 풀면 메모리가 급격히 커진다.
- 따라서 "선택된 expert만 dequantize" 전략을 초기 설계부터 고정하는 것이 좋다.

### 리스크 5. batch padding/attention mask 불일치

- batch size 1에서는 맞아도, batch 2 이상에서 padding 처리나 causal/sliding mask 경계가 틀릴 수 있다.
- 특히 서로 다른 길이의 prompt를 섞으면 마지막 token logits이 쉽게 어긋난다.
- 따라서 batch `2`, `3`, `4`를 고정 검증 세트로 두고 초기에 계속 비교해야 한다.

## Hugging Face `transformers` 기준 검증 계획

- 검증 기준 구현은 Hugging Face root checkpoint인 [`images/gpt-oss-20b`](/home/et16/aps/images/gpt-oss-20b)를 사용한다.
- 같은 `input_ids`, 같은 `attention_mask`, 같은 batch ordering으로 우리 구현과 Hugging Face `transformers`를 동시에 실행한다.
- 비교 대상은 generation text만이 아니라 `전체 logits`이다.
- 검증 스크립트는 [`llama3.2-1B_practice/scripts/compare_hf_logits.py`](/home/et16/aps/llama3.2-1B_practice/scripts/compare_hf_logits.py) 패턴을 참고해 새로 만든다.

### 검증 배치 구성

- `B=2`, `B=3`, `B=4` 세 가지 고정 배치를 준비한다.
- 각 배치는 prompt 길이가 서로 다르게 되도록 만든다.
- 최소 1개 배치는 아주 짧은 요청과 상대적으로 긴 요청을 섞는다.
- 최소 1개 배치는 sliding window 영향이 드러나도록 어느 정도 길이를 확보한다.
- harmony wrapper가 준비되기 전에는 pretokenized batch로 먼저 검증하고, 준비된 뒤에는 harmony tokenize 결과로 다시 검증한다.

예시:

- batch 2: 짧은 prompt 1개 + 중간 길이 prompt 1개
- batch 3: 서로 다른 길이의 prompt 3개
- batch 4: 매우 짧은 prompt, 일반 prompt, 비교적 긴 prompt, 특수 토큰이 포함된 prompt

### 비교 항목

- 전체 logits shape 일치 여부
- valid token 위치에서의 `max abs diff`
- valid token 위치에서의 `mean abs diff`
- valid token 위치에서의 `RMSE`
- 각 batch row 마지막 valid token의 argmax 일치 여부
- 각 batch row 마지막 valid token의 top-k 겹침 정도
- greedy decoding 시 생성 token sequence 일치 여부

### 검증 시 주의할 점

- padding된 위치는 비교 대상에서 제외하거나, Hugging Face의 `attention_mask`와 정확히 동일하게 맞춘 뒤 비교한다.
- logits 비교는 가능하면 `float32`로 캐스팅한 뒤 수행한다.
- batch 내 각 row의 `lengths`를 별도로 유지해 마지막 valid token 기준으로 비교한다.
- generation 비교는 sampling이 아니라 greedy로 고정한다.

### 스크립트/데이터 계획

- `scripts/build_hf_validation_batch.py`
  - batch `2`, `3`, `4` 검증용 prompt 세트와 token batch 생성
- `scripts/compare_hf_logits.py`
  - 우리 `main --logits-output` 결과와 Hugging Face `transformers` logits 비교
- `data/hf_validation_batch2.txt`
- `data/hf_validation_batch3.txt`
- `data/hf_validation_batch4.txt`
- `data/hf_validation_batch2.bin`
- `data/hf_validation_batch3.bin`
- `data/hf_validation_batch4.bin`

### Phase별 반영 방식

- Phase 1 직후
  - loader가 읽은 tensor shape를 Hugging Face model parameter shape와 spot check
- Phase 2 직후
  - batch `2`, `3`, `4` forward logits를 Hugging Face와 직접 비교
- Phase 3 직후
  - batch `2`, `3`, `4` greedy generation 결과 비교
- Phase 5 직후
  - 각 `*_gpu()` 치환 후 동일한 Hugging Face 기준으로 회귀 검증

## 검증 계획

- Hugging Face `transformers`를 기준 레퍼런스로 사용
- 고정 token batch `B=2`, `B=3`, `B=4`에 대해 logits 저장 및 비교
- 각 batch는 서로 다른 sequence length를 포함하도록 구성
- valid token 위치 기준으로 `max abs diff`, `mean abs diff`, `RMSE` 측정
- 각 row의 마지막 valid token top-k와 argmax 비교
- 작은 prompt에 대해 batch `2~4` greedy generation 결과 비교
- sliding layer와 full layer를 모두 지나가는 길이의 입력으로 테스트
- batch `2~4`가 안정화된 뒤에 필요하면 `B=1`과 더 큰 batch로 확장

## 다음 작업 제안

다음 순서는 아래가 가장 안전하다.

1. `gpt-oss-20B_practice` 골격 복사
2. `GptOssConfig`와 sharded loader 작성
3. expert quantized tensor 표현 타입 설계
4. CPU attention block 구현
5. CPU MoE block 구현
6. forward-only 검증 harness 추가

## 참고 자료

- 로컬 템플릿: [`llama3.2-1B_practice`](/home/et16/aps/llama3.2-1B_practice)
- 로컬 모델: [`images/gpt-oss-20b`](/home/et16/aps/images/gpt-oss-20b)
- 공식 공개 구현: https://github.com/openai/gpt-oss
- 공식 모델 카드: https://huggingface.co/openai/gpt-oss-20b
