# gpt-oss-20B_practice

CUDA 커널 연습용 `gpt-oss-20b` inference 프로젝트입니다.

- active runtime은 GPU-only 경로입니다.
- weight는 checkpoint의 native dtype(BF16, MXFP4/U8)를 변환 없이 device에 올립니다.
- legacy CPU reference는 소스 안에 `#if 0` 블록으로 남겨 두고, build/runtime에서는 제외합니다.
- 목표는 학생이 `*_gpu_*` 경계를 따라 커널을 하나씩 교체해 가며 `gpt-oss-20b` 추론 경로를 직접 완성하는 것입니다.

## 기준 모델

이 프로젝트는 Hugging Face의 `openai/gpt-oss-20b` 모델을 기준으로 구현하고 검증합니다.

- 기준 모델 페이지: `https://huggingface.co/openai/gpt-oss-20b`
- 기본 checkpoint 경로도 이 모델을 내려받은 `images/gpt-oss-20b/` root 디렉터리를 가정합니다.

## 기대 효과

이 저장소에서 커널을 하나씩 직접 구현해 보면, 현재 `gpt-oss-20b` 같은 최신 open-weight MoE 모델을 실제로 어떻게 추론하는지 구조적으로 이해할 수 있습니다.

- attention 경로를 구현하면서 `GQA`, `YaRN RoPE`, `sink attention`, `sliding/full attention`의 실제 데이터 흐름을 익힐 수 있습니다.
- MoE 경로를 구현하면서 `router`, `top-k expert selection`, `expert weight 적용`, `weighted reduce`가 어떻게 연결되는지 알 수 있습니다.
- `BF16`, `MXFP4`, `FP32 accumulation`을 분리해 다루면서 실제 모델 추론에서 dtype 설계가 왜 중요한지 익힐 수 있습니다.
- sharded safetensors loader를 통해 최신 대형 모델 checkpoint가 어떤 형식으로 저장되고 로드되는지 이해할 수 있습니다.
- CPU reference와 Hugging Face `transformers` 비교를 통해, 커널 최적화와 별개로 먼저 정답 경로를 맞추는 습관을 훈련할 수 있습니다.
- `*_gpu_*` 경계별로 구현을 교체해 가며, 큰 모델을 한 번에 다 쓰지 않고 attention, MoE, norm, matmul 같은 단위로 나눠 개발하는 방식을 연습할 수 있습니다.

## 현재 범위

- root checkpoint `images/gpt-oss-20b/` 지원
- sharded safetensors 로딩
- BF16 일반 weight raw 로딩
- MXFP4 expert weight raw 로딩 및 on-the-fly dot product
- GQA + YaRN RoPE + sliding/full attention + sink attention
- top-4 MoE + interleaved SwiGLU
- forward-only / greedy generation
- Python text wrapper
- Hugging Face `transformers` 비교 스크립트와 batch 2~4 검증 입력 생성기

## 빌드

```bash
make
```

Python wrapper와 Hugging Face 비교 스크립트는 저장소 루트의 공용 환경 `../.venv`를 기준으로 사용하는 것을 권장합니다.
`run.sh`와 `make run`은 `PYTHON_BIN`을 따로 주지 않으면 먼저 `../.venv/bin/python`을 찾습니다.
`gpt-oss-20b`를 `transformers`에서 GPU로 비교하려면 공용 환경에 최소 `transformers`, `torch`, `numpy`, `accelerate`, `kernels`가 있어야 합니다.

## 최소 실행 예시

토큰 입력 파일 형식:

- `int32 B`
- `int32 T`
- `int32 lengths[B]`
- `int32 token_ids[B*T]`

forward-only:

```bash
./main -m /path/to/gpt-oss-20b \
       --token-input /path/to/prompt_tokens.bin \
       --logits-output ./data/logits.bin
```

generation:

```bash
./main -m /path/to/gpt-oss-20b \
       --token-input /path/to/prompt_tokens.bin \
       --token-output ./data/generated_tokens.bin \
       --max-new-tokens 16
```

text wrapper:

```bash
PYTHON_BIN=../.venv/bin/python \
MODEL_DIR=/path/to/gpt-oss-20b \
INPUT_TXT=./data/requests.txt \
OUTPUT_TXT=./data/responses.txt \
make run
```

## 검증 스크립트

고정 검증 배치 생성:

```bash
../.venv/bin/python ./scripts/build_hf_validation_batch.py \
        --model-dir /path/to/gpt-oss-20b \
        --output-dir ./data
```

Hugging Face 비교:

```bash
../.venv/bin/python ./scripts/compare_hf_logits.py \
        --model-dir /path/to/gpt-oss-20b \
        --requests ./data/hf_validation_batch2.txt \
        --mode both
```

## 주의

- 이 저장소는 학습용 custom CUDA kernel 경로라 빠르지 않을 수 있습니다.
- C++ binary 내부 CPU validation은 제거했고, 정답 비교는 Hugging Face `transformers` 스크립트 기준으로만 지원합니다.
- 공용 환경 `../.venv`에 `transformers`, `torch`, `numpy`, `accelerate`, `kernels`가 설치되어 있어야 Python wrapper/HF 비교 스크립트를 실행할 수 있습니다.
- legacy CPU reference는 build/runtime에서 비활성화되어 있으며, 읽기 전용 참고 자료로만 남아 있습니다.
