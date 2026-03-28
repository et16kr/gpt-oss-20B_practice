# gpt-oss-20B_practice

CUDA 커널 연습용 `gpt-oss-20b` inference 프로젝트입니다.

- CPU reference forward/generation을 먼저 제공합니다.
- 각 연산은 `*_gpu()` 경계를 가지고 있으며, 현재는 CPU fallback으로 동작합니다.
- 목표는 학생이 커널을 하나씩 교체해 가며 `gpt-oss-20b` 추론 경로를 직접 완성하는 것입니다.

## 현재 범위

- root checkpoint `images/gpt-oss-20b/` 지원
- sharded safetensors 로딩
- BF16 일반 weight 로딩
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

- 이 저장소는 학습용 CPU reference 중심이라 느립니다.
- 공용 환경 `../.venv`에 `transformers`, `torch`, `numpy`, `accelerate`, `kernels`가 설치되어 있어야 Python wrapper/HF 비교 스크립트를 실행할 수 있습니다.
- `*_gpu()`는 현재 CPU fallback이며, 이후 커널 구현용 자리입니다.
