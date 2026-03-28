#!/bin/sh
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
DEFAULT_PYTHON_BIN="${SCRIPT_DIR}/../.venv/bin/python"

MODEL_DIR="${MODEL_DIR:-}"
if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [ -x "${DEFAULT_PYTHON_BIN}" ]; then
  PYTHON_BIN="${DEFAULT_PYTHON_BIN}"
else
  PYTHON_BIN="python3"
fi
INPUT_TXT="${INPUT_TXT:-./data/requests.txt}"
OUTPUT_TXT="${OUTPUT_TXT:-./data/responses.txt}"
MAIN_BIN="${MAIN_BIN:-./main}"

if [ -z "${MODEL_DIR}" ]; then
  echo "MODEL_DIR is not set."
  echo "Example:"
  echo "  MODEL_DIR=/path/to/gpt-oss-20b INPUT_TXT=./data/requests.txt make run"
  exit 1
fi

"${PYTHON_BIN}" ./scripts/run_text_generation.py \
  --model-dir "${MODEL_DIR}" \
  --input "${INPUT_TXT}" \
  --output "${OUTPUT_TXT}" \
  --main-binary "${MAIN_BIN}" \
  "$@"
