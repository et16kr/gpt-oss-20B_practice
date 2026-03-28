#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from run_text_generation import encode_request, load_tokenizer, write_token_batch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT.parent / "images" / "gpt-oss-20b"


def repeated_prompt(prefix: str, repeat: int) -> str:
    body = " ".join(
        [
            "CUDA kernel scheduling",
            "memory coalescing",
            "shared memory tiling",
            "warp divergence",
            "occupancy tuning",
            "attention masking",
            "MoE routing",
            "vectorized loads",
        ]
        * repeat
    )
    return f"{prefix} {body}"


def request_sets() -> dict[str, list[str]]:
    return {
        "batch2": [
            "RMSNorm과 LayerNorm의 차이를 짧게 설명해줘.",
            repeated_prompt("Sliding window attention의 동작을 자세히 설명해줘.", 24),
        ],
        "batch3": [
            "GQA가 무엇인지 한 문단으로 설명해줘.",
            repeated_prompt("MXFP4 양자화와 dequant 흐름을 단계별로 설명해줘.", 10),
            repeated_prompt("MoE top-k routing이 왜 계산량을 줄이는지 설명해줘.", 20),
        ],
        "batch4": [
            "CUDA?",
            "RoPE의 핵심 아이디어를 설명해줘.",
            repeated_prompt("sink attention이 softmax에 어떻게 들어가는지 설명해줘.", 12),
            repeated_prompt("긴 컨텍스트에서 sliding attention의 한계와 장점을 비교해줘.", 16),
        ],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build fixed batch-2/3/4 text files and token batches for HF validation."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--system", default="")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    tokenizer = load_tokenizer(args.model_dir)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.bos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer is missing pad/eos/bos token ids")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, prompts in request_sets().items():
        txt_path = args.output_dir / f"hf_validation_{name}.txt"
        bin_path = args.output_dir / f"hf_validation_{name}.bin"
        txt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")
        sequences = [encode_request(tokenizer, prompt, args.system) for prompt in prompts]
        write_token_batch(bin_path, sequences, pad_token_id)
        print(f"Wrote {txt_path}")
        print(f"Wrote {bin_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
