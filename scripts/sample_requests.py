#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "data" / "hf_verified_request_bank.txt"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "requests.txt"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Randomly sample a requested number of prompts from a verified prompt bank."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, default=20260311)
    return parser


def load_prompts(path: Path) -> list[str]:
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    prompts = [prompt for prompt in prompts if prompt]
    unique = list(dict.fromkeys(prompts))
    if not unique:
        raise ValueError(f"no prompts found in {path}")
    return unique


def main() -> int:
    args = build_parser().parse_args()
    prompts = load_prompts(args.input)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.count > len(prompts):
        raise ValueError(
            f"requested {args.count} prompts, but only {len(prompts)} are available"
        )

    rng = random.Random(args.seed)
    sampled = rng.sample(prompts, args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sampled) + "\n", encoding="utf-8")
    print(f"saved {len(sampled)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
