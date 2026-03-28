#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import subprocess
import tempfile
from pathlib import Path

from run_text_generation import (
    encode_request,
    load_tokenizer,
    read_token_batch,
    write_token_batch,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare local C++ logits/generation against Hugging Face gpt-oss."
    )
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--token-input", type=Path)
    parser.add_argument("--requests", type=Path)
    parser.add_argument("--system", default="")
    parser.add_argument("--main-binary", type=Path, default=PROJECT_ROOT / "main")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mode", choices=["logits", "generation", "both"], default="both")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    parser.add_argument(
        "--hf-dtype",
        choices=["float32", "bfloat16"],
        default="bfloat16",
        help="dtype for non-quantized HF modules when loading the reference model",
    )
    parser.add_argument(
        "--gpu-max-memory-gib",
        type=int,
        default=20,
        help="GPU budget used when --device auto dispatches with device_map=auto",
    )
    parser.add_argument(
        "--cpu-max-memory-gib",
        type=int,
        default=48,
        help="CPU budget used when --device auto dispatches with device_map=auto",
    )
    return parser


def load_or_build_sequences(args: argparse.Namespace) -> tuple[list[list[int]], int]:
    tokenizer = load_tokenizer(args.model_dir)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.bos_token_id
    if pad_token_id is None:
        raise ValueError("tokenizer is missing pad/eos/bos token ids")

    if args.token_input is not None:
        sequences, _ = read_token_batch(args.token_input)
        return sequences, pad_token_id

    if args.requests is None:
        raise ValueError("either --token-input or --requests is required")

    prompts = [line.rstrip("\r\n") for line in args.requests.read_text(encoding="utf-8").splitlines()]
    if not prompts:
        raise ValueError("request file is empty")
    sequences = [encode_request(tokenizer, prompt, args.system) for prompt in prompts]
    return sequences, pad_token_id


def run_cpp_logits(
    main_binary: Path, model_dir: Path, sequences: list[list[int]], pad_token_id: int, vocab_size: int
) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="gpt_oss_compare_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        main_binary = main_binary.expanduser()
        if not main_binary.is_absolute():
            main_binary = main_binary.resolve()
        token_path = tmpdir_path / "prompt_tokens.bin"
        logits_path = tmpdir_path / "logits.bin"
        write_token_batch(token_path, sequences, pad_token_id)
        cmd = [
            str(main_binary),
            "-m",
            str(model_dir),
            "--token-input",
            str(token_path),
            "--logits-output",
            str(logits_path),
        ]
        subprocess.run(cmd, check=True)
        lengths = [len(seq) for seq in sequences]
        batch = len(sequences)
        max_len = max(lengths)
        logits = np.fromfile(logits_path, dtype=np.float32)
        expected = batch * max_len * vocab_size
        if logits.size != expected:
            raise ValueError(f"unexpected logits size: got {logits.size}, expected {expected}")
        return logits.reshape(batch, max_len, vocab_size)


def run_cpp_generation(
    main_binary: Path,
    model_dir: Path,
    sequences: list[list[int]],
    pad_token_id: int,
    max_new_tokens: int,
) -> list[list[int]]:
    with tempfile.TemporaryDirectory(prefix="gpt_oss_gen_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        main_binary = main_binary.expanduser()
        if not main_binary.is_absolute():
            main_binary = main_binary.resolve()
        token_path = tmpdir_path / "prompt_tokens.bin"
        output_path = tmpdir_path / "generated_tokens.bin"
        write_token_batch(token_path, sequences, pad_token_id)
        cmd = [
            str(main_binary),
            "-m",
            str(model_dir),
            "--token-input",
            str(token_path),
            "--token-output",
            str(output_path),
            "--max-new-tokens",
            str(max_new_tokens),
        ]
        subprocess.run(cmd, check=True)
        return read_token_batch(output_path)[0]


def load_hf_model(model_dir: Path, args: argparse.Namespace):
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("torch and transformers are required for HF comparison") from exc

    want_cuda = args.device in ("cuda", "auto") and torch.cuda.is_available()
    if args.hf_dtype == "float32":
        load_dtype = torch.float32
    else:
        load_dtype = torch.bfloat16 if want_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
    )
    load_kwargs = {
        "local_files_only": True,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "dtype": load_dtype,
    }
    if args.device == "cuda" and want_cuda:
        load_kwargs["device_map"] = "cuda"
    elif args.device == "auto" and want_cuda:
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = {
            0: f"{args.gpu_max_memory_gib}GiB",
            "cpu": f"{args.cpu_max_memory_gib}GiB",
        }

    model = AutoModelForCausalLM.from_pretrained(str(model_dir), **load_kwargs)
    model.eval()
    device_map = getattr(model, "hf_device_map", None)
    if device_map is not None:
        input_device = infer_model_input_device(model, torch.device("cpu"))
        return tokenizer, model, input_device

    device = torch.device("cuda" if want_cuda else "cpu")
    model.to(device)
    return tokenizer, model, device


def make_batch_arrays(sequences: list[list[int]], pad_token_id: int):
    batch = len(sequences)
    max_len = max(len(seq) for seq in sequences)
    input_ids = np.full((batch, max_len), pad_token_id, dtype=np.int64)
    attention_mask = np.zeros((batch, max_len), dtype=np.int64)
    lengths = np.zeros(batch, dtype=np.int64)
    for i, seq in enumerate(sequences):
        input_ids[i, : len(seq)] = np.asarray(seq, dtype=np.int64)
        attention_mask[i, : len(seq)] = 1
        lengths[i] = len(seq)
    return input_ids, attention_mask, lengths


def normalize_hf_device(device_value):
    import torch

    if isinstance(device_value, torch.device):
        return device_value
    if isinstance(device_value, int):
        return torch.device(f"cuda:{device_value}")
    return torch.device(str(device_value))


def infer_model_input_device(model, fallback_device):
    import torch

    device_map = getattr(model, "hf_device_map", None)
    if not device_map:
        return fallback_device

    for key in ("model.embed_tokens", "embed_tokens", "model", ""):
        if key not in device_map:
            continue
        target = device_map[key]
        if target in ("cpu", "disk"):
            return torch.device("cpu")
        return normalize_hf_device(target)

    for target in device_map.values():
        if target not in ("cpu", "disk"):
            return normalize_hf_device(target)
    return torch.device("cpu")


def make_torch_batch(device, sequences: list[list[int]], pad_token_id: int):
    import torch

    input_ids, attention_mask, lengths = make_batch_arrays(sequences, pad_token_id)
    return (
        torch.as_tensor(input_ids, device=device),
        torch.as_tensor(attention_mask, device=device),
        lengths,
    )


def run_hf_logits(model, device, sequences: list[list[int]], pad_token_id: int) -> np.ndarray:
    input_ids, attention_mask, _ = make_torch_batch(device, sequences, pad_token_id)
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    return outputs.logits.float().cpu().numpy()


def run_hf_generation(model, device, sequences: list[list[int]], pad_token_id: int, max_new_tokens: int) -> list[list[int]]:
    input_ids, attention_mask, lengths = make_torch_batch(device, sequences, pad_token_id)
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=pad_token_id,
            eos_token_id=model.config.eos_token_id,
        )
    generated = generated.cpu().numpy()
    outputs: list[list[int]] = []
    for row, prompt_len in zip(generated, lengths.tolist()):
        outputs.append(row[prompt_len:].tolist())
    return outputs


def print_topk(title: str, tokenizer, logits: np.ndarray, k: int) -> None:
    values = logits
    top_ids = np.argsort(values)[-k:][::-1]
    print(title)
    for rank, token_id in enumerate(top_ids, start=1):
        text = tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")
        print(f"  {rank}. id={int(token_id):6d} logit={values[token_id]:10.6f} text={text!r}")


def compare_logits(tokenizer, cpp_logits: np.ndarray, hf_logits: np.ndarray, lengths: list[int], top_k: int) -> None:
    diff_values: list[np.ndarray] = []
    argmax_match = True
    for batch_idx, valid_len in enumerate(lengths):
        row_cpp = cpp_logits[batch_idx, :valid_len]
        row_hf = hf_logits[batch_idx, :valid_len]
        diff_values.append((row_cpp - row_hf).reshape(-1))
        last_cpp = row_cpp[-1]
        last_hf = row_hf[-1]
        cpp_next = int(np.argmax(last_cpp))
        hf_next = int(np.argmax(last_hf))
        argmax_match = argmax_match and cpp_next == hf_next
        print(f"[batch {batch_idx}] last-token argmax cpp={cpp_next} hf={hf_next}")
        print_topk(f"[batch {batch_idx}] C++ top-k:", tokenizer, last_cpp, top_k)
        print_topk(f"[batch {batch_idx}] HF top-k:", tokenizer, last_hf, top_k)

    diff = np.concatenate(diff_values)
    print(f"Overall max abs diff: {np.max(np.abs(diff)):.6f}")
    print(f"Overall mean abs diff: {np.mean(np.abs(diff)):.6f}")
    print(f"Overall RMSE: {np.sqrt(np.mean(diff * diff)):.6f}")
    print(f"Argmax all match: {'YES' if argmax_match else 'NO'}")


def compare_generation(cpp_tokens: list[list[int]], hf_tokens: list[list[int]]) -> None:
    all_match = True
    for idx, (cpp_row, hf_row) in enumerate(zip(cpp_tokens, hf_tokens)):
        match = cpp_row == hf_row
        all_match = all_match and match
        print(f"[batch {idx}] completion match: {'YES' if match else 'NO'}")
        print(f"  cpp={cpp_row}")
        print(f"  hf ={hf_row}")
    print(f"Generation all match: {'YES' if all_match else 'NO'}")


def main() -> int:
    args = build_parser().parse_args()
    try:
        global np
        import numpy as np
        global torch
        import torch
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("numpy is required for compare_hf_logits.py") from exc

    sequences, pad_token_id = load_or_build_sequences(args)
    lengths = [len(seq) for seq in sequences]
    cpp_logits = None
    cpp_generated = None

    if args.mode in ("logits", "both"):
        config = AutoConfig.from_pretrained(
            str(args.model_dir),
            local_files_only=True,
            trust_remote_code=True,
        )
        cpp_logits = run_cpp_logits(
            args.main_binary, args.model_dir, sequences, pad_token_id, config.vocab_size
        )

    if args.mode in ("generation", "both"):
        cpp_generated = run_cpp_generation(
            args.main_binary, args.model_dir, sequences, pad_token_id, args.max_new_tokens
        )

    tokenizer, model, device = load_hf_model(args.model_dir, args)

    if args.mode in ("logits", "both"):
        hf_logits = run_hf_logits(model, device, sequences, pad_token_id)
        if cpp_logits.shape != hf_logits.shape:
            raise ValueError(f"logits shape mismatch: cpp={cpp_logits.shape}, hf={hf_logits.shape}")
        compare_logits(tokenizer, cpp_logits, hf_logits, lengths, args.top_k)

    if args.mode in ("generation", "both"):
        hf_generated = run_hf_generation(model, device, sequences, pad_token_id, args.max_new_tokens)
        compare_generation(cpp_generated, hf_generated)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
