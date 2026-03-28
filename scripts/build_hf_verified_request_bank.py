#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT.parent / "images" / "Llama-3.2-1B-Instruct"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "hf_verified_request_bank.txt"


GLOBAL_BANNED_PATTERNS = (
    "for that type of information",
    "please refer to meta",
    "meta's privacy",
    "meta’s privacy",
    "privacy policy",
    "native cryptocurrency",
    "cryptocurrency",
    "decentralized finance",
    "taker token",
    "bidirectional encoder representations",
    "audio processing",
    "text compression",
    "not enough information",
    "database systems",
    "database queries",
    "sql",
)


@dataclass(frozen=True)
class Topic:
    phrase: str
    must_have: tuple[str, ...]
    banned: tuple[str, ...] = ()


@dataclass(frozen=True)
class PairTopic:
    left: str
    right: str
    must_have: tuple[str, ...]
    banned: tuple[str, ...] = ()


@dataclass(frozen=True)
class PromptSpec:
    text: str
    must_have: tuple[str, ...]
    banned: tuple[str, ...] = ()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a bank of HF-verified requests and save them as one prompt per line."
    )
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--count", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=20260311)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=20)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    return parser


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def explanation_topics() -> list[Topic]:
    return [
        Topic("causal masking", ("causal", "mask", "attention", "future token")),
        Topic("rotary positional embeddings", ("rotary", "position", "embedding", "rope")),
        Topic("grouped-query attention", ("grouped", "query", "attention", "key-value")),
        Topic("root mean square normalization", ("normalization", "rms", "variance", "layer")),
        Topic("layer normalization", ("layer", "normalization", "mean", "variance")),
        Topic("residual connections", ("residual", "connection", "gradient", "signal")),
        Topic("self-attention", ("attention", "token", "context", "query")),
        Topic("cross-attention", ("attention", "encoder", "decoder", "context")),
        Topic("feed-forward networks in transformers", ("feed-forward", "activation", "hidden", "block")),
        Topic("the query projection in self-attention", ("query", "projection", "attention", "hidden state")),
        Topic("the key projection in self-attention", ("key", "projection", "attention", "hidden state")),
        Topic("the value projection in self-attention", ("value", "projection", "attention", "hidden state")),
        Topic("the attention output projection", ("output projection", "attention", "heads", "hidden state")),
        Topic("token embeddings", ("embedding", "token", "vector", "representation")),
        Topic("positional information in transformers", ("position", "token", "order", "embedding")),
        Topic("the context window", ("context window", "token", "generation", "sequence")),
        Topic("the end-of-sequence token", ("sequence", "token", "generation", "stop")),
        Topic("greedy decoding", ("greedy", "token", "highest", "generation")),
        Topic("top-k sampling", ("sampling", "top-k", "token", "distribution")),
        Topic("nucleus sampling", ("sampling", "top-p", "probability", "token")),
        Topic("logits", ("logits", "score", "probability", "softmax")),
        Topic("probabilities after softmax", ("probability", "softmax", "distribution", "logit")),
        Topic("softmax normalization", ("softmax", "probability", "normalization", "distribution")),
        Topic("the max-subtraction trick in softmax", ("softmax", "max", "stability", "overflow")),
        Topic("attention score scaling", ("scale", "attention", "dot product", "head dimension")),
        Topic("the key-value cache", ("cache", "key", "value", "generation")),
        Topic("valid sequence lengths in padded batches", ("length", "padding", "attention", "sequence")),
        Topic("left padding versus right padding for decoder-only models", ("padding", "left", "right", "decoder")),
        Topic("tensor shapes in attention", ("shape", "attention", "tensor", "dimension")),
        Topic("the head dimension", ("head dimension", "attention", "shape", "projection")),
        Topic("the batch dimension", ("batch", "dimension", "tensor", "shape")),
        Topic("the sequence length dimension", ("sequence length", "dimension", "token", "shape")),
        Topic("hidden size in a transformer", ("hidden size", "representation", "dimension", "layer")),
        Topic("the intermediate size in the feed-forward block", ("intermediate", "feed-forward", "hidden", "dimension")),
        Topic("the bfloat16 data type", ("bfloat16", "precision", "range", "inference"), ("bidirectional encoder",)),
        Topic("the float16 data type", ("float16", "precision", "range", "inference")),
        Topic("float32 accumulation", ("float32", "accumulation", "precision", "stability")),
        Topic("numerical stability in transformer inference", ("numerical", "stability", "overflow", "underflow")),
        Topic("memory bandwidth bottlenecks", ("memory bandwidth", "memory", "bottleneck", "throughput")),
        Topic("memory coalescing on a GPU", ("coalescing", "memory", "gpu", "access")),
        Topic("shared memory reuse", ("shared memory", "reuse", "memory", "tile")),
        Topic("register pressure", ("register", "pressure", "occupancy", "kernel")),
        Topic("warp divergence", ("warp", "divergence", "branch", "thread")),
        Topic("kernel launch overhead", ("kernel launch", "overhead", "latency", "launch")),
        Topic("occupancy on a CUDA GPU", ("occupancy", "thread", "block", "gpu")),
        Topic("kernel fusion", ("fusion", "kernel", "memory", "launch")),
        Topic("tensor cores", ("tensor core", "matrix", "throughput", "gpu")),
        Topic("pinned host memory", ("pinned", "host memory", "transfer", "cuda")),
        Topic("asynchronous memory copies", ("asynchronous", "copy", "overlap", "transfer")),
        Topic("throughput", ("throughput", "request", "token", "second")),
        Topic("latency", ("latency", "response", "delay", "request")),
        Topic("batching in transformer inference", ("batch", "throughput", "latency", "request")),
        Topic("validating GPU output against a CPU reference", ("validate", "cpu", "gpu", "reference")),
        Topic("debugging NaNs in model outputs", ("nan", "debug", "overflow", "numerical")),
        Topic("profiling CUDA kernels", ("profile", "kernel", "cuda", "bottleneck")),
        Topic("prompt tokenization", ("tokenization", "prompt", "token", "text")),
        Topic("chat templates in instruction-tuned models", ("chat template", "message", "prompt", "token")),
        Topic("attention masks", ("attention mask", "mask", "attention", "token")),
        Topic("causal attention masks", ("causal", "mask", "attention", "future token")),
        Topic("the normalization epsilon constant", ("epsilon", "normalization", "stability", "division")),
        Topic("parameter tying between the embedding matrix and the language modeling head", ("embedding", "head", "tie", "parameter")),
        Topic("temperature in text generation", ("temperature", "sampling", "distribution", "token")),
        Topic("prompt truncation", ("truncation", "prompt", "context window", "token")),
        Topic("sharing key-value heads across query heads", ("key-value", "query", "head", "grouped")),
        Topic("merging attention heads", ("merge", "head", "attention", "hidden state")),
        Topic("splitting hidden states into attention heads", ("split", "head", "attention", "hidden state")),
        Topic("rotary angle frequencies", ("rotary", "frequency", "angle", "position")),
        Topic("query-key dot products", ("query", "key", "dot product", "attention")),
        Topic("context vectors in attention", ("context vector", "attention", "value", "weighted sum")),
        Topic("the autoregressive generation loop", ("autoregressive", "generation", "token", "loop")),
        Topic("prompt length versus generation length", ("prompt", "generation", "length", "context")),
        Topic("padding masks in batched attention", ("padding", "mask", "attention", "batch")),
        Topic("the language modeling head", ("language modeling head", "logit", "vocabulary", "projection")),
        Topic("the token embedding matrix", ("embedding matrix", "token", "vector", "vocabulary")),
        Topic("the vocabulary dimension of logits", ("vocabulary", "logit", "dimension", "token")),
        Topic("the attention score matrix", ("attention score", "matrix", "query", "key")),
        Topic("the attention probability matrix", ("attention", "probability", "softmax", "matrix")),
        Topic("the attention context tensor", ("context", "attention", "value", "tensor")),
        Topic("the feed-forward activation tensor", ("feed-forward", "activation", "tensor", "intermediate")),
        Topic("the query states tensor in self-attention", ("query state", "self-attention", "tensor", "shape")),
        Topic("the key states tensor in self-attention", ("key state", "self-attention", "tensor", "shape")),
        Topic("the value states tensor in self-attention", ("value state", "self-attention", "tensor", "shape")),
        Topic("prompt padding", ("padding", "prompt", "token", "batch")),
        Topic("autoregressive next-token prediction", ("autoregressive", "next token", "prediction", "generation")),
        Topic("the difference between model correctness and model speed", ("correctness", "speed", "validation", "performance")),
        Topic("attention head grouping", ("attention", "group", "head", "key-value")),
        Topic("batch-major tensor layout", ("batch", "tensor", "layout", "memory")),
    ]


def comparison_topics() -> list[PairTopic]:
    return [
        PairTopic("throughput", "latency", ("throughput", "latency", "request", "delay")),
        PairTopic("bfloat16", "float16", ("bfloat16", "float16", "precision", "range"), ("bidirectional encoder",)),
        PairTopic("root mean square normalization", "layer normalization", ("normalization", "rms", "layer", "variance")),
        PairTopic("token embeddings", "positional embeddings", ("embedding", "token", "position", "order")),
        PairTopic("query projection", "key projection", ("query", "key", "projection", "attention")),
        PairTopic("value projection", "attention output projection", ("value", "output projection", "attention", "head")),
        PairTopic("logits", "probabilities", ("logits", "probability", "softmax", "score")),
        PairTopic("causal masking", "padding masking", ("mask", "causal", "padding", "attention")),
        PairTopic("greedy decoding", "sampling", ("greedy", "sampling", "token", "generation")),
        PairTopic("batching", "single-request inference", ("batch", "latency", "throughput", "request")),
        PairTopic("shared memory", "global memory", ("shared memory", "global memory", "latency", "bandwidth")),
        PairTopic("memory bandwidth bottlenecks", "compute bottlenecks", ("memory", "compute", "bottleneck", "throughput")),
        PairTopic("left padding", "right padding", ("padding", "left", "right", "decoder")),
        PairTopic("a key-value cache", "full recomputation", ("cache", "recompute", "generation", "latency")),
        PairTopic("prompt length", "generation length", ("prompt", "generation", "length", "context")),
        PairTopic("tensor cores", "standard CUDA cores", ("tensor core", "cuda core", "matrix", "throughput")),
        PairTopic("validation tests", "performance benchmarks", ("validation", "benchmark", "correctness", "performance")),
        PairTopic("an end-of-sequence token", "a padding token", ("sequence", "padding", "token", "generation")),
        PairTopic("split-heads layouts", "merge-heads layouts", ("split", "merge", "head", "layout")),
        PairTopic("numerical stability", "raw throughput", ("numerical", "stability", "throughput", "correctness")),
    ]


def component_topics() -> list[Topic]:
    return [
        Topic("query projection matrix", ("query", "projection", "attention", "hidden state")),
        Topic("key projection matrix", ("key", "projection", "attention", "hidden state")),
        Topic("value projection matrix", ("value", "projection", "attention", "hidden state")),
        Topic("attention output projection matrix", ("output projection", "attention", "head", "hidden state")),
        Topic("token embedding matrix", ("embedding", "token", "vector", "vocabulary")),
        Topic("language modeling head", ("language modeling head", "logit", "vocabulary", "projection")),
        Topic("root mean square normalization layer", ("normalization", "rms", "scale", "variance")),
        Topic("layer normalization layer", ("layer", "normalization", "mean", "variance")),
        Topic("residual connection", ("residual", "connection", "signal", "layer")),
        Topic("feed-forward network", ("feed-forward", "activation", "hidden", "block")),
        Topic("SwiGLU activation pattern", ("swiglu", "activation", "gate", "feed-forward")),
        Topic("causal attention mask", ("causal", "mask", "attention", "future token")),
        Topic("padding mask", ("padding", "mask", "attention", "sequence")),
        Topic("key-value cache", ("cache", "key", "value", "generation")),
        Topic("gate projection in the feed-forward block", ("gate projection", "feed-forward", "activation", "swi")),
        Topic("up projection in the feed-forward block", ("up projection", "feed-forward", "hidden", "activation")),
        Topic("down projection in the feed-forward block", ("down projection", "feed-forward", "hidden", "projection")),
        Topic("attention score tensor", ("attention score", "query", "key", "tensor")),
        Topic("attention probability tensor", ("attention", "probability", "softmax", "tensor")),
        Topic("context tensor", ("context", "attention", "value", "weighted sum")),
    ]


def debug_topics() -> list[Topic]:
    return [
        Topic("NaNs after softmax", ("nan", "softmax", "overflow", "stability")),
        Topic("NaNs in a CUDA attention kernel", ("nan", "attention", "cuda", "debug")),
        Topic("mismatched tensor shapes", ("shape", "tensor", "dimension", "mismatch")),
        Topic("invalid padding lengths", ("padding", "length", "sequence", "attention")),
        Topic("incorrect causal masks", ("causal", "mask", "future token", "attention")),
        Topic("wrong rotary positional embedding math", ("rotary", "position", "embedding", "angle")),
        Topic("a bad split-heads layout", ("split", "head", "layout", "tensor")),
        Topic("a bad merge-heads layout", ("merge", "head", "layout", "tensor")),
        Topic("a broken key-value cache", ("cache", "key", "value", "generation")),
        Topic("numerical overflow in logits", ("overflow", "logit", "stability", "softmax")),
        Topic("underflow in attention probabilities", ("underflow", "attention", "probability", "softmax")),
        Topic("uncoalesced memory accesses", ("coalescing", "memory", "access", "bandwidth")),
        Topic("register pressure", ("register", "pressure", "occupancy", "kernel")),
        Topic("warp divergence", ("warp", "divergence", "branch", "thread")),
        Topic("too many kernel launches", ("kernel launch", "overhead", "latency", "fusion")),
        Topic("incorrect bfloat16 casts", ("bfloat16", "cast", "precision", "stability"), ("bidirectional encoder",)),
        Topic("a broken attention mask", ("attention mask", "mask", "token", "sequence")),
        Topic("wrong valid sequence lengths", ("valid length", "sequence", "padding", "attention")),
        Topic("a mismatch between CPU and GPU outputs", ("cpu", "gpu", "mismatch", "reference")),
        Topic("incorrect tokenization assumptions", ("tokenization", "prompt", "token", "text")),
    ]


def shape_topics() -> list[Topic]:
    return [
        Topic("query states in self-attention", ("query state", "self-attention", "shape", "head")),
        Topic("key states in self-attention", ("key state", "self-attention", "shape", "head")),
        Topic("value states in self-attention", ("value state", "self-attention", "shape", "head")),
        Topic("the attention score matrix", ("attention score", "matrix", "shape", "query")),
        Topic("the attention probability matrix", ("attention", "probability", "matrix", "shape")),
        Topic("the context tensor", ("context", "tensor", "shape", "value")),
        Topic("the merged attention output", ("merge", "attention", "output", "shape")),
        Topic("the feed-forward activation tensor", ("feed-forward", "activation", "shape", "intermediate")),
        Topic("the logits tensor", ("logit", "tensor", "shape", "vocabulary")),
        Topic("the token id batch", ("token", "batch", "shape", "sequence")),
        Topic("the padding mask tensor", ("padding", "mask", "tensor", "shape")),
        Topic("the key-value cache layout", ("cache", "layout", "shape", "key")),
    ]


EXPLANATION_TEMPLATES = (
    "Explain {phrase} in simple terms for someone learning transformer inference.",
    "Give a concise explanation of {phrase} in the context of autoregressive text generation.",
    "Summarize {phrase} without using equations.",
    "Why is {phrase} important in a transformer language model?",
    "What problem does {phrase} solve in transformer inference?",
    "Describe the role of {phrase} during text generation.",
    "How would you teach {phrase} to a student implementing a transformer from scratch?",
    "Give an intuitive explanation of {phrase} for GPU inference work.",
    "What is the main idea behind {phrase} in a modern transformer?",
    "Why does an inference implementation need {phrase}?",
    "Explain why {phrase} matters when bringing up a new transformer inference stack.",
    "Give a practical explanation of {phrase} for debugging model outputs.",
    "How does {phrase} influence correctness or stability during inference?",
    "What should a new engineer remember about {phrase}?",
)

COMPARISON_TEMPLATES = (
    "Compare {left} and {right} in transformer inference.",
    "What is the difference between {left} and {right} in a transformer model?",
    "When would you prefer {left} over {right} in inference code?",
    "Give a short comparison of {left} versus {right}.",
    "Explain {left} and {right} to someone implementing attention.",
    "What tradeoff separates {left} from {right}?",
    "How do {left} and {right} affect model behavior differently?",
    "Describe the practical difference between {left} and {right}.",
    "Why might an inference engineer choose {left} instead of {right}?",
    "Which details most clearly distinguish {left} from {right}?",
)

COMPONENT_TEMPLATES = (
    "What does the {phrase} do in self-attention?",
    "Explain the purpose of the {phrase} inside a transformer block.",
    "How does the {phrase} affect the result of self-attention?",
    "Why is the {phrase} needed during autoregressive generation?",
    "Describe the job of the {phrase} in plain language.",
    "What information flows through the {phrase}?",
    "How would you explain the {phrase} to a new engineer?",
    "Give a short practical explanation of the {phrase}.",
    "Why does the transformer block keep the {phrase} around?",
    "What would break if the {phrase} were wrong?",
)

DEBUG_TEMPLATES = (
    "How would you debug problems caused by {phrase} in transformer inference?",
    "What is a practical debugging plan for {phrase} in CUDA inference code?",
    "Why can {phrase} break correctness in a transformer implementation?",
    "How can you detect {phrase} early during model bring-up?",
    "What symptoms usually point to {phrase} in a transformer stack?",
    "How would you narrow down {phrase} in an attention kernel?",
    "What logs or checks help diagnose {phrase}?",
    "What is a safe first step when investigating {phrase}?",
    "What is the most likely failure mode behind {phrase}?",
    "How do you separate {phrase} from a performance-only issue?",
)

SHAPE_TEMPLATES = (
    "Explain the tensor shape of {phrase} in a transformer.",
    "How would you describe the dimensions of {phrase} to a beginner?",
    "Why is that shape used for {phrase}?",
    "What does each dimension mean in the tensor for {phrase}?",
    "Give an intuitive explanation of the tensor shape for {phrase}.",
    "How should an engineer read the shape of {phrase} during debugging?",
    "Why is the shape of {phrase} important for correctness?",
    "What mistakes happen when the shape of {phrase} is misunderstood?",
    "What should each axis mean in the shape of {phrase}?",
    "How can the shape of {phrase} reveal an implementation bug?",
)


def prompt_specs() -> list[PromptSpec]:
    specs: list[PromptSpec] = []

    for topic in explanation_topics():
        for template in EXPLANATION_TEMPLATES:
            specs.append(PromptSpec(template.format(phrase=topic.phrase), topic.must_have, topic.banned))

    for topic in comparison_topics():
        for template in COMPARISON_TEMPLATES:
            specs.append(PromptSpec(template.format(left=topic.left, right=topic.right), topic.must_have, topic.banned))

    for topic in component_topics():
        for template in COMPONENT_TEMPLATES:
            specs.append(PromptSpec(template.format(phrase=topic.phrase), topic.must_have, topic.banned))

    for topic in debug_topics():
        for template in DEBUG_TEMPLATES:
            specs.append(PromptSpec(template.format(phrase=topic.phrase), topic.must_have, topic.banned))

    for topic in shape_topics():
        for template in SHAPE_TEMPLATES:
            specs.append(PromptSpec(template.format(phrase=topic.phrase), topic.must_have, topic.banned))

    deduped: dict[str, PromptSpec] = {}
    for spec in specs:
        deduped.setdefault(spec.text, spec)
    return list(deduped.values())


def choose_device(arg: str, torch) -> str:
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def load_model_and_tokenizer(model_dir: Path, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("tokenizer is missing pad_token_id and eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        str(model_dir),
        local_files_only=True,
        dtype=dtype,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def generate_batch(tokenizer, model, prompts: list[str], max_new_tokens: int, device: str) -> list[str]:
    import torch

    rendered = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]

    batch = tokenizer(
        rendered,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    prompt_len = input_ids.shape[1]

    with torch.inference_mode():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    new_tokens = outputs[:, prompt_len:]
    return [
        tokenizer.decode(tokens.tolist(), skip_special_tokens=True).strip()
        for tokens in new_tokens
    ]


def is_good_response(spec: PromptSpec, response: str) -> bool:
    text = normalize(response)
    if len(text.split()) < 5:
        return False
    if "====" in response:
        return False
    if any(pattern in text for pattern in GLOBAL_BANNED_PATTERNS):
        return False
    if any(pattern in text for pattern in spec.banned):
        return False
    return any(keyword in text for keyword in spec.must_have)


def write_prompts(path: Path, prompts: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt)
            f.write("\n")


def main() -> int:
    args = build_parser().parse_args()

    import torch

    device = choose_device(args.device, torch)
    rng = random.Random(args.seed)

    specs = prompt_specs()
    rng.shuffle(specs)

    tokenizer, model = load_model_and_tokenizer(args.model_dir, device)

    accepted: list[str] = []
    checked = 0

    while checked < len(specs) and len(accepted) < args.count:
        batch_specs = specs[checked : checked + args.batch_size]
        checked += len(batch_specs)
        responses = generate_batch(
            tokenizer,
            model,
            [spec.text for spec in batch_specs],
            args.max_new_tokens,
            device,
        )

        for spec, response in zip(batch_specs, responses):
            if is_good_response(spec, response):
                accepted.append(spec.text)
                if len(accepted) >= args.count:
                    break

        print(
            f"checked={checked}/{len(specs)} accepted={len(accepted)}/{args.count}",
            flush=True,
        )

    if len(accepted) < args.count:
        raise RuntimeError(
            f"only accepted {len(accepted)} prompts out of requested {args.count}"
        )

    write_prompts(args.output, accepted)
    print(f"saved {len(accepted)} prompts to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
