#include "app_options.h"

#include <stdexcept>
#include <string>

namespace {

std::string require_value(int argc, char **argv, int *index, const char *flag) {
  if (*index + 1 >= argc) {
    throw std::runtime_error(std::string("missing value for ") + flag);
  }
  ++(*index);
  return argv[*index];
}

}  // namespace

CliOptions parse_args(int argc, char **argv) {
  CliOptions options;

  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "-m" || arg == "--model-dir") {
      options.model_dir = require_value(argc, argv, &index, "--model-dir");
    } else if (arg == "--token-input") {
      options.token_input_path = require_value(argc, argv, &index, "--token-input");
    } else if (arg == "--token-output") {
      options.token_output_path = require_value(argc, argv, &index, "--token-output");
    } else if (arg == "--logits-output") {
      options.logits_output_path = require_value(argc, argv, &index, "--logits-output");
    } else if (arg == "-c" || arg == "--context-len") {
      options.context_len = std::stoi(require_value(argc, argv, &index, "--context-len"));
    } else if (arg == "-n" || arg == "--max-new-tokens") {
      options.max_new_tokens =
          std::stoi(require_value(argc, argv, &index, "--max-new-tokens"));
    } else if (arg == "-v" || arg == "--validate") {
      options.run_validation = true;
    } else if (arg == "-w" || arg == "--warmup") {
      options.run_warmup = true;
    } else if (arg == "--cuda-profiler-range") {
      options.cuda_profiler_range = true;
    } else if (arg == "-h" || arg == "--help") {
      options.help = true;
    } else {
      throw std::runtime_error("unknown flag: " + arg);
    }
  }

  return options;
}

void print_help(const CliOptions &defaults) {
  (void)defaults;
  printf("Usage: ./main -m MODEL_DIR --token-input prompts.bin [options]\n\n");
  printf("Generation mode:\n");
  printf("  -m, --model-dir PATH       gpt-oss model directory\n");
  printf("      --token-input PATH     Tokenized prompt batch\n");
  printf("      --token-output PATH    Generated token batch output\n");
  printf("  -c, --context-len N        Context window used during generation\n");
  printf("  -n, --max-new-tokens N     Greedy generation length (default: 64)\n");
  printf("  -v, --validate             Compare forward output against CPU reference\n");
  printf("  -w, --warmup               Warm up once before generation\n");
  printf("      --cuda-profiler-range  Wrap timed inference with cudaProfilerStart/Stop\n\n");
  printf("Forward-only mode:\n");
  printf("      --logits-output PATH   Write binary logits instead of generated tokens\n\n");
  printf("Token batch format:\n");
  printf("  int32 B\n");
  printf("  int32 T\n");
  printf("  int32 lengths[B]           optional in legacy files, required for variable length\n");
  printf("  int32 token_ids[B*T]\n\n");
  printf("Generated token output format:\n");
  printf("  int32 B\n");
  printf("  int32 T_generated_max\n");
  printf("  int32 lengths[B]\n");
  printf("  int32 token_ids[B*T_generated_max]\n");
}

void validate_options(const CliOptions &options) {
  if (options.model_dir.empty()) {
    throw std::runtime_error("--model-dir is required");
  }
  if (options.token_input_path.empty()) {
    throw std::runtime_error("--token-input is required");
  }
  if (options.max_new_tokens < 0) {
    throw std::runtime_error("--max-new-tokens must be non-negative");
  }
}
