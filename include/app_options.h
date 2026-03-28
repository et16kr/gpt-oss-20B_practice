#pragma once

#include <string>

struct CliOptions {
  std::string model_dir;
  std::string token_input_path;
  std::string token_output_path = "./data/generated_tokens.bin";
  std::string logits_output_path;
  int context_len = 0;
  int max_new_tokens = 64;
  bool run_validation = false;
  bool run_warmup = false;
  bool cuda_profiler_range = false;
  bool help = false;
};

CliOptions parse_args(int argc, char **argv);
void print_help(const CliOptions &defaults);
void validate_options(const CliOptions &options);
