#include <cstdio>
#include <stdexcept>

#include "app_options.h"
#include "generation.h"
#include "model.h"

namespace {

void ensure_file_exists(const std::string &path, const char *label) {
  FILE *fp = fopen(path.c_str(), "rb");
  if (fp == nullptr) {
    throw std::runtime_error(std::string(label) + " not found: " + path);
  }
  fclose(fp);
}

}  // namespace

int main(int argc, char **argv) {
  try {
    CliOptions options = parse_args(argc, argv);
    if (options.help) {
      print_help(options);
      return 0;
    }

    validate_options(options);
    ensure_file_exists(options.model_dir + "/config.json", "config");
    ensure_file_exists(options.model_dir + "/model.safetensors.index.json", "model index");

    initialize_model(options.model_dir.c_str());
    if (!options.logits_output_path.empty()) {
      run_forward_only_mode(options);
    } else {
      run_generation_mode(options);
    }
    free_activations();
    finalize_model();
    return 0;
  } catch (const std::exception &ex) {
    std::fprintf(stderr, "fatal: %s\n", ex.what());
    return 1;
  }
}
