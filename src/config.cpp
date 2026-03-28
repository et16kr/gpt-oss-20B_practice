#include "gpt_oss_config.h"

#include <fstream>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

namespace {

template <typename T>
T read_required(const nlohmann::json &obj, const char *key) {
  if (!obj.contains(key)) {
    throw std::runtime_error(std::string("missing config key: ") + key);
  }
  return obj.at(key).get<T>();
}

template <typename T>
T read_optional(const nlohmann::json &obj, const char *key, const T &fallback) {
  if (!obj.contains(key) || obj.at(key).is_null()) {
    return fallback;
  }
  return obj.at(key).get<T>();
}

std::vector<int> read_eos_ids(const nlohmann::json &obj) {
  if (!obj.contains("eos_token_id")) {
    return {};
  }
  const nlohmann::json &value = obj.at("eos_token_id");
  if (value.is_array()) {
    return value.get<std::vector<int>>();
  }
  return {value.get<int>()};
}

}  // namespace

bool GptOssConfig::uses_sliding_window(size_t layer_idx) const {
  if (layer_types.empty()) {
    return (layer_idx % 2) == 0 && sliding_window > 0;
  }
  if (layer_idx >= layer_types.size()) {
    return false;
  }
  return layer_types[layer_idx] == "sliding_attention" && sliding_window > 0;
}

bool GptOssConfig::is_eos(int token_id) const {
  for (int eos_id : eos_token_ids) {
    if (token_id == eos_id) {
      return true;
    }
  }
  return false;
}

int GptOssConfig::primary_eos_token_id() const {
  return eos_token_ids.empty() ? -1 : eos_token_ids.front();
}

GptOssConfig load_gpt_oss_config(const char *path) {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error(std::string("failed to open config: ") + path);
  }

  nlohmann::json json;
  input >> json;

  GptOssConfig cfg;
  cfg.vocab_size = read_required<size_t>(json, "vocab_size");
  cfg.hidden_size = read_required<size_t>(json, "hidden_size");
  cfg.intermediate_size = read_required<size_t>(json, "intermediate_size");
  cfg.num_hidden_layers = read_required<size_t>(json, "num_hidden_layers");
  cfg.num_attention_heads = read_required<size_t>(json, "num_attention_heads");
  cfg.num_key_value_heads = read_required<size_t>(json, "num_key_value_heads");
  cfg.num_local_experts =
      read_optional<size_t>(json, "num_local_experts",
                            read_optional<size_t>(json, "num_experts", 0));
  cfg.experts_per_token =
      read_optional<size_t>(json, "experts_per_token",
                            read_optional<size_t>(json, "num_experts_per_tok", 0));
  cfg.head_dim = read_required<size_t>(json, "head_dim");
  cfg.max_position_embeddings =
      read_optional<size_t>(json, "max_position_embeddings", 0);
  cfg.initial_context_length =
      read_required<size_t>(json, "initial_context_length");
  cfg.sliding_window = read_required<size_t>(json, "sliding_window");
  cfg.rms_norm_eps = read_optional<float>(json, "rms_norm_eps", 1.0e-5f);
  cfg.rope_theta = read_optional<float>(json, "rope_theta", 150000.0f);
  cfg.swiglu_limit = read_optional<float>(json, "swiglu_limit", 7.0f);
  cfg.bos_token_id = read_optional<int>(json, "bos_token_id", 199998);
  cfg.pad_token_id = read_optional<int>(json, "pad_token_id", -1);
  cfg.eos_token_ids = read_eos_ids(json);
  cfg.layer_types =
      read_optional<std::vector<std::string>>(json, "layer_types", {});

  if (json.contains("rope_scaling") && !json.at("rope_scaling").is_null()) {
    const nlohmann::json &rope_scaling = json.at("rope_scaling");
    cfg.rope_scaling_factor = read_optional<float>(rope_scaling, "factor", 1.0f);
    cfg.rope_ntk_alpha =
        read_optional<float>(rope_scaling, "beta_slow",
                             read_optional<float>(json, "rope_ntk_alpha", 1.0f));
    cfg.rope_ntk_beta =
        read_optional<float>(rope_scaling, "beta_fast",
                             read_optional<float>(json, "rope_ntk_beta", 32.0f));
  } else {
    cfg.rope_scaling_factor =
        read_optional<float>(json, "rope_scaling_factor", 1.0f);
    cfg.rope_ntk_alpha = read_optional<float>(json, "rope_ntk_alpha", 1.0f);
    cfg.rope_ntk_beta = read_optional<float>(json, "rope_ntk_beta", 32.0f);
  }

  if (cfg.max_position_embeddings == 0) {
    cfg.max_position_embeddings =
        cfg.initial_context_length * (size_t)cfg.rope_scaling_factor;
  }

  if (cfg.vocab_size == 0 || cfg.hidden_size == 0 || cfg.intermediate_size == 0 ||
      cfg.num_hidden_layers == 0 || cfg.num_attention_heads == 0 ||
      cfg.num_key_value_heads == 0 || cfg.num_local_experts == 0 ||
      cfg.experts_per_token == 0 || cfg.head_dim == 0) {
    throw std::runtime_error("config contains zero-sized dimensions");
  }
  if (cfg.hidden_size % cfg.num_attention_heads != 0) {
    throw std::runtime_error("hidden_size must be divisible by num_attention_heads");
  }
  if (cfg.num_attention_heads % cfg.num_key_value_heads != 0) {
    throw std::runtime_error(
        "num_attention_heads must be divisible by num_key_value_heads");
  }
  if ((cfg.head_dim % 2) != 0) {
    throw std::runtime_error("head_dim must be even for RoPE");
  }
  return cfg;
}
