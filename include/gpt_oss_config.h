#pragma once

#include <cstddef>
#include <string>
#include <vector>

struct GptOssConfig {
  size_t vocab_size = 0;
  size_t hidden_size = 0;
  size_t intermediate_size = 0;
  size_t num_hidden_layers = 0;
  size_t num_attention_heads = 0;
  size_t num_key_value_heads = 0;
  size_t num_local_experts = 0;
  size_t experts_per_token = 0;
  size_t head_dim = 0;
  size_t max_position_embeddings = 0;
  size_t initial_context_length = 0;
  size_t sliding_window = 0;
  float rms_norm_eps = 1.0e-5f;
  float rope_theta = 150000.0f;
  float rope_scaling_factor = 1.0f;
  float rope_ntk_alpha = 1.0f;
  float rope_ntk_beta = 32.0f;
  float swiglu_limit = 7.0f;
  int bos_token_id = 0;
  int pad_token_id = -1;
  std::vector<int> eos_token_ids;
  std::vector<std::string> layer_types;

  size_t q_per_kv() const { return num_attention_heads / num_key_value_heads; }
  bool uses_sliding_window(size_t layer_idx) const;
  bool is_eos(int token_id) const;
  int primary_eos_token_id() const;
};

GptOssConfig load_gpt_oss_config(const char *path);
