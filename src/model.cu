#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "layer.h"
#include "safetensors_loader.h"
#include "util.h"

namespace {

constexpr float kFp4Lut[16] = {
    +0.0f, +0.5f, +1.0f, +1.5f, +2.0f, +3.0f, +4.0f, +6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

constexpr TensorDType kHiddenActivationDType = TensorDType::BF16;
constexpr TensorDType kAttentionAccumDType = TensorDType::F32;

Activation *make_activation(const std::vector<size_t> &shape, TensorDType dtype) {
  return new Activation(shape, dtype);
}

GptOssConfig config_;

Parameter *tok_embeddings = nullptr;
std::vector<Parameter *> input_norm_weight;
std::vector<Parameter *> q_proj_weight;
std::vector<Parameter *> q_proj_bias;
std::vector<Parameter *> k_proj_weight;
std::vector<Parameter *> k_proj_bias;
std::vector<Parameter *> v_proj_weight;
std::vector<Parameter *> v_proj_bias;
std::vector<Parameter *> o_proj_weight;
std::vector<Parameter *> o_proj_bias;
std::vector<Parameter *> attn_sinks;
std::vector<Parameter *> post_attn_norm_weight;
std::vector<Parameter *> router_weight;
std::vector<Parameter *> router_bias;
std::vector<QuantizedExpertMatrix> gate_up_proj;
std::vector<QuantizedExpertMatrix> down_proj;
Parameter *final_norm_weight = nullptr;
Parameter *lm_head_weight = nullptr;

Activation *x = nullptr;
Activation *residual = nullptr;
Activation *norm_buf = nullptr;
Activation *q_proj = nullptr;
Activation *k_proj = nullptr;
Activation *v_proj = nullptr;
Activation *q = nullptr;
Activation *k = nullptr;
Activation *v = nullptr;
Activation *att_scores = nullptr;
Activation *att_probs = nullptr;
Activation *context = nullptr;
Activation *merged = nullptr;
Activation *attn_out = nullptr;
Activation *router_logits = nullptr;
Activation *gate_up_buf = nullptr;
Activation *gated_buf = nullptr;
Activation *expert_out_buf = nullptr;
Activation *moe_out = nullptr;
Activation *final_norm = nullptr;
ExpertSelection *expert_selection = nullptr;

size_t current_batch = 0;
size_t current_seq = 0;
TokenBatch *current_tokens = nullptr;

void delete_tensor(Tensor *&tensor) {
  if (tensor != nullptr) {
    delete tensor;
    tensor = nullptr;
  }
}

void delete_tensor_vector(std::vector<Parameter *> &tensors) {
  for (Tensor *tensor : tensors) {
    delete tensor;
  }
  tensors.clear();
}

void delete_selection(ExpertSelection *&selection) {
  if (selection != nullptr) {
    delete selection;
    selection = nullptr;
  }
}

void load_layer_parameters(ShardedSafetensorsLoader *loader, size_t layer_idx) {
  const std::string prefix = "model.layers." + std::to_string(layer_idx) + ".";
  const size_t hidden = config_.hidden_size;
  const size_t q_hidden = config_.num_attention_heads * config_.head_dim;
  const size_t kv_hidden = config_.num_key_value_heads * config_.head_dim;
  const size_t num_experts = config_.num_local_experts;
  const size_t intermediate = config_.intermediate_size;

  input_norm_weight.push_back(
      loader->load_parameter((prefix + "input_layernorm.weight").c_str(), {hidden}));
  q_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.q_proj.weight").c_str(),
                             {q_hidden, hidden}));
  q_proj_bias.push_back(
      loader->load_parameter((prefix + "self_attn.q_proj.bias").c_str(), {q_hidden}));
  k_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.k_proj.weight").c_str(),
                             {kv_hidden, hidden}));
  k_proj_bias.push_back(
      loader->load_parameter((prefix + "self_attn.k_proj.bias").c_str(), {kv_hidden}));
  v_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.v_proj.weight").c_str(),
                             {kv_hidden, hidden}));
  v_proj_bias.push_back(
      loader->load_parameter((prefix + "self_attn.v_proj.bias").c_str(), {kv_hidden}));
  o_proj_weight.push_back(
      loader->load_parameter((prefix + "self_attn.o_proj.weight").c_str(),
                             {hidden, q_hidden}));
  o_proj_bias.push_back(
      loader->load_parameter((prefix + "self_attn.o_proj.bias").c_str(), {hidden}));
  attn_sinks.push_back(
      loader->load_parameter((prefix + "self_attn.sinks").c_str(), {q_hidden / config_.head_dim}));
  post_attn_norm_weight.push_back(
      loader->load_parameter((prefix + "post_attention_layernorm.weight").c_str(),
                             {hidden}));
  router_weight.push_back(
      loader->load_parameter((prefix + "mlp.router.weight").c_str(),
                             {num_experts, hidden}));
  router_bias.push_back(
      loader->load_parameter((prefix + "mlp.router.bias").c_str(), {num_experts}));
  gate_up_proj.push_back(loader->load_quantized_expert_matrix(
      (prefix + "mlp.experts.gate_up_proj_blocks").c_str(),
      (prefix + "mlp.experts.gate_up_proj_scales").c_str(),
      (prefix + "mlp.experts.gate_up_proj_bias").c_str(), num_experts,
      intermediate * 2, hidden));
  down_proj.push_back(loader->load_quantized_expert_matrix(
      (prefix + "mlp.experts.down_proj_blocks").c_str(),
      (prefix + "mlp.experts.down_proj_scales").c_str(),
      (prefix + "mlp.experts.down_proj_bias").c_str(), num_experts, hidden,
      intermediate));
}

void load_parameters(const char *model_dir) {
  char config_path[512];
  snprintf(config_path, sizeof(config_path), "%s/config.json", model_dir);

  config_ = load_gpt_oss_config(config_path);
  ShardedSafetensorsLoader loader(model_dir);

  tok_embeddings =
      loader.load_parameter("model.embed_tokens.weight",
                            {config_.vocab_size, config_.hidden_size});

  for (size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    load_layer_parameters(&loader, layer);
  }

  final_norm_weight =
      loader.load_parameter("model.norm.weight", {config_.hidden_size});
  lm_head_weight =
      loader.load_parameter("lm_head.weight", {config_.vocab_size, config_.hidden_size});
}

void free_parameters() {
  delete_tensor(tok_embeddings);
  delete_tensor_vector(input_norm_weight);
  delete_tensor_vector(q_proj_weight);
  delete_tensor_vector(q_proj_bias);
  delete_tensor_vector(k_proj_weight);
  delete_tensor_vector(k_proj_bias);
  delete_tensor_vector(v_proj_weight);
  delete_tensor_vector(v_proj_bias);
  delete_tensor_vector(o_proj_weight);
  delete_tensor_vector(o_proj_bias);
  delete_tensor_vector(attn_sinks);
  delete_tensor_vector(post_attn_norm_weight);
  delete_tensor_vector(router_weight);
  delete_tensor_vector(router_bias);
  gate_up_proj.clear();
  down_proj.clear();
  delete_tensor(final_norm_weight);
  delete_tensor(lm_head_weight);
}

inline float dot_quantized_row(const QuantizedExpertMatrix &matrix, size_t expert_idx,
                               size_t row_idx, const float *input) {
  const uint8_t *blocks = matrix.row_blocks(expert_idx, row_idx);
  const uint8_t *scales = matrix.row_scales(expert_idx, row_idx);
  float sum = matrix.row_bias(expert_idx, row_idx);
  const size_t values_per_group = matrix.bytes_per_block * 2;

  for (size_t g = 0; g < matrix.groups; ++g) {
    const int exponent = (int)scales[g] - 127;
    const uint8_t *group_blocks = blocks + g * matrix.bytes_per_block;
    const size_t base = g * values_per_group;
    for (size_t i = 0; i < matrix.bytes_per_block; ++i) {
      const uint8_t value = group_blocks[i];
      const float lo = ldexpf(kFp4Lut[value & 0x0F], exponent);
      const float hi = ldexpf(kFp4Lut[(value >> 4) & 0x0F], exponent);
      sum += lo * input[base + 2 * i];
      sum += hi * input[base + 2 * i + 1];
    }
  }
  return sum;
}

void ExpertGateUp(Tensor *input, const ExpertSelection *selection, size_t layer_idx,
                  Tensor *output) {
  CHECK_ERROR(input->ndim == 3 && output->ndim == 4, "ExpertGateUp rank mismatch");
  const QuantizedExpertMatrix &matrix = gate_up_proj[layer_idx];
  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t hidden = input->shape[2];
  const size_t K = selection->K;
  CHECK_ERROR(hidden == matrix.in_dim, "ExpertGateUp hidden size mismatch");
  CHECK_ERROR(output->shape[0] == B && output->shape[1] == T && output->shape[2] == K &&
                  output->shape[3] == matrix.out_dim,
              "ExpertGateUp output shape mismatch");

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t k_idx = 0; k_idx < K; ++k_idx) {
        float *dst = output->buf + (((b * T + t) * K + k_idx) * matrix.out_dim);
        if (current_tokens != nullptr && t >= (size_t)current_tokens->lengths[b]) {
          memset(dst, 0, matrix.out_dim * sizeof(float));
          continue;
        }
        const float *src = input->buf + (b * T + t) * hidden;
        const size_t expert_idx = (size_t)selection->index(b, t, k_idx);
        for (size_t row = 0; row < matrix.out_dim; ++row) {
          dst[row] = dot_quantized_row(matrix, expert_idx, row, src);
        }
      }
    }
  }
}

void ExpertDown(Tensor *input, const ExpertSelection *selection, size_t layer_idx,
                Tensor *output) {
  CHECK_ERROR(input->ndim == 4 && output->ndim == 4, "ExpertDown rank mismatch");
  const QuantizedExpertMatrix &matrix = down_proj[layer_idx];
  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t K = input->shape[2];
  const size_t intermediate = input->shape[3];
  CHECK_ERROR(intermediate == matrix.in_dim, "ExpertDown intermediate mismatch");
  CHECK_ERROR(output->shape[0] == B && output->shape[1] == T && output->shape[2] == K &&
                  output->shape[3] == matrix.out_dim,
              "ExpertDown output shape mismatch");

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t k_idx = 0; k_idx < K; ++k_idx) {
        float *dst = output->buf + (((b * T + t) * K + k_idx) * matrix.out_dim);
        if (current_tokens != nullptr && t >= (size_t)current_tokens->lengths[b]) {
          memset(dst, 0, matrix.out_dim * sizeof(float));
          continue;
        }
        const float *src = input->buf + (((b * T + t) * K + k_idx) * intermediate);
        const size_t expert_idx = (size_t)selection->index(b, t, k_idx);
        for (size_t row = 0; row < matrix.out_dim; ++row) {
          dst[row] = dot_quantized_row(matrix, expert_idx, row, src);
        }
      }
    }
  }
}

void WeightedExpertReduce(Tensor *expert_outputs, const ExpertSelection *selection,
                          Tensor *output) {
  CHECK_ERROR(expert_outputs->ndim == 4 && output->ndim == 3,
              "WeightedExpertReduce rank mismatch");
  const size_t B = output->shape[0];
  const size_t T = output->shape[1];
  const size_t hidden = output->shape[2];
  const size_t K = selection->K;
  CHECK_ERROR(expert_outputs->shape[0] == B && expert_outputs->shape[1] == T &&
                  expert_outputs->shape[2] == K && expert_outputs->shape[3] == hidden,
              "WeightedExpertReduce shape mismatch");

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      float *dst = output->buf + (b * T + t) * hidden;
      if (current_tokens != nullptr && t >= (size_t)current_tokens->lengths[b]) {
        memset(dst, 0, hidden * sizeof(float));
        continue;
      }
      for (size_t h = 0; h < hidden; ++h) {
        float sum = 0.0f;
        for (size_t k_idx = 0; k_idx < K; ++k_idx) {
          const float *src =
              expert_outputs->buf + (((b * T + t) * K + k_idx) * hidden);
          sum += selection->weight(b, t, k_idx) * src[h];
        }
        dst[h] = sum;
      }
    }
  }
}

void transformer_block_cpu(size_t layer_idx) {
  RMSNorm(x, input_norm_weight[layer_idx], norm_buf, config_.rms_norm_eps);

  LinearBias(norm_buf, q_proj_weight[layer_idx], q_proj_bias[layer_idx], q_proj);
  LinearBias(norm_buf, k_proj_weight[layer_idx], k_proj_bias[layer_idx], k_proj);
  LinearBias(norm_buf, v_proj_weight[layer_idx], v_proj_bias[layer_idx], v_proj);

  SplitQHeadsGrouped(q_proj, q, config_.num_key_value_heads, config_.q_per_kv(),
                     config_.head_dim);
  SplitKVHeads(k_proj, k, config_.num_key_value_heads, config_.head_dim);
  SplitKVHeads(v_proj, v, config_.num_key_value_heads, config_.head_dim);
  ApplyYaRNRoPE(q, k, config_);
  AttentionScoresWithSink(q, k, attn_sinks[layer_idx], att_scores);
  ScaleMaskSoftmax(att_scores, att_probs, config_.head_dim, current_tokens,
                   config_.uses_sliding_window(layer_idx) ? config_.sliding_window : 0);
  AttentionContextGrouped(att_probs, v, context);
  MergeHeadsGrouped(context, merged);
  LinearBias(merged, o_proj_weight[layer_idx], o_proj_bias[layer_idx], attn_out);
  ResidualAdd(x, attn_out, residual);
  std::swap(x, residual);

  RMSNorm(x, post_attn_norm_weight[layer_idx], norm_buf, config_.rms_norm_eps);
  LinearBias(norm_buf, router_weight[layer_idx], router_bias[layer_idx], router_logits);
  TopKExperts(router_logits, expert_selection);
  ExpertGateUp(norm_buf, expert_selection, layer_idx, gate_up_buf);
  SwiGLUClamp(gate_up_buf, gated_buf, config_.swiglu_limit);
  ExpertDown(gated_buf, expert_selection, layer_idx, expert_out_buf);
  WeightedExpertReduce(expert_out_buf, expert_selection, moe_out);
  ResidualAdd(x, moe_out, residual);
  std::swap(x, residual);
}

void transformer_block_gpu(size_t layer_idx) {
  RMSNorm_gpu_bf16xbf16_to_bf16(x, input_norm_weight[layer_idx], norm_buf,
                                config_.rms_norm_eps);

  LinearBias_gpu_bf16xbf16xbf16_to_bf16(norm_buf, q_proj_weight[layer_idx],
                                        q_proj_bias[layer_idx], q_proj);
  LinearBias_gpu_bf16xbf16xbf16_to_bf16(norm_buf, k_proj_weight[layer_idx],
                                        k_proj_bias[layer_idx], k_proj);
  LinearBias_gpu_bf16xbf16xbf16_to_bf16(norm_buf, v_proj_weight[layer_idx],
                                        v_proj_bias[layer_idx], v_proj);

  SplitQHeadsGrouped_gpu_bf16_to_bf16(q_proj, q, config_.num_key_value_heads,
                                      config_.q_per_kv(), config_.head_dim);
  SplitKVHeads_gpu_bf16_to_bf16(k_proj, k, config_.num_key_value_heads,
                                config_.head_dim);
  SplitKVHeads_gpu_bf16_to_bf16(v_proj, v, config_.num_key_value_heads,
                                config_.head_dim);
  ApplyYaRNRoPE_gpu_bf16(q, k, config_);
  AttentionScoresWithSink_gpu_bf16xbf16xbf16_to_f32(q, k, attn_sinks[layer_idx],
                                                    att_scores);
  ScaleMaskSoftmax_gpu_f32_to_f32(
      att_scores, att_probs, config_.head_dim, current_tokens,
      config_.uses_sliding_window(layer_idx) ? config_.sliding_window : 0);
  AttentionContextGrouped_gpu_f32xbf16_to_bf16(att_probs, v, context);
  MergeHeadsGrouped_gpu_bf16_to_bf16(context, merged);
  LinearBias_gpu_bf16xbf16xbf16_to_bf16(merged, o_proj_weight[layer_idx],
                                        o_proj_bias[layer_idx], attn_out);
  ResidualAdd_gpu_bf16xbf16_to_bf16(x, attn_out, residual);
  std::swap(x, residual);

  RMSNorm_gpu_bf16xbf16_to_bf16(x, post_attn_norm_weight[layer_idx], norm_buf,
                                config_.rms_norm_eps);
  LinearBias_gpu_bf16xbf16xbf16_to_bf16(norm_buf, router_weight[layer_idx],
                                        router_bias[layer_idx], router_logits);
  TopKExperts_gpu_bf16_to_i32f32(router_logits, expert_selection);
  ExpertGateUp(norm_buf, expert_selection, layer_idx, gate_up_buf);
  SwiGLUClamp_gpu_bf16_to_bf16(gate_up_buf, gated_buf, config_.swiglu_limit);
  ExpertDown(gated_buf, expert_selection, layer_idx, expert_out_buf);
  WeightedExpertReduce(expert_out_buf, expert_selection, moe_out);
  ResidualAdd_gpu_bf16xbf16_to_bf16(x, moe_out, residual);
  std::swap(x, residual);
}

void gpt_oss_forward_cpu(TokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == config_.vocab_size,
              "Logits tensor shape mismatch");

  EmbeddingLookup(tokens, tok_embeddings, x);
  for (size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    transformer_block_cpu(layer);
  }
  RMSNorm(x, final_norm_weight, final_norm, config_.rms_norm_eps);
  LMHead(final_norm, lm_head_weight, logits);
}

void gpt_oss_forward_gpu(TokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == config_.vocab_size,
              "Logits tensor shape mismatch");

  EmbeddingLookup_gpu_i32xbf16_to_bf16(tokens, tok_embeddings, x);
  for (size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    transformer_block_gpu(layer);
  }
  RMSNorm_gpu_bf16xbf16_to_bf16(x, final_norm_weight, final_norm,
                                config_.rms_norm_eps);
  LMHead_gpu_bf16xbf16_to_f32(final_norm, lm_head_weight, logits);
}

}  // namespace

TokenBatch load_tokens(const char *path) {
  FILE *f = fopen(path, "rb");
  CHECK_ERROR(f != nullptr, "Failed to open token file %s", path);

  int32_t B = 0;
  int32_t T = 0;
  CHECK_ERROR(fread(&B, sizeof(int32_t), 1, f) == 1,
              "Failed to read batch size from %s", path);
  CHECK_ERROR(fread(&T, sizeof(int32_t), 1, f) == 1,
              "Failed to read sequence length from %s", path);
  CHECK_ERROR(B > 0 && T > 0, "Invalid token shape in %s", path);

  CHECK_ERROR(fseek(f, 0, SEEK_END) == 0, "Failed to seek %s", path);
  long file_size = ftell(f);
  CHECK_ERROR(file_size >= 0, "Failed to stat %s", path);
  rewind(f);
  CHECK_ERROR(fread(&B, sizeof(int32_t), 1, f) == 1,
              "Failed to read batch size from %s", path);
  CHECK_ERROR(fread(&T, sizeof(int32_t), 1, f) == 1,
              "Failed to read sequence length from %s", path);

  TokenBatch batch((size_t)B, (size_t)T);
  const size_t tokens_bytes = (size_t)B * (size_t)T * sizeof(int32_t);
  const size_t lengths_bytes = (size_t)B * sizeof(int32_t);
  const size_t header_bytes = 2 * sizeof(int32_t);
  const size_t file_bytes = (size_t)file_size;

  bool has_lengths = false;
  if (file_bytes == header_bytes + tokens_bytes) {
    has_lengths = false;
  } else if (file_bytes == header_bytes + lengths_bytes + tokens_bytes) {
    has_lengths = true;
  } else {
    CHECK_ERROR(false, "Unsupported token file size for %s", path);
  }

  if (has_lengths) {
    CHECK_ERROR(fread(batch.lengths, sizeof(int32_t), (size_t)B, f) == (size_t)B,
                "Failed to read lengths from %s", path);
  } else {
    for (size_t b = 0; b < batch.B; ++b) {
      batch.lengths[b] = (int32_t)batch.T;
    }
  }

  const size_t expected = (size_t)B * (size_t)T;
  CHECK_ERROR(fread(batch.buf, sizeof(int32_t), expected, f) == expected,
              "Failed to read token ids from %s", path);
  const int trailing = fgetc(f);
  fclose(f);
  CHECK_ERROR(trailing == EOF, "Unexpected trailing bytes in token file %s", path);
  for (size_t b = 0; b < batch.B; ++b) {
    CHECK_ERROR(batch.lengths[b] > 0 && batch.lengths[b] <= (int32_t)batch.T,
                "Invalid sequence length %d in %s", batch.lengths[b], path);
  }
  return batch;
}

void initialize_model(const char *model_dir) { load_parameters(model_dir); }

void alloc_activations(size_t batch_size, size_t seq_len) {
  CHECK_ERROR(batch_size > 0 && seq_len > 0, "Activation shape must be positive");
  CHECK_ERROR(seq_len <= config_.max_position_embeddings,
              "Sequence length %zu exceeds max_position_embeddings %zu", seq_len,
              config_.max_position_embeddings);

  free_activations();

  current_batch = batch_size;
  current_seq = seq_len;

  const size_t hidden = config_.hidden_size;
  const size_t q_hidden = config_.num_attention_heads * config_.head_dim;
  const size_t kv_hidden = config_.num_key_value_heads * config_.head_dim;
  const size_t q_per_kv = config_.q_per_kv();
  const size_t intermediate = config_.intermediate_size;
  const size_t experts_per_token = config_.experts_per_token;

  x = make_activation({batch_size, seq_len, hidden}, kHiddenActivationDType);
  residual = make_activation({batch_size, seq_len, hidden}, kHiddenActivationDType);
  norm_buf = make_activation({batch_size, seq_len, hidden}, kHiddenActivationDType);
  q_proj = make_activation({batch_size, seq_len, q_hidden}, kHiddenActivationDType);
  k_proj = make_activation({batch_size, seq_len, kv_hidden}, kHiddenActivationDType);
  v_proj = make_activation({batch_size, seq_len, kv_hidden}, kHiddenActivationDType);
  q = make_activation({batch_size, config_.num_key_value_heads, q_per_kv, seq_len,
                       config_.head_dim},
                      kHiddenActivationDType);
  k = make_activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim},
      kHiddenActivationDType);
  v = make_activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim},
      kHiddenActivationDType);
  att_scores = make_activation(
      {batch_size, config_.num_key_value_heads, q_per_kv, seq_len, seq_len + 1},
      kAttentionAccumDType);
  att_probs = make_activation(
      {batch_size, config_.num_key_value_heads, q_per_kv, seq_len, seq_len + 1},
      kAttentionAccumDType);
  context = make_activation({batch_size, config_.num_key_value_heads, q_per_kv, seq_len,
                             config_.head_dim},
                            kHiddenActivationDType);
  merged = make_activation({batch_size, seq_len, q_hidden}, kHiddenActivationDType);
  attn_out = make_activation({batch_size, seq_len, hidden}, kHiddenActivationDType);
  router_logits =
      make_activation({batch_size, seq_len, config_.num_local_experts},
                      kHiddenActivationDType);
  gate_up_buf = make_activation(
      {batch_size, seq_len, experts_per_token, intermediate * 2},
      kHiddenActivationDType);
  gated_buf = make_activation(
      {batch_size, seq_len, experts_per_token, intermediate},
      kHiddenActivationDType);
  expert_out_buf = make_activation(
      {batch_size, seq_len, experts_per_token, hidden},
      kHiddenActivationDType);
  moe_out = make_activation({batch_size, seq_len, hidden}, kHiddenActivationDType);
  final_norm = make_activation({batch_size, seq_len, hidden}, kHiddenActivationDType);
  expert_selection = new ExpertSelection(batch_size, seq_len, experts_per_token);
}

void gpt_oss_forward(TokenBatch *tokens, Tensor *logits) {
  current_tokens = tokens;
  gpt_oss_forward_gpu(tokens, logits);
  current_tokens = nullptr;
}

void validate_against_cpu(TokenBatch *tokens, Tensor *logits_gpu) {
  Tensor reference({tokens->B, tokens->T, config_.vocab_size});
  current_tokens = tokens;
  gpt_oss_forward_cpu(tokens, &reference);
  current_tokens = nullptr;

  const int diff = validate_buffer(logits_gpu->buf, reference.buf, reference.num_elem(),
                                   1.0e-3f, 1.0e-3f);
  if (diff < 0) {
    printf("Validation: PASSED\n");
    return;
  }

  printf("Validation: FAILED\n");
  printf("First mismatch at index %d: output=%f reference=%f\n", diff,
         logits_gpu->buf[diff], reference.buf[diff]);
  EXIT(EXIT_FAILURE);
}

void finalize_model() { free_parameters(); }

void free_activations() {
  delete_tensor(x);
  delete_tensor(residual);
  delete_tensor(norm_buf);
  delete_tensor(q_proj);
  delete_tensor(k_proj);
  delete_tensor(v_proj);
  delete_tensor(q);
  delete_tensor(k);
  delete_tensor(v);
  delete_tensor(att_scores);
  delete_tensor(att_probs);
  delete_tensor(context);
  delete_tensor(merged);
  delete_tensor(attn_out);
  delete_tensor(router_logits);
  delete_tensor(gate_up_buf);
  delete_tensor(gated_buf);
  delete_tensor(expert_out_buf);
  delete_tensor(moe_out);
  delete_tensor(final_norm);
  delete_selection(expert_selection);
  current_batch = 0;
  current_seq = 0;
}

const GptOssConfig &model_config() { return config_; }
