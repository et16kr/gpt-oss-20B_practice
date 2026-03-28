#include "model.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "layer.h"
#include "safetensors_loader.h"
#include "util.h"

namespace {

constexpr TensorDType kResidualActivationDType = TensorDType::BF16;
constexpr TensorDType kAttentionActivationDType = TensorDType::BF16;
constexpr TensorDType kRouterActivationDType = TensorDType::BF16;
constexpr TensorDType kMoeIntermediateDType = TensorDType::BF16;
constexpr TensorDType kAttentionAccumDType = TensorDType::F32;
constexpr int kGenerationBlockSize = 256;

inline size_t ceil_div(size_t n, size_t d) { return (n + d - 1) / d; }

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
Activation *moe_norm_buf = nullptr;
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
DeviceTokenBatch *current_tokens = nullptr;
int32_t *eos_token_ids_gpu = nullptr;
size_t eos_token_count = 0;

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

#define CUDA_LAUNCH_CHECK()            \
  do {                                \
    CHECK_CUDA(cudaGetLastError());   \
    CHECK_CUDA(cudaDeviceSynchronize()); \
  } while (0)

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

  eos_token_count = config_.eos_token_ids.size();
  if (eos_token_count > 0) {
    CHECK_CUDA(cudaMalloc((void **)&eos_token_ids_gpu,
                          eos_token_count * sizeof(int32_t)));
    CHECK_CUDA(cudaMemcpy(eos_token_ids_gpu, config_.eos_token_ids.data(),
                          eos_token_count * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
  }
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
  for (QuantizedExpertMatrix &matrix : gate_up_proj) {
    matrix.free_gpu();
  }
  for (QuantizedExpertMatrix &matrix : down_proj) {
    matrix.free_gpu();
  }
  gate_up_proj.clear();
  down_proj.clear();
  delete_tensor(final_norm_weight);
  delete_tensor(lm_head_weight);
  if (eos_token_ids_gpu != nullptr) {
    CHECK_CUDA(cudaFree(eos_token_ids_gpu));
    eos_token_ids_gpu = nullptr;
  }
  eos_token_count = 0;
}

void transformer_block(size_t layer_idx) {
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
  ScaleMaskSoftmax(
      att_scores, att_probs, config_.head_dim, current_tokens,
      config_.uses_sliding_window(layer_idx) ? config_.sliding_window : 0);
  AttentionContextGrouped(att_probs, v, context);
  MergeHeadsGrouped(context, merged);
  LinearBias(merged, o_proj_weight[layer_idx], o_proj_bias[layer_idx], attn_out);
  ResidualAdd(x, attn_out, residual);
  std::swap(x, residual);

  RMSNorm(x, post_attn_norm_weight[layer_idx], moe_norm_buf, config_.rms_norm_eps);
  LinearBias(moe_norm_buf, router_weight[layer_idx], router_bias[layer_idx], router_logits);
  TopKExperts(router_logits, expert_selection);
  ExpertGateUp(moe_norm_buf, expert_selection, gate_up_proj[layer_idx], current_tokens,
               gate_up_buf);
  SwiGLUClamp(gate_up_buf, gated_buf, config_.swiglu_limit);
  ExpertDown(gated_buf, expert_selection, down_proj[layer_idx], current_tokens,
             expert_out_buf);
  WeightedExpertReduce(expert_out_buf, expert_selection, current_tokens, moe_out);
  ResidualAdd(x, moe_out, residual);
  std::swap(x, residual);
}

void gpt_oss_forward_impl(DeviceTokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == config_.vocab_size,
              "Logits tensor shape mismatch");

  logits->ensure_gpu();
  EmbeddingLookup(tokens, tok_embeddings, x);
  for (size_t layer = 0; layer < config_.num_hidden_layers; ++layer) {
    transformer_block(layer);
  }
  RMSNorm(x, final_norm_weight, final_norm, config_.rms_norm_eps);
  LMHead(final_norm, lm_head_weight, logits);
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

  x = make_activation({batch_size, seq_len, hidden}, kResidualActivationDType);
  residual = make_activation({batch_size, seq_len, hidden}, kResidualActivationDType);
  norm_buf = make_activation({batch_size, seq_len, hidden}, kResidualActivationDType);
  moe_norm_buf = make_activation({batch_size, seq_len, hidden}, kAttentionActivationDType);
  q_proj = make_activation({batch_size, seq_len, q_hidden}, kAttentionActivationDType);
  k_proj = make_activation({batch_size, seq_len, kv_hidden}, kAttentionActivationDType);
  v_proj = make_activation({batch_size, seq_len, kv_hidden}, kAttentionActivationDType);
  q = make_activation({batch_size, config_.num_key_value_heads, q_per_kv, seq_len,
                       config_.head_dim},
                      kAttentionActivationDType);
  k = make_activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim},
      kAttentionActivationDType);
  v = make_activation(
      {batch_size, config_.num_key_value_heads, seq_len, config_.head_dim},
      kAttentionActivationDType);
  att_scores = make_activation(
      {batch_size, config_.num_key_value_heads, q_per_kv, seq_len, seq_len + 1},
      kAttentionAccumDType);
  att_probs = make_activation(
      {batch_size, config_.num_key_value_heads, q_per_kv, seq_len, seq_len + 1},
      kAttentionAccumDType);
  context = make_activation({batch_size, config_.num_key_value_heads, q_per_kv, seq_len,
                             config_.head_dim},
                            kAttentionActivationDType);
  merged = make_activation({batch_size, seq_len, q_hidden}, kAttentionActivationDType);
  attn_out = make_activation({batch_size, seq_len, hidden}, kResidualActivationDType);
  router_logits =
      make_activation({batch_size, seq_len, config_.num_local_experts},
                      kRouterActivationDType);
  gate_up_buf = make_activation(
      {batch_size, seq_len, experts_per_token, intermediate * 2},
      kMoeIntermediateDType);
  gated_buf = make_activation(
      {batch_size, seq_len, experts_per_token, intermediate},
      kMoeIntermediateDType);
  expert_out_buf = make_activation(
      {batch_size, seq_len, experts_per_token, hidden},
      kMoeIntermediateDType);
  moe_out = make_activation({batch_size, seq_len, hidden}, kResidualActivationDType);
  final_norm = make_activation({batch_size, seq_len, hidden}, kResidualActivationDType);
  expert_selection = new ExpertSelection(batch_size, seq_len, experts_per_token);
}

void gpt_oss_forward(DeviceTokenBatch *tokens, Tensor *logits) {
  current_tokens = tokens;
  gpt_oss_forward_impl(tokens, logits);
  current_tokens = nullptr;
}

void select_next_tokens(DeviceTokenBatch *tokens, Tensor *logits, int32_t *next_tokens) {
  CHECK_ERROR(logits->dtype == TensorDType::F32,
              "select_next_tokens expects F32 logits");

  const float *logit_buf = (const float *)logits->buf;
  const int32_t *length_buf = tokens->lengths;
  int32_t *next_token_buf = next_tokens;
  const size_t batch_size = tokens->B;
  const size_t seq_len = tokens->T;
  const size_t vocab_size = logits->shape[2];

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   const size_t valid = (size_t)length_buf[b];
  //   if (valid == 0 || valid > seq_len) {
  //     next_token_buf[b] = 0;
  //     continue;
  //   }
  //   const float *row = logit_buf + ((b * seq_len) + (valid - 1)) * vocab_size;
  //   size_t best = 0;
  //   float best_value = row[0];
  //   for (size_t i = 1; i < vocab_size; ++i) {
  //     if (row[i] > best_value) {
  //       best_value = row[i];
  //       best = i;
  //     }
  //   }
  //   next_token_buf[b] = (int32_t)best;
  // }
  CUDA_LAUNCH_CHECK();
}

void append_next_tokens(DeviceTokenBatch *tokens, const int32_t *next_tokens,
                        int32_t *generated_tokens, int32_t *generated_lengths,
                        uint8_t *finished, size_t max_new_tokens) {
  const int32_t *next_token_buf = next_tokens;
  const int32_t *eos_buf = eos_token_ids_gpu;
  int32_t *token_buf = tokens->buf;
  int32_t *length_buf = tokens->lengths;
  int32_t *generated_token_buf = generated_tokens;
  int32_t *generated_length_buf = generated_lengths;
  uint8_t *finished_buf = finished;
  const size_t seq_len = tokens->T;
  const size_t eos_count = eos_token_count;
  const size_t batch_size = tokens->B;

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   if (finished_buf[b] != 0) {
  //     continue;
  //   }
  //   const int32_t next = next_token_buf[b];
  //   bool is_eos = false;
  //   for (size_t i = 0; i < eos_count; ++i) {
  //     if (next == eos_buf[i]) {
  //       is_eos = true;
  //       break;
  //     }
  //   }
  //   if (is_eos) {
  //     finished_buf[b] = 1;
  //     continue;
  //   }
  //   const int32_t generated_len = generated_length_buf[b];
  //   if ((size_t)generated_len < max_new_tokens) {
  //     generated_token_buf[b * max_new_tokens + (size_t)generated_len] = next;
  //     generated_length_buf[b] = generated_len + 1;
  //   }
  //   int32_t valid = length_buf[b];
  //   if (valid < 0) {
  //     valid = 0;
  //   }
  //   if ((size_t)valid < seq_len) {
  //     token_buf[b * seq_len + (size_t)valid] = next;
  //     length_buf[b] = valid + 1;
  //     continue;
  //   }
  //   for (size_t t = 1; t < seq_len; ++t) {
  //     token_buf[b * seq_len + (t - 1)] = token_buf[b * seq_len + t];
  //   }
  //   token_buf[b * seq_len + (seq_len - 1)] = next;
  //   length_buf[b] = (int32_t)seq_len;
  // }
  CUDA_LAUNCH_CHECK();
}

void finalize_model() { free_parameters(); }

void free_activations() {
  delete_tensor(x);
  delete_tensor(residual);
  delete_tensor(norm_buf);
  delete_tensor(moe_norm_buf);
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
  current_tokens = nullptr;
}

const GptOssConfig &model_config() { return config_; }
