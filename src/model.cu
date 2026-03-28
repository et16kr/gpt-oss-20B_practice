#include "model.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cuda_common.h"
#include "layer.h"
#include "safetensors_loader.h"
#include "util.h"

namespace {

constexpr TensorDType kHiddenActivationDType = TensorDType::BF16;
constexpr TensorDType kAttentionAccumDType = TensorDType::F32;
constexpr int kMoeBlockSize = 128;
constexpr int kGenerationBlockSize = 256;

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

#define CUDA_LAUNCH_CHECK() CHECK_CUDA(cudaGetLastError())

__device__ inline float fp4_value(uint8_t code) {
  switch (code & 0x0F) {
    case 0x0:
      return +0.0f;
    case 0x1:
      return +0.5f;
    case 0x2:
      return +1.0f;
    case 0x3:
      return +1.5f;
    case 0x4:
      return +2.0f;
    case 0x5:
      return +3.0f;
    case 0x6:
      return +4.0f;
    case 0x7:
      return +6.0f;
    case 0x8:
      return -0.0f;
    case 0x9:
      return -0.5f;
    case 0xA:
      return -1.0f;
    case 0xB:
      return -1.5f;
    case 0xC:
      return -2.0f;
    case 0xD:
      return -3.0f;
    case 0xE:
      return -4.0f;
    default:
      return -6.0f;
  }
}

__global__ void expert_gate_up_kernel(const uint16_t *input, const int32_t *lengths,
                                      const int32_t *selection_indices,
                                      const uint8_t *blocks, const uint8_t *scales,
                                      const void *bias, TensorDType bias_dtype,
                                      uint16_t *output, size_t B, size_t T,
                                      size_t hidden, size_t K, size_t out_dim,
                                      size_t num_experts, size_t groups,
                                      size_t bytes_per_block, size_t input_elems,
                                      size_t blocks_bytes, size_t scales_elems,
                                      size_t bias_elems) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * T * K * out_dim;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t row = tmp % out_dim;
  tmp /= out_dim;
  const size_t k_idx = tmp % K;
  tmp /= K;
  const size_t t = tmp % T;
  const size_t b = tmp / T;

  if (lengths != nullptr && t >= (size_t)lengths[b]) {
    output[idx] = 0;
    return;
  }

  const size_t expert_idx = (size_t)selection_indices[(b * T + t) * K + k_idx];
  if (expert_idx >= num_experts) {
    output[idx] = 0;
    return;
  }
  const size_t row_offset = expert_idx * out_dim + row;
  if (row_offset >= bias_elems) {
    output[idx] = 0;
    return;
  }
  float sum = cuda_common::load_tensor_value(bias, bias_dtype, row_offset);
  const size_t values_per_group = bytes_per_block * 2;
  const size_t input_base = (b * T + t) * hidden;
  for (size_t g = 0; g < groups; ++g) {
    const size_t scale_index = row_offset * groups + g;
    if (scale_index >= scales_elems) {
      output[idx] = 0;
      return;
    }
    const size_t block_offset = scale_index * bytes_per_block;
    if (block_offset + bytes_per_block > blocks_bytes) {
      output[idx] = 0;
      return;
    }
    const int exponent = (int)scales[scale_index] - 127;
    const uint8_t *group_blocks = blocks + block_offset;
    const size_t base = g * values_per_group;
    for (size_t i = 0; i < bytes_per_block; ++i) {
      const size_t lo_index = input_base + base + 2 * i;
      const size_t hi_index = lo_index + 1;
      if (hi_index >= input_elems) {
        output[idx] = 0;
        return;
      }
      const uint8_t value = group_blocks[i];
      const float lo = ldexpf(fp4_value(value & 0x0F), exponent);
      const float hi = ldexpf(fp4_value((value >> 4) & 0x0F), exponent);
      sum += lo * cuda_common::bf16_to_float(input[lo_index]);
      sum += hi * cuda_common::bf16_to_float(input[hi_index]);
    }
  }
  output[idx] = cuda_common::float_to_bf16(sum);
}

__global__ void expert_down_kernel(const uint16_t *input, const int32_t *lengths,
                                   const int32_t *selection_indices,
                                   const uint8_t *blocks, const uint8_t *scales,
                                   const void *bias, TensorDType bias_dtype,
                                   uint16_t *output, size_t B, size_t T,
                                   size_t K, size_t in_dim, size_t out_dim,
                                   size_t num_experts, size_t groups,
                                   size_t bytes_per_block, size_t input_elems,
                                   size_t blocks_bytes, size_t scales_elems,
                                   size_t bias_elems) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * T * K * out_dim;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t row = tmp % out_dim;
  tmp /= out_dim;
  const size_t k_idx = tmp % K;
  tmp /= K;
  const size_t t = tmp % T;
  const size_t b = tmp / T;

  if (lengths != nullptr && t >= (size_t)lengths[b]) {
    output[idx] = 0;
    return;
  }

  const size_t expert_idx = (size_t)selection_indices[(b * T + t) * K + k_idx];
  if (expert_idx >= num_experts) {
    output[idx] = 0;
    return;
  }
  const size_t row_offset = expert_idx * out_dim + row;
  if (row_offset >= bias_elems) {
    output[idx] = 0;
    return;
  }
  float sum = cuda_common::load_tensor_value(bias, bias_dtype, row_offset);
  const size_t values_per_group = bytes_per_block * 2;
  const size_t input_base = ((b * T + t) * K + k_idx) * in_dim;
  for (size_t g = 0; g < groups; ++g) {
    const size_t scale_index = row_offset * groups + g;
    if (scale_index >= scales_elems) {
      output[idx] = 0;
      return;
    }
    const size_t block_offset = scale_index * bytes_per_block;
    if (block_offset + bytes_per_block > blocks_bytes) {
      output[idx] = 0;
      return;
    }
    const int exponent = (int)scales[scale_index] - 127;
    const uint8_t *group_blocks = blocks + block_offset;
    const size_t base = g * values_per_group;
    for (size_t i = 0; i < bytes_per_block; ++i) {
      const size_t lo_index = input_base + base + 2 * i;
      const size_t hi_index = lo_index + 1;
      if (hi_index >= input_elems) {
        output[idx] = 0;
        return;
      }
      const uint8_t value = group_blocks[i];
      const float lo = ldexpf(fp4_value(value & 0x0F), exponent);
      const float hi = ldexpf(fp4_value((value >> 4) & 0x0F), exponent);
      sum += lo * cuda_common::bf16_to_float(input[lo_index]);
      sum += hi * cuda_common::bf16_to_float(input[hi_index]);
    }
  }
  output[idx] = cuda_common::float_to_bf16(sum);
}

__global__ void weighted_expert_reduce_kernel(const uint16_t *expert_outputs,
                                              const int32_t *lengths,
                                              const float *selection_weights,
                                              uint16_t *output, size_t B,
                                              size_t T, size_t K,
                                              size_t hidden) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * T * hidden;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t h = tmp % hidden;
  tmp /= hidden;
  const size_t t = tmp % T;
  const size_t b = tmp / T;

  if (lengths != nullptr && t >= (size_t)lengths[b]) {
    output[idx] = 0;
    return;
  }

  float sum = 0.0f;
  for (size_t k_idx = 0; k_idx < K; ++k_idx) {
    const float weight = selection_weights[(b * T + t) * K + k_idx];
    const size_t src =
        (((b * T + t) * K + k_idx) * hidden) + h;
    sum += weight * cuda_common::bf16_to_float(expert_outputs[src]);
  }
  output[idx] = cuda_common::float_to_bf16(sum);
}

__global__ void argmax_last_token_kernel(const float *logits, const int32_t *lengths,
                                         int32_t *next_tokens, size_t B, size_t T,
                                         size_t vocab_size) {
  const size_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) {
    return;
  }

  const size_t valid = (size_t)lengths[b];
  if (valid == 0 || valid > T) {
    next_tokens[b] = 0;
    return;
  }

  const float *row = logits + ((b * T) + (valid - 1)) * vocab_size;
  size_t best = 0;
  float best_value = row[0];
  for (size_t i = 1; i < vocab_size; ++i) {
    if (row[i] > best_value) {
      best_value = row[i];
      best = i;
    }
  }
  next_tokens[b] = (int32_t)best;
}

__device__ bool is_eos_device(int32_t token_id, const int32_t *eos_ids, size_t eos_count) {
  for (size_t i = 0; i < eos_count; ++i) {
    if (token_id == eos_ids[i]) {
      return true;
    }
  }
  return false;
}

__global__ void append_next_tokens_kernel(
    int32_t *window_tokens, int32_t *window_lengths, size_t T, const int32_t *next_tokens,
    int32_t *generated_tokens, int32_t *generated_lengths, uint8_t *finished,
    size_t max_new_tokens, const int32_t *eos_ids, size_t eos_count, size_t B) {
  const size_t b = blockIdx.x * blockDim.x + threadIdx.x;
  if (b >= B) {
    return;
  }
  if (finished[b] != 0) {
    return;
  }

  const int32_t next = next_tokens[b];
  if (is_eos_device(next, eos_ids, eos_count)) {
    finished[b] = 1;
    return;
  }

  int32_t generated_len = generated_lengths[b];
  if ((size_t)generated_len < max_new_tokens) {
    generated_tokens[b * max_new_tokens + (size_t)generated_len] = next;
    generated_lengths[b] = generated_len + 1;
  }

  int32_t valid = window_lengths[b];
  if (valid < 0) {
    valid = 0;
  }
  if ((size_t)valid < T) {
    window_tokens[b * T + (size_t)valid] = next;
    window_lengths[b] = valid + 1;
    return;
  }

  for (size_t t = 1; t < T; ++t) {
    window_tokens[b * T + (t - 1)] = window_tokens[b * T + t];
  }
  window_tokens[b * T + (T - 1)] = next;
  window_lengths[b] = (int32_t)T;
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

#if 0  // Legacy CPU reference retained for kernel-study notes only.
inline float dot_quantized_row(const QuantizedExpertMatrix &matrix, size_t expert_idx,
                               size_t row_idx, const float *input) {
  return 0.0f;
}

void ExpertGateUp(Tensor *input, const ExpertSelection *selection, size_t layer_idx,
                  Tensor *output) {
}
#endif

void ExpertGateUp_gpu(Tensor *input, const ExpertSelection *selection,
                      size_t layer_idx, Tensor *output) {
  CHECK_ERROR(input->ndim == 3 && output->ndim == 4, "ExpertGateUp rank mismatch");
  const QuantizedExpertMatrix &matrix = gate_up_proj[layer_idx];
  const size_t total =
      input->shape[0] * input->shape[1] * selection->K * matrix.out_dim;
  output->ensure_gpu();
  const dim3 block(kMoeBlockSize);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  expert_gate_up_kernel<<<grid, block>>>(
      (const uint16_t *)input->buf,
      current_tokens != nullptr ? current_tokens->lengths : nullptr,
      selection->indices, matrix.blocks, matrix.scales, matrix.bias, matrix.bias_dtype,
      (uint16_t *)output->buf, input->shape[0], input->shape[1], input->shape[2],
      selection->K, matrix.out_dim, matrix.num_experts, matrix.groups,
      matrix.bytes_per_block, input->num_elem(), matrix.blocks_bytes,
      matrix.num_experts * matrix.out_dim * matrix.groups,
      matrix.num_experts * matrix.out_dim);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void ExpertDown(Tensor *input, const ExpertSelection *selection, size_t layer_idx,
                Tensor *output) {
}
#endif

void ExpertDown_gpu(Tensor *input, const ExpertSelection *selection, size_t layer_idx,
                    Tensor *output) {
  CHECK_ERROR(input->ndim == 4 && output->ndim == 4, "ExpertDown rank mismatch");
  const QuantizedExpertMatrix &matrix = down_proj[layer_idx];
  const size_t total =
      input->shape[0] * input->shape[1] * input->shape[2] * matrix.out_dim;
  output->ensure_gpu();
  const dim3 block(kMoeBlockSize);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  expert_down_kernel<<<grid, block>>>(
      (const uint16_t *)input->buf,
      current_tokens != nullptr ? current_tokens->lengths : nullptr,
      selection->indices, matrix.blocks, matrix.scales, matrix.bias, matrix.bias_dtype,
      (uint16_t *)output->buf, input->shape[0], input->shape[1], input->shape[2],
      input->shape[3], matrix.out_dim, matrix.num_experts, matrix.groups,
      matrix.bytes_per_block, input->num_elem(), matrix.blocks_bytes,
      matrix.num_experts * matrix.out_dim * matrix.groups,
      matrix.num_experts * matrix.out_dim);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void WeightedExpertReduce(Tensor *expert_outputs, const ExpertSelection *selection,
                          Tensor *output) {
}
#endif

void WeightedExpertReduce_gpu(Tensor *expert_outputs, const ExpertSelection *selection,
                              Tensor *output) {
  CHECK_ERROR(expert_outputs->ndim == 4 && output->ndim == 3,
              "WeightedExpertReduce rank mismatch");
  const size_t total = output->shape[0] * output->shape[1] * output->shape[2];
  output->ensure_gpu();
  const dim3 block(kMoeBlockSize);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  weighted_expert_reduce_kernel<<<grid, block>>>(
      (const uint16_t *)expert_outputs->buf,
      current_tokens != nullptr ? current_tokens->lengths : nullptr,
      selection->weights, (uint16_t *)output->buf, output->shape[0],
      output->shape[1], selection->K, output->shape[2]);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void transformer_block_cpu(size_t layer_idx) {}
#endif

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
  ExpertGateUp_gpu(norm_buf, expert_selection, layer_idx, gate_up_buf);
  SwiGLUClamp_gpu_bf16_to_bf16(gate_up_buf, gated_buf, config_.swiglu_limit);
  ExpertDown_gpu(gated_buf, expert_selection, layer_idx, expert_out_buf);
  WeightedExpertReduce_gpu(expert_out_buf, expert_selection, moe_out);
  ResidualAdd_gpu_bf16xbf16_to_bf16(x, moe_out, residual);
  std::swap(x, residual);
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void gpt_oss_forward_cpu(TokenBatch *tokens, Tensor *logits) {}
#endif

void gpt_oss_forward_gpu(DeviceTokenBatch *tokens, Tensor *logits) {
  CHECK_ERROR(tokens->B == current_batch && tokens->T == current_seq,
              "Token batch shape differs from allocated activations");
  CHECK_ERROR(logits->shape[0] == tokens->B && logits->shape[1] == tokens->T &&
                  logits->shape[2] == config_.vocab_size,
              "Logits tensor shape mismatch");

  logits->ensure_gpu();
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

void gpt_oss_forward(DeviceTokenBatch *tokens, Tensor *logits) {
  current_tokens = tokens;
  gpt_oss_forward_gpu(tokens, logits);
  current_tokens = nullptr;
}

void select_next_tokens(DeviceTokenBatch *tokens, Tensor *logits, int32_t *next_tokens) {
  CHECK_ERROR(logits->dtype == TensorDType::F32,
              "select_next_tokens expects F32 logits");
  const dim3 block(kGenerationBlockSize);
  const dim3 grid((unsigned int)cuda_common::ceil_div(tokens->B, (size_t)block.x));
  argmax_last_token_kernel<<<grid, block>>>(
      (const float *)logits->buf, tokens->lengths, next_tokens, tokens->B, tokens->T,
      logits->shape[2]);
  CUDA_LAUNCH_CHECK();
}

void append_next_tokens(DeviceTokenBatch *tokens, const int32_t *next_tokens,
                        int32_t *generated_tokens, int32_t *generated_lengths,
                        uint8_t *finished, size_t max_new_tokens) {
  const dim3 block(kGenerationBlockSize);
  const dim3 grid((unsigned int)cuda_common::ceil_div(tokens->B, (size_t)block.x));
  append_next_tokens_kernel<<<grid, block>>>(
      tokens->buf, tokens->lengths, tokens->T, next_tokens, generated_tokens,
      generated_lengths, finished, max_new_tokens, eos_token_ids_gpu, eos_token_count,
      tokens->B);
  CUDA_LAUNCH_CHECK();
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
  current_tokens = nullptr;
}

const GptOssConfig &model_config() { return config_; }
