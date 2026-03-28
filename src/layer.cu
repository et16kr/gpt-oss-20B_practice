#include "layer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cuda_common.h"
#include "util.h"

namespace {

constexpr int kBlockSize1D = 256;
constexpr int kLinearBlockCols = 128;
constexpr int kReductionBlockSize = 256;

inline size_t flat_rows(Tensor *tensor) {
  CHECK_ERROR(tensor->ndim >= 2, "Tensor must have at least rank 2");
  return tensor->num_elem() / tensor->shape[tensor->ndim - 1];
}

inline size_t last_dim(Tensor *tensor) { return tensor->shape[tensor->ndim - 1]; }

struct YarnParams {
  float factor = 1.0f;
  float attention_factor = 1.0f;
  float low = 0.0f;
  float high = 0.0f;
};

YarnParams build_yarn_params(const GptOssConfig &config, size_t dim) {
  constexpr float kPi = 3.14159265358979323846f;
  CHECK_ERROR((dim % 2) == 0, "RoPE head_dim must be even");

  YarnParams params;
  params.factor = config.rope_scaling_factor;
  if (params.factor > 1.0f) {
    params.attention_factor = 0.1f * logf(params.factor) + 1.0f;
  }

  const float low_rot = config.rope_ntk_beta;
  const float high_rot = config.rope_ntk_alpha;
  const float d_half = (float)(dim / 2);
  params.low = d_half *
               logf((float)config.initial_context_length /
                    (low_rot * 2.0f * kPi)) /
               logf(config.rope_theta);
  params.high = d_half *
                logf((float)config.initial_context_length /
                     (high_rot * 2.0f * kPi)) /
                logf(config.rope_theta);
  params.low = std::max(params.low, 0.0f);
  params.high = std::min(params.high, (float)(dim / 2 - 1));
  return params;
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
inline float rope_ramp_host(float idx, float low, float high) {
  if (low == high) {
    high += 0.001f;
  }
  float ramp = (idx - low) / (high - low);
  if (ramp < 0.0f) {
    ramp = 0.0f;
  }
  if (ramp > 1.0f) {
    ramp = 1.0f;
  }
  return ramp;
}

inline float rope_inv_freq_host(const GptOssConfig &config, const YarnParams &params,
                                size_t dim, size_t idx) {
  const float exponent = (2.0f * (float)idx) / (float)dim;
  const float pos_freq = powf(config.rope_theta, exponent);
  const float extrapolation = 1.0f / pos_freq;
  if (params.factor <= 1.0f) {
    return extrapolation;
  }
  const float interpolation = 1.0f / (params.factor * pos_freq);
  const float ramp = rope_ramp_host((float)idx, params.low, params.high);
  return interpolation * ramp + extrapolation * (1.0f - ramp);
}

std::pair<std::vector<float>, std::vector<float>> build_yarn_cos_sin(
    const GptOssConfig &config, size_t seq_len, size_t dim) {
  const size_t half_dim = dim / 2;
  const YarnParams params = build_yarn_params(config, dim);

  std::vector<float> cos(seq_len * half_dim, 0.0f);
  std::vector<float> sin(seq_len * half_dim, 0.0f);
  for (size_t t = 0; t < seq_len; ++t) {
    for (size_t i = 0; i < half_dim; ++i) {
      const float angle = (float)t * rope_inv_freq_host(config, params, dim, i);
      cos[t * half_dim + i] = cosf(angle) * params.attention_factor;
      sin[t * half_dim + i] = sinf(angle) * params.attention_factor;
    }
  }
  return {std::move(cos), std::move(sin)};
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }
#endif

#define CUDA_LAUNCH_CHECK() CHECK_CUDA(cudaGetLastError())

__device__ inline float rope_ramp_device(float idx, float low, float high) {
  if (low == high) {
    high += 0.001f;
  }
  float ramp = (idx - low) / (high - low);
  if (ramp < 0.0f) {
    ramp = 0.0f;
  }
  if (ramp > 1.0f) {
    ramp = 1.0f;
  }
  return ramp;
}

__device__ inline float rope_inv_freq_device(float rope_theta, float factor, float low,
                                             float high, size_t dim, size_t idx) {
  const float exponent = (2.0f * (float)idx) / (float)dim;
  const float pos_freq = powf(rope_theta, exponent);
  const float extrapolation = 1.0f / pos_freq;
  if (factor <= 1.0f) {
    return extrapolation;
  }
  const float interpolation = 1.0f / (factor * pos_freq);
  const float ramp = rope_ramp_device((float)idx, low, high);
  return interpolation * ramp + extrapolation * (1.0f - ramp);
}

__global__ void embedding_lookup_kernel(const int32_t *tokens,
                                        const uint16_t *embedding, void *output,
                                        TensorDType output_dtype,
                                        size_t num_tokens, size_t hidden,
                                        size_t vocab_size) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = num_tokens * hidden;
  if (idx >= total) {
    return;
  }

  const size_t token_slot = idx / hidden;
  const size_t h = idx % hidden;
  const int32_t token_id = tokens[token_slot];
  if (token_id < 0 || token_id >= (int32_t)vocab_size) {
    cuda_common::store_tensor_value(output, output_dtype, idx, 0.0f);
    return;
  }
  cuda_common::store_tensor_value(
      output, output_dtype, idx,
      cuda_common::bf16_to_float(embedding[(size_t)token_id * hidden + h]));
}

__global__ void rmsnorm_kernel(const void *input, TensorDType input_dtype,
                               const void *weight, TensorDType weight_dtype,
                               void *output, TensorDType output_dtype,
                               size_t rows, size_t cols, float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float shared[kReductionBlockSize];
  float sum = 0.0f;
  for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        cuda_common::load_tensor_value(input, input_dtype, row * cols + col);
    sum = fmaf(value, value, sum);
  }
  shared[threadIdx.x] = sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    }
    __syncthreads();
  }

  const float scale = rsqrtf(shared[0] / (float)cols + eps);
  for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const size_t index = row * cols + col;
    const float value = cuda_common::load_tensor_value(input, input_dtype, index);
    const float w = cuda_common::load_tensor_value(weight, weight_dtype, col);
    cuda_common::store_tensor_value(output, output_dtype, index, value * scale * w);
  }
}

__global__ void linear_kernel(const void *input, TensorDType input_dtype,
                              const void *weight, TensorDType weight_dtype,
                              const void *bias, TensorDType bias_dtype,
                              void *output, TensorDType output_dtype,
                              size_t rows, size_t in_dim, size_t out_dim,
                              bool has_bias) {
  const size_t row = blockIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= out_dim) {
    return;
  }

  float sum = has_bias ? cuda_common::load_tensor_value(bias, bias_dtype, col) : 0.0f;
  for (size_t k = 0; k < in_dim; ++k) {
    const float lhs =
        cuda_common::load_tensor_value(input, input_dtype, row * in_dim + k);
    const float rhs =
        cuda_common::load_tensor_value(weight, weight_dtype, col * in_dim + k);
    sum = fmaf(lhs, rhs, sum);
  }
  cuda_common::store_tensor_value(output, output_dtype, row * out_dim + col, sum);
}

__global__ void split_q_heads_grouped_kernel(const uint16_t *input,
                                             uint16_t *output, size_t B, size_t T,
                                             size_t num_kv_heads, size_t q_per_kv,
                                             size_t head_dim) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * num_kv_heads * q_per_kv * T * head_dim;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t d = tmp % head_dim;
  tmp /= head_dim;
  const size_t t = tmp % T;
  tmp /= T;
  const size_t qm = tmp % q_per_kv;
  tmp /= q_per_kv;
  const size_t kv = tmp % num_kv_heads;
  const size_t b = tmp / num_kv_heads;

  const size_t src_head = kv * q_per_kv + qm;
  const size_t src =
      (b * T + t) * (num_kv_heads * q_per_kv * head_dim) + src_head * head_dim + d;
  output[idx] = input[src];
}

__global__ void split_kv_heads_kernel(const uint16_t *input, uint16_t *output,
                                      size_t B, size_t T, size_t num_kv_heads,
                                      size_t head_dim) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * num_kv_heads * T * head_dim;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t d = tmp % head_dim;
  tmp /= head_dim;
  const size_t t = tmp % T;
  tmp /= T;
  const size_t kv = tmp % num_kv_heads;
  const size_t b = tmp / num_kv_heads;

  const size_t src = (b * T + t) * (num_kv_heads * head_dim) + kv * head_dim + d;
  output[idx] = input[src];
}

__global__ void apply_yarn_q_kernel(uint16_t *q, size_t B, size_t KV, size_t QM,
                                    size_t T, size_t D, float rope_theta,
                                    float factor, float low, float high,
                                    float attention_factor) {
  const size_t half = D / 2;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * KV * QM * T * half;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t d = tmp % half;
  tmp /= half;
  const size_t t = tmp % T;
  tmp /= T;
  const size_t qm = tmp % QM;
  tmp /= QM;
  const size_t kv = tmp % KV;
  const size_t b = tmp / KV;

  const size_t base = ((((b * KV + kv) * QM + qm) * T + t) * D);
  const float angle =
      (float)t * rope_inv_freq_device(rope_theta, factor, low, high, D, d);
  const float c = cosf(angle) * attention_factor;
  const float s = sinf(angle) * attention_factor;
  const float x0 = cuda_common::bf16_to_float(q[base + d]);
  const float x1 = cuda_common::bf16_to_float(q[base + d + half]);
  q[base + d] = cuda_common::float_to_bf16(x0 * c - x1 * s);
  q[base + d + half] = cuda_common::float_to_bf16(x1 * c + x0 * s);
}

__global__ void apply_yarn_k_kernel(uint16_t *k, size_t B, size_t KV, size_t T,
                                    size_t D, float rope_theta, float factor,
                                    float low, float high,
                                    float attention_factor) {
  const size_t half = D / 2;
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * KV * T * half;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t d = tmp % half;
  tmp /= half;
  const size_t t = tmp % T;
  tmp /= T;
  const size_t kv = tmp % KV;
  const size_t b = tmp / KV;

  const size_t base = (((b * KV + kv) * T + t) * D);
  const float angle =
      (float)t * rope_inv_freq_device(rope_theta, factor, low, high, D, d);
  const float c = cosf(angle) * attention_factor;
  const float s = sinf(angle) * attention_factor;
  const float x0 = cuda_common::bf16_to_float(k[base + d]);
  const float x1 = cuda_common::bf16_to_float(k[base + d + half]);
  k[base + d] = cuda_common::float_to_bf16(x0 * c - x1 * s);
  k[base + d + half] = cuda_common::float_to_bf16(x1 * c + x0 * s);
}

__global__ void attention_scores_with_sink_kernel(
    const uint16_t *q, const uint16_t *k, const void *sinks, TensorDType sink_dtype,
    float *scores, size_t B, size_t KV, size_t QM, size_t T, size_t D) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t rows = B * KV * QM * T;
  if (row >= rows) {
    return;
  }

  size_t tmp = row;
  const size_t tq = tmp % T;
  tmp /= T;
  const size_t qm = tmp % QM;
  tmp /= QM;
  const size_t kv = tmp % KV;
  const size_t b = tmp / KV;

  const size_t q_base = ((((b * KV + kv) * QM + qm) * T + tq) * D);
  const size_t score_base = row * (T + 1);
  for (size_t tk = 0; tk < T; ++tk) {
    const size_t k_base = (((b * KV + kv) * T + tk) * D);
    float sum = 0.0f;
    for (size_t d = 0; d < D; ++d) {
      sum = fmaf(cuda_common::bf16_to_float(q[q_base + d]),
                 cuda_common::bf16_to_float(k[k_base + d]), sum);
    }
    scores[score_base + tk] = sum;
  }
  scores[score_base + T] =
      cuda_common::load_tensor_value(sinks, sink_dtype, kv * QM + qm);
}

__global__ void scale_mask_softmax_kernel(const float *scores, float *probs,
                                          const int32_t *lengths,
                                          size_t B, size_t KV, size_t QM,
                                          size_t T, size_t head_dim,
                                          size_t sliding_window) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t rows = B * KV * QM * T;
  if (row >= rows) {
    return;
  }

  const float scale = rsqrtf((float)head_dim);
  size_t tmp = row;
  const size_t tq = tmp % T;
  tmp /= T;
  tmp /= QM;
  const size_t b = tmp / KV;

  const size_t valid_t = lengths != nullptr ? (size_t)lengths[b] : T;
  const size_t row_base = row * (T + 1);
  for (size_t i = 0; i < T + 1; ++i) {
    probs[row_base + i] = 0.0f;
  }
  if (tq >= valid_t) {
    return;
  }

  size_t tk_begin = 0;
  if (sliding_window > 0 && tq + 1 > sliding_window) {
    tk_begin = tq + 1 - sliding_window;
  }
  const size_t tk_end = min(tq, valid_t - 1);

  float row_max = -1.0e30f;
  for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
    row_max = max(row_max, scores[row_base + tk] * scale);
  }
  row_max = max(row_max, scores[row_base + T]);

  float sum = 0.0f;
  for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
    const float e = expf(scores[row_base + tk] * scale - row_max);
    probs[row_base + tk] = e;
    sum += e;
  }
  const float sink_exp = expf(scores[row_base + T] - row_max);
  probs[row_base + T] = sink_exp;
  sum += sink_exp;

  if (sum > 0.0f) {
    const float inv_sum = 1.0f / sum;
    for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
      probs[row_base + tk] *= inv_sum;
    }
    probs[row_base + T] *= inv_sum;
  }
}

__global__ void attention_context_grouped_kernel(const float *probs,
                                                 const uint16_t *v, void *context,
                                                 TensorDType context_dtype,
                                                 size_t B, size_t KV, size_t QM,
                                                 size_t T, size_t D) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * KV * QM * T * D;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t d = tmp % D;
  tmp /= D;
  const size_t tq = tmp % T;
  tmp /= T;
  const size_t qm = tmp % QM;
  tmp /= QM;
  const size_t kv = tmp % KV;
  const size_t b = tmp / KV;

  const size_t prob_base = ((((b * KV + kv) * QM + qm) * T + tq) * (T + 1));
  float sum = 0.0f;
  for (size_t tk = 0; tk < T; ++tk) {
    const size_t v_base = (((b * KV + kv) * T + tk) * D);
    sum = fmaf(probs[prob_base + tk], cuda_common::bf16_to_float(v[v_base + d]),
               sum);
  }
  cuda_common::store_tensor_value(context, context_dtype, idx, sum);
}

__global__ void merge_heads_grouped_kernel(const void *context,
                                           TensorDType context_dtype,
                                           void *merged,
                                           TensorDType merged_dtype, size_t B,
                                           size_t KV, size_t QM, size_t T,
                                           size_t D) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = B * KV * QM * T * D;
  if (idx >= total) {
    return;
  }

  size_t tmp = idx;
  const size_t d = tmp % D;
  tmp /= D;
  const size_t t = tmp % T;
  tmp /= T;
  const size_t qm = tmp % QM;
  tmp /= QM;
  const size_t kv = tmp % KV;
  const size_t b = tmp / KV;

  const size_t head = kv * QM + qm;
  const size_t dst = (b * T + t) * (KV * QM * D) + head * D + d;
  const float value = cuda_common::load_tensor_value(context, context_dtype, idx);
  cuda_common::store_tensor_value(merged, merged_dtype, dst, value);
}

__global__ void residual_add_kernel(const void *input, TensorDType input_dtype,
                                    const void *addend, TensorDType addend_dtype,
                                    void *output, TensorDType output_dtype, size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  const float lhs = cuda_common::load_tensor_value(input, input_dtype, idx);
  const float rhs = cuda_common::load_tensor_value(addend, addend_dtype, idx);
  cuda_common::store_tensor_value(output, output_dtype, idx, lhs + rhs);
}

__global__ void topk_experts_kernel(const void *router_logits,
                                    TensorDType router_dtype, int32_t *indices,
                                    float *weights, size_t rows, size_t experts,
                                    size_t topk) {
  const size_t row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) {
    return;
  }

  constexpr int kMaxTopK = 8;
  float best_values[kMaxTopK];
  int32_t best_indices[kMaxTopK];
  for (size_t i = 0; i < topk; ++i) {
    best_values[i] = -1.0e30f;
    best_indices[i] = -1;
  }

  const size_t base = row * experts;
  for (size_t e = 0; e < experts; ++e) {
    float value = cuda_common::load_tensor_value(router_logits, router_dtype, base + e);
    if (!isfinite(value)) {
      value = -1.0e30f;
    }
    for (size_t k = 0; k < topk; ++k) {
      if (value > best_values[k] ||
          (value == best_values[k] && (int32_t)e < best_indices[k])) {
        for (size_t shift = topk - 1; shift > k; --shift) {
          best_values[shift] = best_values[shift - 1];
          best_indices[shift] = best_indices[shift - 1];
        }
        best_values[k] = value;
        best_indices[k] = (int32_t)e;
        break;
      }
    }
  }

  float row_max = best_values[0];
  float sum = 0.0f;
  for (size_t k = 0; k < topk; ++k) {
    indices[row * topk + k] = best_indices[k];
    const float w = expf(best_values[k] - row_max);
    weights[row * topk + k] = w;
    sum += w;
  }
  const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
  for (size_t k = 0; k < topk; ++k) {
    weights[row * topk + k] *= inv_sum;
  }
}

__global__ void swiglu_clamp_kernel(const uint16_t *input, uint16_t *output,
                                    size_t rows, size_t out_dim, float limit) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * out_dim;
  if (idx >= total) {
    return;
  }

  const size_t row = idx / out_dim;
  const size_t col = idx % out_dim;
  const size_t base = row * out_dim * 2 + col * 2;
  float gate = cuda_common::bf16_to_float(input[base]);
  float up = cuda_common::bf16_to_float(input[base + 1]);
  if (gate > limit) {
    gate = limit;
  }
  if (up > limit) {
    up = limit;
  } else if (up < -limit) {
    up = -limit;
  }
  const float value = (up + 1.0f) * gate * cuda_common::sigmoid(1.702f * gate);
  output[idx] = cuda_common::float_to_bf16(value);
}

}  // namespace

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void EmbeddingLookup(TokenBatch *tokens, Tensor *embedding, Tensor *output) {
  CHECK_ERROR(embedding->ndim == 2, "Embedding tensor must be rank 2");
  CHECK_ERROR(output->shape[0] == tokens->B && output->shape[1] == tokens->T,
              "Embedding output shape mismatch");
  CHECK_ERROR(output->shape[2] == embedding->shape[1],
              "Embedding hidden size mismatch");

  const size_t hidden = embedding->shape[1];
  const size_t vocab_size = embedding->shape[0];

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < tokens->B; ++b) {
    for (size_t t = 0; t < tokens->T; ++t) {
      const int32_t token_id = tokens->buf[b * tokens->T + t];
      CHECK_ERROR(token_id >= 0 && token_id < (int32_t)vocab_size,
                  "Token id %d out of range", token_id);
      const float *src = embedding->buf + (size_t)token_id * hidden;
      float *dst = output->buf + (b * tokens->T + t) * hidden;
      memcpy(dst, src, hidden * sizeof(float));
    }
  }
  commit_tensor_dtype(output);
}
#endif

void EmbeddingLookup_gpu_i32xbf16_to_bf16(DeviceTokenBatch *tokens, Tensor *embedding,
                                          Tensor *output) {
  CHECK_ERROR(embedding->dtype == TensorDType::BF16,
              "Embedding GPU path expects BF16 weights");
  embedding->ensure_gpu();
  output->ensure_gpu();
  const size_t num_tokens = tokens->B * tokens->T;
  const size_t hidden = embedding->shape[1];
  const size_t total = num_tokens * hidden;
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  embedding_lookup_kernel<<<grid, block>>>(
      tokens->buf, (const uint16_t *)embedding->buf, output->buf, output->dtype,
      num_tokens, hidden, embedding->shape[0]);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void RMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps) {
  const size_t rows = flat_rows(input);
  const size_t cols = last_dim(input);
  CHECK_ERROR(weight->ndim == 1 && weight->shape[0] == cols,
              "RMSNorm parameter shape mismatch");
  CHECK_ERROR(output->num_elem() == input->num_elem(),
              "RMSNorm output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * cols;
    float *out = output->buf + row * cols;

    float mean_sq = 0.0f;
    for (size_t col = 0; col < cols; ++col) {
      mean_sq += in[col] * in[col];
    }
    mean_sq /= (float)cols;
    const float scale = rsqrtf(mean_sq + eps);
    for (size_t col = 0; col < cols; ++col) {
      out[col] = in[col] * scale * weight->buf[col];
    }
  }
  commit_tensor_dtype(output);
}
#endif

void RMSNorm_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *weight, Tensor *output,
                                   float eps) {
  weight->ensure_gpu();
  output->ensure_gpu();
  const size_t rows = flat_rows(input);
  const size_t cols = last_dim(input);
  rmsnorm_kernel<<<(unsigned int)rows, kReductionBlockSize>>>(
      input->buf, input->dtype, weight->buf, weight->dtype, output->buf,
      output->dtype, rows, cols, eps);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void LinearBias(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  CHECK_ERROR(weight->ndim == 2, "LinearBias weight must be rank 2");
  CHECK_ERROR(weight->shape[1] == in_dim, "LinearBias input dim mismatch");
  CHECK_ERROR(bias->ndim == 1 && bias->shape[0] == weight->shape[0],
              "LinearBias bias mismatch");

  const size_t out_dim = weight->shape[0];
  CHECK_ERROR(output->num_elem() == rows * out_dim,
              "LinearBias output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * in_dim;
    float *out = output->buf + row * out_dim;
    for (size_t col = 0; col < out_dim; ++col) {
      const float *w = weight->buf + col * in_dim;
      float sum = bias->buf[col];
      for (size_t k = 0; k < in_dim; ++k) {
        sum += in[k] * w[k];
      }
      out[col] = sum;
    }
  }
  commit_tensor_dtype(output);
}
#endif

void LinearBias_gpu_bf16xbf16xbf16_to_bf16(Tensor *input, Tensor *weight,
                                           Tensor *bias, Tensor *output) {
  weight->ensure_gpu();
  bias->ensure_gpu();
  output->ensure_gpu();
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  const size_t out_dim = weight->shape[0];
  const dim3 block(kLinearBlockCols);
  const dim3 grid((unsigned int)cuda_common::ceil_div(out_dim, (size_t)block.x),
                  (unsigned int)rows);
  linear_kernel<<<grid, block>>>(
      input->buf, input->dtype, weight->buf, weight->dtype, bias->buf, bias->dtype,
      output->buf, output->dtype, rows, in_dim, out_dim, true);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void Linear(Tensor *input, Tensor *weight, Tensor *output) {
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  CHECK_ERROR(weight->ndim == 2, "Linear weight must be rank 2");
  CHECK_ERROR(weight->shape[1] == in_dim, "Linear input dim mismatch");

  const size_t out_dim = weight->shape[0];
  CHECK_ERROR(output->num_elem() == rows * out_dim, "Linear output shape mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *in = input->buf + row * in_dim;
    float *out = output->buf + row * out_dim;
    for (size_t col = 0; col < out_dim; ++col) {
      const float *w = weight->buf + col * in_dim;
      float sum = 0.0f;
      for (size_t k = 0; k < in_dim; ++k) {
        sum += in[k] * w[k];
      }
      out[col] = sum;
    }
  }
  commit_tensor_dtype(output);
}
#endif

void Linear_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *weight, Tensor *output) {
  weight->ensure_gpu();
  output->ensure_gpu();
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  const size_t out_dim = weight->shape[0];
  const dim3 block(kLinearBlockCols);
  const dim3 grid((unsigned int)cuda_common::ceil_div(out_dim, (size_t)block.x),
                  (unsigned int)rows);
  linear_kernel<<<grid, block>>>(
      input->buf, input->dtype, weight->buf, weight->dtype, nullptr,
      TensorDType::F32, output->buf, output->dtype, rows, in_dim, out_dim, false);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void SplitQHeadsGrouped(Tensor *input, Tensor *output, size_t num_kv_heads,
                        size_t q_per_kv, size_t head_dim) {
  CHECK_ERROR(input->ndim == 3, "SplitQHeadsGrouped input must be rank 3");
  CHECK_ERROR(output->ndim == 5, "SplitQHeadsGrouped output must be rank 5");
  CHECK_ERROR(output->shape[0] == input->shape[0] && output->shape[3] == input->shape[1],
              "SplitQHeadsGrouped batch/sequence mismatch");
  CHECK_ERROR(output->shape[1] == num_kv_heads && output->shape[2] == q_per_kv &&
                  output->shape[4] == head_dim,
              "SplitQHeadsGrouped head shape mismatch");
  CHECK_ERROR(input->shape[2] == num_kv_heads * q_per_kv * head_dim,
              "SplitQHeadsGrouped hidden size mismatch");

  const size_t B = input->shape[0];
  const size_t T = input->shape[1];

#pragma omp parallel for collapse(4)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < num_kv_heads; ++kv) {
      for (size_t qm = 0; qm < q_per_kv; ++qm) {
        for (size_t t = 0; t < T; ++t) {
          const size_t src_head = kv * q_per_kv + qm;
          const size_t src_base =
              (b * T + t) * (num_kv_heads * q_per_kv * head_dim) + src_head * head_dim;
          const size_t dst_base =
              ((((b * num_kv_heads + kv) * q_per_kv + qm) * T) + t) * head_dim;
          memcpy(output->buf + dst_base, input->buf + src_base, head_dim * sizeof(float));
        }
      }
    }
  }
  commit_tensor_dtype(output);
}
#endif

void SplitQHeadsGrouped_gpu_bf16_to_bf16(Tensor *input, Tensor *output,
                                         size_t num_kv_heads, size_t q_per_kv,
                                         size_t head_dim) {
  output->ensure_gpu();
  const size_t total = input->shape[0] * num_kv_heads * q_per_kv * input->shape[1] *
                       head_dim;
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  split_q_heads_grouped_kernel<<<grid, block>>>(
      (const uint16_t *)input->buf, (uint16_t *)output->buf, input->shape[0],
      input->shape[1], num_kv_heads, q_per_kv, head_dim);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void SplitKVHeads(Tensor *input, Tensor *output, size_t num_kv_heads,
                  size_t head_dim) {
  CHECK_ERROR(input->ndim == 3, "SplitKVHeads input must be rank 3");
  CHECK_ERROR(output->ndim == 4, "SplitKVHeads output must be rank 4");
  CHECK_ERROR(output->shape[0] == input->shape[0] && output->shape[2] == input->shape[1],
              "SplitKVHeads batch/sequence mismatch");
  CHECK_ERROR(output->shape[1] == num_kv_heads && output->shape[3] == head_dim,
              "SplitKVHeads head shape mismatch");
  CHECK_ERROR(input->shape[2] == num_kv_heads * head_dim,
              "SplitKVHeads hidden size mismatch");

  const size_t B = input->shape[0];
  const size_t T = input->shape[1];

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < num_kv_heads; ++kv) {
      for (size_t t = 0; t < T; ++t) {
        const size_t src_base = (b * T + t) * (num_kv_heads * head_dim) + kv * head_dim;
        const size_t dst_base = ((b * num_kv_heads + kv) * T + t) * head_dim;
        memcpy(output->buf + dst_base, input->buf + src_base, head_dim * sizeof(float));
      }
    }
  }
  commit_tensor_dtype(output);
}
#endif

void SplitKVHeads_gpu_bf16_to_bf16(Tensor *input, Tensor *output,
                                   size_t num_kv_heads, size_t head_dim) {
  output->ensure_gpu();
  const size_t total = input->shape[0] * num_kv_heads * input->shape[1] * head_dim;
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  split_kv_heads_kernel<<<grid, block>>>((const uint16_t *)input->buf,
                                         (uint16_t *)output->buf,
                                         input->shape[0], input->shape[1],
                                         num_kv_heads, head_dim);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void ApplyYaRNRoPE(Tensor *q, Tensor *k, const GptOssConfig &config) {
  CHECK_ERROR(q->ndim == 5 && k->ndim == 4, "ApplyYaRNRoPE rank mismatch");
  const size_t B = q->shape[0];
  const size_t KV = q->shape[1];
  const size_t QM = q->shape[2];
  const size_t T = q->shape[3];
  const size_t D = q->shape[4];
  CHECK_ERROR(k->shape[0] == B && k->shape[1] == KV && k->shape[2] == T &&
                  k->shape[3] == D,
              "ApplyYaRNRoPE shape mismatch");

  const size_t half = D / 2;
  const auto cos_sin = build_yarn_cos_sin(config, T, D);
  const std::vector<float> &cos = cos_sin.first;
  const std::vector<float> &sin = cos_sin.second;

#pragma omp parallel for collapse(4)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < KV; ++kv) {
      for (size_t qm = 0; qm < QM; ++qm) {
        for (size_t t = 0; t < T; ++t) {
          float *ptr = q->buf + ((((b * KV + kv) * QM + qm) * T + t) * D);
          for (size_t i = 0; i < half; ++i) {
            const float c = cos[t * half + i];
            const float s = sin[t * half + i];
            const float x0 = ptr[i];
            const float x1 = ptr[i + half];
            ptr[i] = x0 * c - x1 * s;
            ptr[i + half] = x1 * c + x0 * s;
          }
        }
      }
    }
  }

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < KV; ++kv) {
      for (size_t t = 0; t < T; ++t) {
        float *ptr = k->buf + (((b * KV + kv) * T + t) * D);
        for (size_t i = 0; i < half; ++i) {
          const float c = cos[t * half + i];
          const float s = sin[t * half + i];
          const float x0 = ptr[i];
          const float x1 = ptr[i + half];
          ptr[i] = x0 * c - x1 * s;
          ptr[i + half] = x1 * c + x0 * s;
        }
      }
    }
  }
  commit_tensor_dtype(q);
  commit_tensor_dtype(k);
}
#endif

void ApplyYaRNRoPE_gpu_bf16(Tensor *q, Tensor *k, const GptOssConfig &config) {
  const YarnParams params = build_yarn_params(config, q->shape[4]);
  const size_t q_total = q->shape[0] * q->shape[1] * q->shape[2] * q->shape[3] *
                         (q->shape[4] / 2);
  const size_t k_total = k->shape[0] * k->shape[1] * k->shape[2] *
                         (k->shape[3] / 2);
  const dim3 block(kBlockSize1D);
  const dim3 q_grid((unsigned int)cuda_common::ceil_div(q_total, (size_t)block.x));
  const dim3 k_grid((unsigned int)cuda_common::ceil_div(k_total, (size_t)block.x));
  apply_yarn_q_kernel<<<q_grid, block>>>(
      (uint16_t *)q->buf, q->shape[0], q->shape[1], q->shape[2], q->shape[3],
      q->shape[4], config.rope_theta, params.factor, params.low, params.high,
      params.attention_factor);
  CUDA_LAUNCH_CHECK();
  apply_yarn_k_kernel<<<k_grid, block>>>(
      (uint16_t *)k->buf, k->shape[0], k->shape[1], k->shape[2], k->shape[3],
      config.rope_theta, params.factor, params.low, params.high,
      params.attention_factor);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void AttentionScoresWithSink(Tensor *q, Tensor *k, Tensor *sinks, Tensor *scores) {
  CHECK_ERROR(q->ndim == 5 && k->ndim == 4 && scores->ndim == 5,
              "AttentionScoresWithSink rank mismatch");
  const size_t B = q->shape[0];
  const size_t KV = q->shape[1];
  const size_t QM = q->shape[2];
  const size_t T = q->shape[3];
  const size_t D = q->shape[4];

  CHECK_ERROR(k->shape[0] == B && k->shape[1] == KV && k->shape[2] == T &&
                  k->shape[3] == D,
              "AttentionScoresWithSink K shape mismatch");
  CHECK_ERROR(scores->shape[0] == B && scores->shape[1] == KV &&
                  scores->shape[2] == QM && scores->shape[3] == T &&
                  scores->shape[4] == T + 1,
              "AttentionScoresWithSink output shape mismatch");
  CHECK_ERROR(sinks->ndim == 1 && sinks->shape[0] == KV * QM,
              "AttentionScoresWithSink sink shape mismatch");

#pragma omp parallel for collapse(4)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < KV; ++kv) {
      for (size_t qm = 0; qm < QM; ++qm) {
        for (size_t tq = 0; tq < T; ++tq) {
          const size_t score_base = ((((b * KV + kv) * QM + qm) * T + tq) * (T + 1));
          const size_t q_base = ((((b * KV + kv) * QM + qm) * T + tq) * D);
          for (size_t tk = 0; tk < T; ++tk) {
            const size_t k_base = (((b * KV + kv) * T + tk) * D);
            float sum = 0.0f;
            for (size_t d = 0; d < D; ++d) {
              sum += q->buf[q_base + d] * k->buf[k_base + d];
            }
            scores->buf[score_base + tk] = sum;
          }
          scores->buf[score_base + T] = sinks->buf[kv * QM + qm];
        }
      }
    }
  }
}
#endif

void AttentionScoresWithSink_gpu_bf16xbf16xbf16_to_f32(Tensor *q, Tensor *k,
                                                       Tensor *sinks,
                                                       Tensor *scores) {
  sinks->ensure_gpu();
  scores->ensure_gpu();
  const size_t rows = q->shape[0] * q->shape[1] * q->shape[2] * q->shape[3];
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(rows, (size_t)block.x));
  attention_scores_with_sink_kernel<<<grid, block>>>(
      (const uint16_t *)q->buf, (const uint16_t *)k->buf, sinks->buf, sinks->dtype,
      (float *)scores->buf, q->shape[0], q->shape[1], q->shape[2], q->shape[3],
      q->shape[4]);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const TokenBatch *tokens, size_t sliding_window) {
  const size_t B = scores->shape[0];
  const size_t KV = scores->shape[1];
  const size_t QM = scores->shape[2];
  const size_t T = scores->shape[3];
  const float scale = 1.0f / sqrtf((float)head_dim);

#pragma omp parallel for collapse(4)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < KV; ++kv) {
      for (size_t qm = 0; qm < QM; ++qm) {
        for (size_t tq = 0; tq < T; ++tq) {
          const size_t valid_t =
              (tokens != nullptr && tokens->lengths != nullptr) ? (size_t)tokens->lengths[b] : T;
          const size_t row_base = ((((b * KV + kv) * QM + qm) * T + tq) * (T + 1));
          for (size_t i = 0; i < T + 1; ++i) {
            probs->buf[row_base + i] = 0.0f;
          }
          if (tq >= valid_t) {
            continue;
          }

          size_t tk_begin = 0;
          if (sliding_window > 0 && tq + 1 > sliding_window) {
            tk_begin = tq + 1 - sliding_window;
          }
          const size_t tk_end = std::min(tq, valid_t - 1);
          float row_max = -std::numeric_limits<float>::infinity();
          for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
            row_max = std::max(row_max, scores->buf[row_base + tk] * scale);
          }
          row_max = std::max(row_max, scores->buf[row_base + T]);

          float sum = 0.0f;
          for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
            const float e = expf(scores->buf[row_base + tk] * scale - row_max);
            probs->buf[row_base + tk] = e;
            sum += e;
          }
          const float sink_exp = expf(scores->buf[row_base + T] - row_max);
          probs->buf[row_base + T] = sink_exp;
          sum += sink_exp;

          if (sum > 0.0f) {
            for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
              probs->buf[row_base + tk] /= sum;
            }
            probs->buf[row_base + T] /= sum;
          }
        }
      }
    }
  }
}
#endif

void ScaleMaskSoftmax_gpu_f32_to_f32(Tensor *scores, Tensor *probs,
                                     size_t head_dim, const DeviceTokenBatch *tokens,
                                     size_t sliding_window) {
  probs->ensure_gpu();
  const size_t rows = scores->shape[0] * scores->shape[1] * scores->shape[2] *
                      scores->shape[3];
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(rows, (size_t)block.x));
  scale_mask_softmax_kernel<<<grid, block>>>(
      (const float *)scores->buf, (float *)probs->buf,
      tokens != nullptr ? tokens->lengths : nullptr, scores->shape[0],
      scores->shape[1], scores->shape[2], scores->shape[3], head_dim,
      sliding_window);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context) {
  CHECK_ERROR(probs->ndim == 5 && v->ndim == 4 && context->ndim == 5,
              "AttentionContextGrouped rank mismatch");
  const size_t B = probs->shape[0];
  const size_t KV = probs->shape[1];
  const size_t QM = probs->shape[2];
  const size_t T = probs->shape[3];
  const size_t D = v->shape[3];

#pragma omp parallel for collapse(4)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < KV; ++kv) {
      for (size_t qm = 0; qm < QM; ++qm) {
        for (size_t tq = 0; tq < T; ++tq) {
          const size_t prob_base = ((((b * KV + kv) * QM + qm) * T + tq) * (T + 1));
          const size_t out_base = ((((b * KV + kv) * QM + qm) * T + tq) * D);
          for (size_t d = 0; d < D; ++d) {
            float sum = 0.0f;
            for (size_t tk = 0; tk < T; ++tk) {
              const size_t v_base = (((b * KV + kv) * T + tk) * D);
              sum += probs->buf[prob_base + tk] * v->buf[v_base + d];
            }
            context->buf[out_base + d] = sum;
          }
        }
      }
    }
  }
  commit_tensor_dtype(context);
}
#endif

void AttentionContextGrouped_gpu_f32xbf16_to_bf16(Tensor *probs, Tensor *v,
                                                  Tensor *context) {
  context->ensure_gpu();
  const size_t total = probs->shape[0] * probs->shape[1] * probs->shape[2] *
                       probs->shape[3] * v->shape[3];
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  attention_context_grouped_kernel<<<grid, block>>>(
      (const float *)probs->buf, (const uint16_t *)v->buf, context->buf,
      context->dtype, probs->shape[0], probs->shape[1], probs->shape[2],
      probs->shape[3], v->shape[3]);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void MergeHeadsGrouped(Tensor *context, Tensor *merged) {
  CHECK_ERROR(context->ndim == 5 && merged->ndim == 3, "MergeHeadsGrouped rank mismatch");
  const size_t B = context->shape[0];
  const size_t KV = context->shape[1];
  const size_t QM = context->shape[2];
  const size_t T = context->shape[3];
  const size_t D = context->shape[4];
  CHECK_ERROR(merged->shape[0] == B && merged->shape[1] == T &&
                  merged->shape[2] == KV * QM * D,
              "MergeHeadsGrouped output shape mismatch");

#pragma omp parallel for collapse(4)
  for (size_t b = 0; b < B; ++b) {
    for (size_t kv = 0; kv < KV; ++kv) {
      for (size_t qm = 0; qm < QM; ++qm) {
        for (size_t t = 0; t < T; ++t) {
          const size_t head = kv * QM + qm;
          const size_t src_base = ((((b * KV + kv) * QM + qm) * T + t) * D);
          const size_t dst_base = (b * T + t) * (KV * QM * D) + head * D;
          memcpy(merged->buf + dst_base, context->buf + src_base, D * sizeof(float));
        }
      }
    }
  }
  commit_tensor_dtype(merged);
}
#endif

void MergeHeadsGrouped_gpu_bf16_to_bf16(Tensor *context, Tensor *merged) {
  merged->ensure_gpu();
  const size_t total = context->shape[0] * context->shape[1] * context->shape[2] *
                       context->shape[3] * context->shape[4];
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  merge_heads_grouped_kernel<<<grid, block>>>(
      context->buf, context->dtype, merged->buf, merged->dtype,
      context->shape[0], context->shape[1], context->shape[2], context->shape[3],
      context->shape[4]);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->num_elem() == addend->num_elem() &&
                  input->num_elem() == output->num_elem(),
              "ResidualAdd shape mismatch");

#pragma omp parallel for
  for (size_t i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i] + addend->buf[i];
  }
  commit_tensor_dtype(output);
}
#endif

void ResidualAdd_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *addend,
                                       Tensor *output) {
  output->ensure_gpu();
  const size_t n = input->num_elem();
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(n, (size_t)block.x));
  residual_add_kernel<<<grid, block>>>(
      input->buf, input->dtype, addend->buf, addend->dtype, output->buf,
      output->dtype, n);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void TopKExperts(Tensor *router_logits, ExpertSelection *selection) {
  CHECK_ERROR(router_logits->ndim == 3, "TopKExperts input must be rank 3");
  const size_t B = router_logits->shape[0];
  const size_t T = router_logits->shape[1];
  const size_t E = router_logits->shape[2];
  CHECK_ERROR(selection->B == B && selection->T == T,
              "TopKExperts selection shape mismatch");
  CHECK_ERROR(selection->K > 0 && selection->K <= E,
              "TopKExperts K mismatch");

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      const float *row = router_logits->buf + (b * T + t) * E;
      std::vector<std::pair<float, int32_t>> values;
      values.reserve(E);
      for (size_t e = 0; e < E; ++e) {
        float value = row[e];
        if (!std::isfinite(value)) {
          value = -std::numeric_limits<float>::infinity();
        }
        values.push_back({value, (int32_t)e});
      }
      std::partial_sort(
          values.begin(), values.begin() + selection->K, values.end(),
          [](const std::pair<float, int32_t> &lhs, const std::pair<float, int32_t> &rhs) {
            if (lhs.first != rhs.first) {
              return lhs.first > rhs.first;
            }
            return lhs.second < rhs.second;
          });

      float row_max = values[0].first;
      float sum = 0.0f;
      for (size_t k = 0; k < selection->K; ++k) {
        selection->index(b, t, k) = values[k].second;
        const float w = expf(values[k].first - row_max);
        selection->weight(b, t, k) = w;
        sum += w;
      }
      for (size_t k = 0; k < selection->K; ++k) {
        selection->weight(b, t, k) /= sum;
      }
    }
  }
}
#endif

void TopKExperts_gpu_bf16_to_i32f32(Tensor *router_logits,
                                    ExpertSelection *selection) {
  selection->ensure_gpu();
  const size_t rows = router_logits->shape[0] * router_logits->shape[1];
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(rows, (size_t)block.x));
  topk_experts_kernel<<<grid, block>>>(
      router_logits->buf, router_logits->dtype, selection->indices,
      selection->weights, rows, router_logits->shape[2], selection->K);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void SwiGLUClamp(Tensor *input, Tensor *output, float limit) {
  CHECK_ERROR(input->num_elem() == output->num_elem() * 2,
              "SwiGLUClamp input/output shape mismatch");
  const size_t rows = flat_rows(output);
  const size_t out_dim = last_dim(output);
  CHECK_ERROR(last_dim(input) == out_dim * 2, "SwiGLUClamp last-dim mismatch");

#pragma omp parallel for
  for (size_t row = 0; row < rows; ++row) {
    const float *src = input->buf + row * out_dim * 2;
    float *dst = output->buf + row * out_dim;
    for (size_t i = 0; i < out_dim; ++i) {
      float x_glu = src[2 * i];
      float x_linear = src[2 * i + 1];
      if (x_glu > limit) {
        x_glu = limit;
      }
      if (x_linear > limit) {
        x_linear = limit;
      } else if (x_linear < -limit) {
        x_linear = -limit;
      }
      dst[i] = x_glu * sigmoidf(1.702f * x_glu) * (x_linear + 1.0f);
    }
  }
  commit_tensor_dtype(output);
}
#endif

void SwiGLUClamp_gpu_bf16_to_bf16(Tensor *input, Tensor *output, float limit) {
  output->ensure_gpu();
  const size_t rows = flat_rows(output);
  const size_t out_dim = last_dim(output);
  const size_t total = rows * out_dim;
  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)cuda_common::ceil_div(total, (size_t)block.x));
  swiglu_clamp_kernel<<<grid, block>>>((const uint16_t *)input->buf,
                                       (uint16_t *)output->buf, rows, out_dim,
                                       limit);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void LMHead(Tensor *input, Tensor *weight, Tensor *output) { Linear(input, weight, output); }
#endif

void Linear_gpu_bf16xbf16_to_f32(Tensor *input, Tensor *weight, Tensor *output) {
  weight->ensure_gpu();
  output->ensure_gpu();
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  const size_t out_dim = weight->shape[0];
  const dim3 block(kLinearBlockCols);
  const dim3 grid((unsigned int)cuda_common::ceil_div(out_dim, (size_t)block.x),
                  (unsigned int)rows);
  linear_kernel<<<grid, block>>>(
      input->buf, input->dtype, weight->buf, weight->dtype, nullptr, TensorDType::F32,
      output->buf, output->dtype, rows, in_dim, out_dim, false);
  CUDA_LAUNCH_CHECK();
}

void LMHead_gpu_bf16xbf16_to_f32(Tensor *input, Tensor *weight, Tensor *output) {
  Linear_gpu_bf16xbf16_to_f32(input, weight, output);
}
