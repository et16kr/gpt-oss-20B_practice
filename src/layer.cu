#include "layer.h"

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "safetensors_loader.h"
#include "util.h"

namespace {

constexpr int kBlockSize1D = 256;
constexpr int kLinearTileSize = 16;
constexpr int kReductionBlockSize = 256;
constexpr int kMoeBlockSize = 128;

inline size_t ceil_div(size_t n, size_t d) { return (n + d - 1) / d; }

inline size_t flat_rows(Tensor *tensor) {
  CHECK_ERROR(tensor->ndim >= 2, "Tensor must have at least rank 2");
  return tensor->num_elem() / tensor->shape[tensor->ndim - 1];
}

inline size_t last_dim(Tensor *tensor) { return tensor->shape[tensor->ndim - 1]; }

using Mxfp4PackedByte = QuantizedExpertMatrix::Mxfp4PackedByte;
using Mxfp4Scale = QuantizedExpertMatrix::Mxfp4Scale;

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

inline float bf16_bits_to_float(uint16_t bits) {
  union {
    uint32_t u;
    float f;
  } value = {(uint32_t)bits << 16};
  return value.f;
}

inline float fp4_value_host(uint8_t code) {
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
#endif

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
      float *dst = output->buf + (b * tokens->T + t) * hidden;
      if (token_id < 0 || token_id >= (int32_t)vocab_size) {
        memset(dst, 0, hidden * sizeof(float));
        continue;
      }
      const float *src = embedding->buf + (size_t)token_id * hidden;
      memcpy(dst, src, hidden * sizeof(float));
    }
  }
  commit_tensor_dtype(output);
}
#endif

namespace {

__global__ void embedding_lookup_kernel_i32xbf16_to_bf16(
    const int32_t *tokens, const __nv_bfloat16 *embedding,
    __nv_bfloat16 *output, size_t num_tokens, size_t hidden,
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
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }
  output[idx] = embedding[(size_t)token_id * hidden + h];
}

}  // namespace

void EmbeddingLookup(DeviceTokenBatch *tokens, Tensor *embedding, Tensor *output) {
  CHECK_ERROR(embedding->dtype == TensorDType::BF16,
              "EmbeddingLookup expects BF16 weights");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "EmbeddingLookup expects BF16 output");
  embedding->ensure_gpu();
  output->ensure_gpu();

  const int32_t *token_buf = tokens->buf;
  const __nv_bfloat16 *embedding_buf = (const __nv_bfloat16 *)embedding->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t num_tokens = tokens->B * tokens->T;
  const size_t hidden = embedding->shape[1];
  const size_t vocab_size = embedding->shape[0];
  const size_t total = num_tokens * hidden;

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  embedding_lookup_kernel_i32xbf16_to_bf16<<<grid, block>>>(
      token_buf, embedding_buf, output_buf, num_tokens, hidden, vocab_size);
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

namespace {

__global__ void rmsnorm_kernel_bf16xbf16_to_bf16(const __nv_bfloat16 *input,
                                                 const __nv_bfloat16 *weight,
                                                 __nv_bfloat16 *output,
                                                 size_t rows, size_t cols,
                                                 float eps) {
  const size_t row = blockIdx.x;
  if (row >= rows) {
    return;
  }

  __shared__ float shared[kReductionBlockSize];
  float sum = 0.0f;
  for (size_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(input[row * cols + col]);
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
    const float value = __bfloat162float(input[index]);
    const float w = __bfloat162float(weight[col]);
    output[index] = __float2bfloat16_rn(value * scale * w);
  }
}

}  // namespace

void RMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "RMSNorm expects BF16 input");
  CHECK_ERROR(weight->dtype == TensorDType::BF16, "RMSNorm expects BF16 weight");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "RMSNorm expects BF16 output");
  weight->ensure_gpu();
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const __nv_bfloat16 *weight_buf = (const __nv_bfloat16 *)weight->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t rows = flat_rows(input);
  const size_t cols = last_dim(input);

  rmsnorm_kernel_bf16xbf16_to_bf16<<<(unsigned int)rows, kReductionBlockSize>>>(
      input_buf, weight_buf, output_buf, rows, cols, eps);
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

namespace {

__global__ void linear_bias_kernel_bf16xbf16xbf16_to_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *weight,
    const __nv_bfloat16 *bias, __nv_bfloat16 *output, size_t rows,
    size_t in_dim, size_t out_dim) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = row < rows && col < out_dim;

  __shared__ __nv_bfloat16 input_tile[kLinearTileSize][kLinearTileSize];
  __shared__ __nv_bfloat16 weight_tile[kLinearTileSize][kLinearTileSize];

  float sum = valid ? __bfloat162float(bias[col]) : 0.0f;
  for (size_t k0 = 0; k0 < in_dim; k0 += kLinearTileSize) {
    const size_t input_col = k0 + threadIdx.x;
    const size_t weight_row = k0 + threadIdx.y;
    input_tile[threadIdx.y][threadIdx.x] =
        (row < rows && input_col < in_dim) ? input[row * in_dim + input_col]
                                           : __float2bfloat16_rn(0.0f);
    weight_tile[threadIdx.y][threadIdx.x] =
        (col < out_dim && weight_row < in_dim)
            ? weight[col * in_dim + weight_row]
            : __float2bfloat16_rn(0.0f);
    __syncthreads();

    for (size_t k = 0; k < kLinearTileSize && (k0 + k) < in_dim; ++k) {
      sum = fmaf(__bfloat162float(input_tile[threadIdx.y][k]),
                 __bfloat162float(weight_tile[k][threadIdx.x]), sum);
    }
    __syncthreads();
  }
  if (valid) {
    output[row * out_dim + col] = __float2bfloat16_rn(sum);
  }
}

}  // namespace

void LinearBias(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "LinearBias expects BF16 input");
  CHECK_ERROR(weight->dtype == TensorDType::BF16, "LinearBias expects BF16 weight");
  CHECK_ERROR(bias->dtype == TensorDType::BF16, "LinearBias expects BF16 bias");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "LinearBias expects BF16 output");
  weight->ensure_gpu();
  bias->ensure_gpu();
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const __nv_bfloat16 *weight_buf = (const __nv_bfloat16 *)weight->buf;
  const __nv_bfloat16 *bias_buf = (const __nv_bfloat16 *)bias->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  const size_t out_dim = weight->shape[0];

  const dim3 block(kLinearTileSize, kLinearTileSize);
  const dim3 grid((unsigned int)ceil_div(out_dim, (size_t)block.x),
                  (unsigned int)ceil_div(rows, (size_t)block.y));
  linear_bias_kernel_bf16xbf16xbf16_to_bf16<<<grid, block>>>(
      input_buf, weight_buf, bias_buf, output_buf, rows, in_dim, out_dim);
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

namespace {

__global__ void linear_kernel_bf16xbf16_to_bf16(const __nv_bfloat16 *input,
                                                const __nv_bfloat16 *weight,
                                                __nv_bfloat16 *output,
                                                size_t rows, size_t in_dim,
                                                size_t out_dim) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = row < rows && col < out_dim;

  __shared__ __nv_bfloat16 input_tile[kLinearTileSize][kLinearTileSize];
  __shared__ __nv_bfloat16 weight_tile[kLinearTileSize][kLinearTileSize];

  float sum = 0.0f;
  for (size_t k0 = 0; k0 < in_dim; k0 += kLinearTileSize) {
    const size_t input_col = k0 + threadIdx.x;
    const size_t weight_row = k0 + threadIdx.y;
    input_tile[threadIdx.y][threadIdx.x] =
        (row < rows && input_col < in_dim) ? input[row * in_dim + input_col]
                                           : __float2bfloat16_rn(0.0f);
    weight_tile[threadIdx.y][threadIdx.x] =
        (col < out_dim && weight_row < in_dim)
            ? weight[col * in_dim + weight_row]
            : __float2bfloat16_rn(0.0f);
    __syncthreads();

    for (size_t k = 0; k < kLinearTileSize && (k0 + k) < in_dim; ++k) {
      sum = fmaf(__bfloat162float(input_tile[threadIdx.y][k]),
                 __bfloat162float(weight_tile[k][threadIdx.x]), sum);
    }
    __syncthreads();
  }
  if (valid) {
    output[row * out_dim + col] = __float2bfloat16_rn(sum);
  }
}

}  // namespace

void Linear(Tensor *input, Tensor *weight, Tensor *output) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "Linear expects BF16 input");
  CHECK_ERROR(weight->dtype == TensorDType::BF16, "Linear expects BF16 weight");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "Linear expects BF16 output");
  weight->ensure_gpu();
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const __nv_bfloat16 *weight_buf = (const __nv_bfloat16 *)weight->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  const size_t out_dim = weight->shape[0];

  const dim3 block(kLinearTileSize, kLinearTileSize);
  const dim3 grid((unsigned int)ceil_div(out_dim, (size_t)block.x),
                  (unsigned int)ceil_div(rows, (size_t)block.y));
  linear_kernel_bf16xbf16_to_bf16<<<grid, block>>>(
      input_buf, weight_buf, output_buf, rows, in_dim, out_dim);
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

namespace {

__global__ void split_q_heads_grouped_kernel_bf16_to_bf16(
    const __nv_bfloat16 *input, __nv_bfloat16 *output, size_t B, size_t T,
    size_t num_kv_heads, size_t q_per_kv, size_t head_dim) {
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

}  // namespace

void SplitQHeadsGrouped(Tensor *input, Tensor *output, size_t num_kv_heads,
                        size_t q_per_kv, size_t head_dim) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "SplitQHeadsGrouped expects BF16 input");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "SplitQHeadsGrouped expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t batch_size = input->shape[0];
  const size_t seq_len = input->shape[1];
  const size_t total = input->shape[0] * num_kv_heads * q_per_kv * input->shape[1] *
                       head_dim;

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  split_q_heads_grouped_kernel_bf16_to_bf16<<<grid, block>>>(
      input_buf, output_buf, batch_size, seq_len, num_kv_heads, q_per_kv,
      head_dim);
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

namespace {

__global__ void split_kv_heads_kernel_bf16_to_bf16(
    const __nv_bfloat16 *input, __nv_bfloat16 *output, size_t B, size_t T,
    size_t num_kv_heads, size_t head_dim) {
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

}  // namespace

void SplitKVHeads(Tensor *input, Tensor *output, size_t num_kv_heads,
                  size_t head_dim) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "SplitKVHeads expects BF16 input");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "SplitKVHeads expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t batch_size = input->shape[0];
  const size_t seq_len = input->shape[1];
  const size_t total = batch_size * num_kv_heads * seq_len * head_dim;

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  split_kv_heads_kernel_bf16_to_bf16<<<grid, block>>>(
      input_buf, output_buf, batch_size, seq_len, num_kv_heads, head_dim);
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

namespace {

__global__ void apply_yarn_q_kernel_bf16(__nv_bfloat16 *q, size_t B, size_t KV,
                                         size_t QM, size_t T, size_t D,
                                         float rope_theta, float factor,
                                         float low, float high,
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
  const float x0 = __bfloat162float(q[base + d]);
  const float x1 = __bfloat162float(q[base + d + half]);
  q[base + d] = __float2bfloat16_rn(x0 * c - x1 * s);
  q[base + d + half] = __float2bfloat16_rn(x1 * c + x0 * s);
}

__global__ void apply_yarn_k_kernel_bf16(__nv_bfloat16 *k, size_t B, size_t KV,
                                         size_t T, size_t D, float rope_theta,
                                         float factor, float low, float high,
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
  const float x0 = __bfloat162float(k[base + d]);
  const float x1 = __bfloat162float(k[base + d + half]);
  k[base + d] = __float2bfloat16_rn(x0 * c - x1 * s);
  k[base + d + half] = __float2bfloat16_rn(x1 * c + x0 * s);
}

}  // namespace

void ApplyYaRNRoPE(Tensor *q, Tensor *k, const GptOssConfig &config) {
  CHECK_ERROR(q->dtype == TensorDType::BF16, "ApplyYaRNRoPE expects BF16 q");
  CHECK_ERROR(k->dtype == TensorDType::BF16, "ApplyYaRNRoPE expects BF16 k");

  __nv_bfloat16 *q_buf = (__nv_bfloat16 *)q->buf;
  __nv_bfloat16 *k_buf = (__nv_bfloat16 *)k->buf;
  const YarnParams params = build_yarn_params(config, q->shape[4]);
  const size_t q_B = q->shape[0];
  const size_t q_KV = q->shape[1];
  const size_t q_QM = q->shape[2];
  const size_t q_T = q->shape[3];
  const size_t q_D = q->shape[4];
  const size_t k_B = k->shape[0];
  const size_t k_KV = k->shape[1];
  const size_t k_T = k->shape[2];
  const size_t k_D = k->shape[3];
  const size_t q_total = q->shape[0] * q->shape[1] * q->shape[2] * q->shape[3] *
                         (q->shape[4] / 2);
  const size_t k_total = k->shape[0] * k->shape[1] * k->shape[2] *
                         (k->shape[3] / 2);

  const dim3 block(kBlockSize1D);
  const dim3 q_grid((unsigned int)ceil_div(q_total, (size_t)block.x));
  const dim3 k_grid((unsigned int)ceil_div(k_total, (size_t)block.x));
  apply_yarn_q_kernel_bf16<<<q_grid, block>>>(
      q_buf, q_B, q_KV, q_QM, q_T, q_D, config.rope_theta, params.factor,
      params.low, params.high, params.attention_factor);
  CUDA_LAUNCH_CHECK();
  apply_yarn_k_kernel_bf16<<<k_grid, block>>>(
      k_buf, k_B, k_KV, k_T, k_D, config.rope_theta, params.factor, params.low,
      params.high,
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

namespace {

__global__ void attention_scores_with_sink_kernel_bf16xbf16xbf16_to_f32(
    const __nv_bfloat16 *q, const __nv_bfloat16 *k,
    const __nv_bfloat16 *sinks, float *scores, size_t B, size_t KV, size_t QM,
    size_t T, size_t D) {
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
      sum = fmaf(__bfloat162float(q[q_base + d]),
                 __bfloat162float(k[k_base + d]), sum);
    }
    scores[score_base + tk] = sum;
  }
  scores[score_base + T] = __bfloat162float(sinks[kv * QM + qm]);
}

}  // namespace

void AttentionScoresWithSink(Tensor *q, Tensor *k, Tensor *sinks, Tensor *scores) {
  CHECK_ERROR(q->dtype == TensorDType::BF16, "AttentionScoresWithSink expects BF16 q");
  CHECK_ERROR(k->dtype == TensorDType::BF16, "AttentionScoresWithSink expects BF16 k");
  CHECK_ERROR(sinks->dtype == TensorDType::BF16,
              "AttentionScoresWithSink expects BF16 sinks");
  CHECK_ERROR(scores->dtype == TensorDType::F32, "AttentionScoresWithSink expects F32 scores");
  sinks->ensure_gpu();
  scores->ensure_gpu();

  const __nv_bfloat16 *q_buf = (const __nv_bfloat16 *)q->buf;
  const __nv_bfloat16 *k_buf = (const __nv_bfloat16 *)k->buf;
  const __nv_bfloat16 *sink_buf = (const __nv_bfloat16 *)sinks->buf;
  float *score_buf = (float *)scores->buf;
  const size_t batch_size = q->shape[0];
  const size_t num_kv_heads = q->shape[1];
  const size_t q_per_kv = q->shape[2];
  const size_t seq_len = q->shape[3];
  const size_t head_dim = q->shape[4];
  const size_t rows = q->shape[0] * q->shape[1] * q->shape[2] * q->shape[3];

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(rows, (size_t)block.x));
  attention_scores_with_sink_kernel_bf16xbf16xbf16_to_f32<<<grid, block>>>(
      q_buf, k_buf, sink_buf, score_buf, batch_size, num_kv_heads, q_per_kv,
      seq_len, head_dim);
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

namespace {

__global__ void scale_mask_softmax_kernel(const float *scores,
                                          const int32_t *lengths, float *probs,
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

}  // namespace

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const DeviceTokenBatch *tokens, size_t sliding_window) {
  probs->ensure_gpu();

  const float *score_buf = (const float *)scores->buf;
  const int32_t *length_buf = tokens != nullptr ? tokens->lengths : nullptr;
  float *prob_buf = (float *)probs->buf;
  const size_t batch_size = scores->shape[0];
  const size_t num_kv_heads = scores->shape[1];
  const size_t q_per_kv = scores->shape[2];
  const size_t seq_len = scores->shape[3];
  const size_t rows = scores->shape[0] * scores->shape[1] * scores->shape[2] *
                      scores->shape[3];

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(rows, (size_t)block.x));
  scale_mask_softmax_kernel<<<grid, block>>>(
      score_buf, length_buf, prob_buf, batch_size, num_kv_heads, q_per_kv,
      seq_len, head_dim, sliding_window);
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

namespace {

__global__ void attention_context_grouped_kernel_f32xbf16_to_bf16(
    const float *probs, const __nv_bfloat16 *v, __nv_bfloat16 *context,
    size_t B, size_t KV, size_t QM, size_t T, size_t D) {
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
    sum = fmaf(probs[prob_base + tk], __bfloat162float(v[v_base + d]), sum);
  }
  context[idx] = __float2bfloat16_rn(sum);
}

}  // namespace

void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context) {
  CHECK_ERROR(probs->dtype == TensorDType::F32, "AttentionContextGrouped expects F32 probs");
  CHECK_ERROR(v->dtype == TensorDType::BF16, "AttentionContextGrouped expects BF16 values");
  CHECK_ERROR(context->dtype == TensorDType::BF16,
              "AttentionContextGrouped expects BF16 context");
  context->ensure_gpu();

  const float *prob_buf = (const float *)probs->buf;
  const __nv_bfloat16 *value_buf = (const __nv_bfloat16 *)v->buf;
  __nv_bfloat16 *context_buf = (__nv_bfloat16 *)context->buf;
  const size_t batch_size = probs->shape[0];
  const size_t num_kv_heads = probs->shape[1];
  const size_t q_per_kv = probs->shape[2];
  const size_t seq_len = probs->shape[3];
  const size_t head_dim = v->shape[3];
  const size_t total = probs->shape[0] * probs->shape[1] * probs->shape[2] *
                       probs->shape[3] * v->shape[3];

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  attention_context_grouped_kernel_f32xbf16_to_bf16<<<grid, block>>>(
      prob_buf, value_buf, context_buf, batch_size, num_kv_heads, q_per_kv,
      seq_len, head_dim);
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

namespace {

__global__ void merge_heads_grouped_kernel_bf16_to_bf16(
    const __nv_bfloat16 *context, __nv_bfloat16 *merged, size_t B, size_t KV,
    size_t QM, size_t T, size_t D) {
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
  merged[dst] = context[idx];
}

}  // namespace

void MergeHeadsGrouped(Tensor *context, Tensor *merged) {
  CHECK_ERROR(context->dtype == TensorDType::BF16, "MergeHeadsGrouped expects BF16 context");
  CHECK_ERROR(merged->dtype == TensorDType::BF16,
              "MergeHeadsGrouped expects BF16 merged output");
  merged->ensure_gpu();

  const __nv_bfloat16 *context_buf = (const __nv_bfloat16 *)context->buf;
  __nv_bfloat16 *merged_buf = (__nv_bfloat16 *)merged->buf;
  const size_t batch_size = context->shape[0];
  const size_t num_kv_heads = context->shape[1];
  const size_t q_per_kv = context->shape[2];
  const size_t seq_len = context->shape[3];
  const size_t head_dim = context->shape[4];
  const size_t total = context->shape[0] * context->shape[1] * context->shape[2] *
                       context->shape[3] * context->shape[4];

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  merge_heads_grouped_kernel_bf16_to_bf16<<<grid, block>>>(
      context_buf, merged_buf, batch_size, num_kv_heads, q_per_kv, seq_len,
      head_dim);
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

namespace {

__global__ void residual_add_kernel_bf16xbf16_to_bf16(
    const __nv_bfloat16 *input, const __nv_bfloat16 *addend,
    __nv_bfloat16 *output, size_t n) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }

  const float lhs = __bfloat162float(input[idx]);
  const float rhs = __bfloat162float(addend[idx]);
  output[idx] = __float2bfloat16_rn(lhs + rhs);
}

}  // namespace

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "ResidualAdd expects BF16 input");
  CHECK_ERROR(addend->dtype == TensorDType::BF16, "ResidualAdd expects BF16 addend");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "ResidualAdd expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const __nv_bfloat16 *addend_buf = (const __nv_bfloat16 *)addend->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t n = input->num_elem();

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(n, (size_t)block.x));
  residual_add_kernel_bf16xbf16_to_bf16<<<grid, block>>>(
      input_buf, addend_buf, output_buf, n);
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
          value = -1.0e30f;
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
      const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
      for (size_t k = 0; k < selection->K; ++k) {
        selection->weight(b, t, k) *= inv_sum;
      }
    }
  }
}
#endif

namespace {

__global__ void topk_experts_kernel_bf16_to_i32f32(
    const __nv_bfloat16 *router_logits, int32_t *indices, float *weights,
    size_t rows, size_t experts, size_t topk) {
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
    float value = __bfloat162float(router_logits[base + e]);
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

}  // namespace

void TopKExperts(Tensor *router_logits, ExpertSelection *selection) {
  CHECK_ERROR(router_logits->dtype == TensorDType::BF16, "TopKExperts expects BF16 router logits");
  selection->ensure_gpu();

  const __nv_bfloat16 *router_buf = (const __nv_bfloat16 *)router_logits->buf;
  int32_t *index_buf = selection->indices;
  float *weight_buf = selection->weights;
  const size_t rows = router_logits->shape[0] * router_logits->shape[1];
  const size_t experts = router_logits->shape[2];
  const size_t topk = selection->K;

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(rows, (size_t)block.x));
  topk_experts_kernel_bf16_to_i32f32<<<grid, block>>>(
      router_buf, index_buf, weight_buf, rows, experts, topk);
  CUDA_LAUNCH_CHECK();
}

namespace {

__global__ void expert_gate_up_kernel_bf16xmxfp4xbf16_to_bf16(
    const __nv_bfloat16 *input, const int32_t *lengths,
    const int32_t *selection_indices, const Mxfp4PackedByte *blocks,
    const Mxfp4Scale *scales, const __nv_bfloat16 *bias, __nv_bfloat16 *output,
    size_t B, size_t T, size_t hidden, size_t K, size_t out_dim,
    size_t num_experts, size_t groups, size_t bytes_per_block,
    size_t input_elems, size_t blocks_elems, size_t scales_elems,
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
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }

  const size_t expert_idx = (size_t)selection_indices[(b * T + t) * K + k_idx];
  if (expert_idx >= num_experts) {
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }
  const size_t row_offset = expert_idx * out_dim + row;
  if (row_offset >= bias_elems) {
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }

  float sum = __bfloat162float(bias[row_offset]);
  const size_t values_per_group = bytes_per_block * 2;
  const size_t input_base = (b * T + t) * hidden;
  for (size_t g = 0; g < groups; ++g) {
    const size_t scale_index = row_offset * groups + g;
    if (scale_index >= scales_elems) {
      output[idx] = __float2bfloat16_rn(0.0f);
      return;
    }
    const size_t block_offset = scale_index * bytes_per_block;
    if (block_offset + bytes_per_block > blocks_elems) {
      output[idx] = __float2bfloat16_rn(0.0f);
      return;
    }

    const int exponent = (int)scales[scale_index].biased_exponent - 127;
    const Mxfp4PackedByte *group_blocks = blocks + block_offset;
    const size_t base = g * values_per_group;
    for (size_t i = 0; i < bytes_per_block; ++i) {
      const size_t lo_index = input_base + base + 2 * i;
      const size_t hi_index = lo_index + 1;
      if (hi_index >= input_elems) {
        output[idx] = __float2bfloat16_rn(0.0f);
        return;
      }
      const uint8_t packed = group_blocks[i].packed;
      const float lo = ldexpf(fp4_value(packed & 0x0F), exponent);
      const float hi = ldexpf(fp4_value((packed >> 4) & 0x0F), exponent);
      sum = fmaf(lo, __bfloat162float(input[lo_index]), sum);
      sum = fmaf(hi, __bfloat162float(input[hi_index]), sum);
    }
  }

  output[idx] = __float2bfloat16_rn(sum);
}

}  // namespace

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void ExpertGateUp(Tensor *input, const ExpertSelection *selection,
                  const QuantizedExpertMatrix &matrix,
                  const DeviceTokenBatch *tokens, Tensor *output) {
  CHECK_ERROR(input->ndim == 3 && output->ndim == 4, "ExpertGateUp rank mismatch");
  CHECK_ERROR(output->shape[0] == input->shape[0] && output->shape[1] == input->shape[1],
              "ExpertGateUp batch/sequence mismatch");
  CHECK_ERROR(output->shape[2] == selection->K && output->shape[3] == matrix.out_dim,
              "ExpertGateUp output shape mismatch");
  CHECK_ERROR(input->shape[2] == matrix.in_dim, "ExpertGateUp hidden size mismatch");

  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t hidden = input->shape[2];
  const size_t K = selection->K;
  const size_t out_dim = matrix.out_dim;
  const size_t values_per_group = matrix.bytes_per_block * 2;
  CHECK_ERROR(matrix.groups * values_per_group == hidden,
              "ExpertGateUp quantized group layout mismatch");

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t k_idx = 0; k_idx < K; ++k_idx) {
        float *dst = output->buf + (((b * T + t) * K + k_idx) * out_dim);
        if (tokens != nullptr && t >= (size_t)tokens->lengths[b]) {
          memset(dst, 0, out_dim * sizeof(float));
          continue;
        }

        const int32_t expert = selection->indices[(b * T + t) * K + k_idx];
        if (expert < 0 || expert >= (int32_t)matrix.num_experts) {
          memset(dst, 0, out_dim * sizeof(float));
          continue;
        }

        const float *src = input->buf + (b * T + t) * hidden;
        for (size_t row = 0; row < out_dim; ++row) {
          const size_t row_offset = (size_t)expert * out_dim + row;
          float sum = bf16_bits_to_float(matrix.bias[row_offset]);
          for (size_t g = 0; g < matrix.groups; ++g) {
            const size_t scale_index = row_offset * matrix.groups + g;
            const int exponent = (int)matrix.scales[scale_index].biased_exponent - 127;
            const Mxfp4PackedByte *group_blocks =
                matrix.blocks + scale_index * matrix.bytes_per_block;
            const size_t base = g * values_per_group;
            for (size_t i = 0; i < matrix.bytes_per_block; ++i) {
              const uint8_t packed = group_blocks[i].packed;
              sum += ldexpf(fp4_value_host(packed & 0x0F), exponent) * src[base + 2 * i];
              sum += ldexpf(fp4_value_host((packed >> 4) & 0x0F), exponent) *
                     src[base + 2 * i + 1];
            }
          }
          dst[row] = sum;
        }
      }
    }
  }
  commit_tensor_dtype(output);
}
#endif

void ExpertGateUp(Tensor *input, const ExpertSelection *selection,
                  const QuantizedExpertMatrix &matrix,
                  const DeviceTokenBatch *tokens, Tensor *output) {
  CHECK_ERROR(input->ndim == 3 && output->ndim == 4, "ExpertGateUp rank mismatch");
  CHECK_ERROR(input->dtype == TensorDType::BF16, "ExpertGateUp expects BF16 input");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "ExpertGateUp expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const int32_t *length_buf = tokens != nullptr ? tokens->lengths : nullptr;
  const int32_t *selection_index_buf = selection->indices;
  const Mxfp4PackedByte *block_buf = matrix.blocks;
  const Mxfp4Scale *scale_buf = matrix.scales;
  const __nv_bfloat16 *bias_buf = (const __nv_bfloat16 *)matrix.bias;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t batch_size = input->shape[0];
  const size_t seq_len = input->shape[1];
  const size_t hidden = input->shape[2];
  const size_t topk = selection->K;
  const size_t out_dim = matrix.out_dim;
  const size_t num_experts = matrix.num_experts;
  const size_t groups = matrix.groups;
  const size_t bytes_per_block = matrix.bytes_per_block;
  const size_t input_elems = input->num_elem();
  const size_t blocks_elems = matrix.blocks_bytes / sizeof(Mxfp4PackedByte);
  const size_t scales_elems = matrix.scales_bytes / sizeof(Mxfp4Scale);
  const size_t bias_elems = matrix.bias_bytes / sizeof(uint16_t);
  const size_t total = batch_size * seq_len * topk * out_dim;

  const dim3 block(kMoeBlockSize);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  expert_gate_up_kernel_bf16xmxfp4xbf16_to_bf16<<<grid, block>>>(
      input_buf, length_buf, selection_index_buf, block_buf, scale_buf, bias_buf,
      output_buf, batch_size, seq_len, hidden, topk, out_dim, num_experts,
      groups, bytes_per_block, input_elems, blocks_elems, scales_elems,
      bias_elems);
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

namespace {

__global__ void swiglu_clamp_kernel_bf16_to_bf16(
    const __nv_bfloat16 *input, __nv_bfloat16 *output, size_t rows,
    size_t out_dim, float limit) {
  const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t total = rows * out_dim;
  if (idx >= total) {
    return;
  }

  const size_t row = idx / out_dim;
  const size_t col = idx % out_dim;
  const size_t base = row * out_dim * 2 + col * 2;
  float gate = __bfloat162float(input[base]);
  float up = __bfloat162float(input[base + 1]);
  if (gate > limit) {
    gate = limit;
  }
  if (up > limit) {
    up = limit;
  } else if (up < -limit) {
    up = -limit;
  }
  const float value =
      (up + 1.0f) * gate * (1.0f / (1.0f + expf(-1.702f * gate)));
  output[idx] = __float2bfloat16_rn(value);
}

}  // namespace

void SwiGLUClamp(Tensor *input, Tensor *output, float limit) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "SwiGLUClamp expects BF16 input");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "SwiGLUClamp expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t rows = flat_rows(output);
  const size_t out_dim = last_dim(output);
  const size_t total = rows * out_dim;

  const dim3 block(kBlockSize1D);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  swiglu_clamp_kernel_bf16_to_bf16<<<grid, block>>>(
      input_buf, output_buf, rows, out_dim, limit);
  CUDA_LAUNCH_CHECK();
}

namespace {

__global__ void expert_down_kernel_bf16xmxfp4xbf16_to_bf16(
    const __nv_bfloat16 *input, const int32_t *lengths,
    const int32_t *selection_indices, const Mxfp4PackedByte *blocks,
    const Mxfp4Scale *scales, const __nv_bfloat16 *bias, __nv_bfloat16 *output,
    size_t B, size_t T, size_t K, size_t in_dim, size_t out_dim,
    size_t num_experts, size_t groups, size_t bytes_per_block,
    size_t input_elems, size_t blocks_elems, size_t scales_elems,
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
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }

  const size_t expert_idx = (size_t)selection_indices[(b * T + t) * K + k_idx];
  if (expert_idx >= num_experts) {
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }
  const size_t row_offset = expert_idx * out_dim + row;
  if (row_offset >= bias_elems) {
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }

  float sum = __bfloat162float(bias[row_offset]);
  const size_t values_per_group = bytes_per_block * 2;
  const size_t input_base = ((b * T + t) * K + k_idx) * in_dim;
  for (size_t g = 0; g < groups; ++g) {
    const size_t scale_index = row_offset * groups + g;
    if (scale_index >= scales_elems) {
      output[idx] = __float2bfloat16_rn(0.0f);
      return;
    }
    const size_t block_offset = scale_index * bytes_per_block;
    if (block_offset + bytes_per_block > blocks_elems) {
      output[idx] = __float2bfloat16_rn(0.0f);
      return;
    }

    const int exponent = (int)scales[scale_index].biased_exponent - 127;
    const Mxfp4PackedByte *group_blocks = blocks + block_offset;
    const size_t base = g * values_per_group;
    for (size_t i = 0; i < bytes_per_block; ++i) {
      const size_t lo_index = input_base + base + 2 * i;
      const size_t hi_index = lo_index + 1;
      if (hi_index >= input_elems) {
        output[idx] = __float2bfloat16_rn(0.0f);
        return;
      }
      const uint8_t packed = group_blocks[i].packed;
      const float lo = ldexpf(fp4_value(packed & 0x0F), exponent);
      const float hi = ldexpf(fp4_value((packed >> 4) & 0x0F), exponent);
      sum = fmaf(lo, __bfloat162float(input[lo_index]), sum);
      sum = fmaf(hi, __bfloat162float(input[hi_index]), sum);
    }
  }

  output[idx] = __float2bfloat16_rn(sum);
}

}  // namespace

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void ExpertDown(Tensor *input, const ExpertSelection *selection,
                const QuantizedExpertMatrix &matrix,
                const DeviceTokenBatch *tokens, Tensor *output) {
  CHECK_ERROR(input->ndim == 4 && output->ndim == 4, "ExpertDown rank mismatch");
  CHECK_ERROR(output->shape[0] == input->shape[0] && output->shape[1] == input->shape[1] &&
                  output->shape[2] == input->shape[2],
              "ExpertDown batch/sequence/topk mismatch");
  CHECK_ERROR(output->shape[3] == matrix.out_dim, "ExpertDown output shape mismatch");
  CHECK_ERROR(input->shape[2] == selection->K, "ExpertDown selection K mismatch");
  CHECK_ERROR(input->shape[3] == matrix.in_dim, "ExpertDown input dim mismatch");

  const size_t B = input->shape[0];
  const size_t T = input->shape[1];
  const size_t K = input->shape[2];
  const size_t in_dim = input->shape[3];
  const size_t out_dim = matrix.out_dim;
  const size_t values_per_group = matrix.bytes_per_block * 2;
  CHECK_ERROR(matrix.groups * values_per_group == in_dim,
              "ExpertDown quantized group layout mismatch");

#pragma omp parallel for collapse(3)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t k_idx = 0; k_idx < K; ++k_idx) {
        float *dst = output->buf + (((b * T + t) * K + k_idx) * out_dim);
        if (tokens != nullptr && t >= (size_t)tokens->lengths[b]) {
          memset(dst, 0, out_dim * sizeof(float));
          continue;
        }

        const int32_t expert = selection->indices[(b * T + t) * K + k_idx];
        if (expert < 0 || expert >= (int32_t)matrix.num_experts) {
          memset(dst, 0, out_dim * sizeof(float));
          continue;
        }

        const float *src = input->buf + (((b * T + t) * K + k_idx) * in_dim);
        for (size_t row = 0; row < out_dim; ++row) {
          const size_t row_offset = (size_t)expert * out_dim + row;
          float sum = bf16_bits_to_float(matrix.bias[row_offset]);
          for (size_t g = 0; g < matrix.groups; ++g) {
            const size_t scale_index = row_offset * matrix.groups + g;
            const int exponent = (int)matrix.scales[scale_index].biased_exponent - 127;
            const Mxfp4PackedByte *group_blocks =
                matrix.blocks + scale_index * matrix.bytes_per_block;
            const size_t base = g * values_per_group;
            for (size_t i = 0; i < matrix.bytes_per_block; ++i) {
              const uint8_t packed = group_blocks[i].packed;
              sum += ldexpf(fp4_value_host(packed & 0x0F), exponent) * src[base + 2 * i];
              sum += ldexpf(fp4_value_host((packed >> 4) & 0x0F), exponent) *
                     src[base + 2 * i + 1];
            }
          }
          dst[row] = sum;
        }
      }
    }
  }
  commit_tensor_dtype(output);
}
#endif

void ExpertDown(Tensor *input, const ExpertSelection *selection,
                const QuantizedExpertMatrix &matrix,
                const DeviceTokenBatch *tokens, Tensor *output) {
  CHECK_ERROR(input->ndim == 4 && output->ndim == 4, "ExpertDown rank mismatch");
  CHECK_ERROR(input->dtype == TensorDType::BF16, "ExpertDown expects BF16 input");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "ExpertDown expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const int32_t *length_buf = tokens != nullptr ? tokens->lengths : nullptr;
  const int32_t *selection_index_buf = selection->indices;
  const Mxfp4PackedByte *block_buf = matrix.blocks;
  const Mxfp4Scale *scale_buf = matrix.scales;
  const __nv_bfloat16 *bias_buf = (const __nv_bfloat16 *)matrix.bias;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t batch_size = input->shape[0];
  const size_t seq_len = input->shape[1];
  const size_t topk = input->shape[2];
  const size_t in_dim = input->shape[3];
  const size_t out_dim = matrix.out_dim;
  const size_t num_experts = matrix.num_experts;
  const size_t groups = matrix.groups;
  const size_t bytes_per_block = matrix.bytes_per_block;
  const size_t input_elems = input->num_elem();
  const size_t blocks_elems = matrix.blocks_bytes / sizeof(Mxfp4PackedByte);
  const size_t scales_elems = matrix.scales_bytes / sizeof(Mxfp4Scale);
  const size_t bias_elems = matrix.bias_bytes / sizeof(uint16_t);
  const size_t total = batch_size * seq_len * topk * out_dim;

  const dim3 block(kMoeBlockSize);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  expert_down_kernel_bf16xmxfp4xbf16_to_bf16<<<grid, block>>>(
      input_buf, length_buf, selection_index_buf, block_buf, scale_buf, bias_buf,
      output_buf, batch_size, seq_len, topk, in_dim, out_dim, num_experts,
      groups, bytes_per_block, input_elems, blocks_elems, scales_elems,
      bias_elems);
  CUDA_LAUNCH_CHECK();
}

namespace {

__global__ void weighted_expert_reduce_kernel_bf16xf32_to_bf16(
    const __nv_bfloat16 *expert_outputs, const int32_t *lengths,
    const float *selection_weights, __nv_bfloat16 *output, size_t B, size_t T,
    size_t K, size_t hidden) {
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
    output[idx] = __float2bfloat16_rn(0.0f);
    return;
  }

  float sum = 0.0f;
  for (size_t k_idx = 0; k_idx < K; ++k_idx) {
    const float weight = selection_weights[(b * T + t) * K + k_idx];
    const size_t src = (((b * T + t) * K + k_idx) * hidden) + h;
    sum = fmaf(weight, __bfloat162float(expert_outputs[src]), sum);
  }

  output[idx] = __float2bfloat16_rn(sum);
}

}  // namespace

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void WeightedExpertReduce(Tensor *expert_outputs,
                          const ExpertSelection *selection,
                          const DeviceTokenBatch *tokens, Tensor *output) {
  CHECK_ERROR(expert_outputs->ndim == 4 && output->ndim == 3,
              "WeightedExpertReduce rank mismatch");
  CHECK_ERROR(expert_outputs->shape[0] == output->shape[0] &&
                  expert_outputs->shape[1] == output->shape[1] &&
                  expert_outputs->shape[2] == selection->K &&
                  expert_outputs->shape[3] == output->shape[2],
              "WeightedExpertReduce shape mismatch");

  const size_t B = output->shape[0];
  const size_t T = output->shape[1];
  const size_t K = selection->K;
  const size_t hidden = output->shape[2];

#pragma omp parallel for collapse(2)
  for (size_t b = 0; b < B; ++b) {
    for (size_t t = 0; t < T; ++t) {
      float *dst = output->buf + (b * T + t) * hidden;
      if (tokens != nullptr && t >= (size_t)tokens->lengths[b]) {
        memset(dst, 0, hidden * sizeof(float));
        continue;
      }

      memset(dst, 0, hidden * sizeof(float));
      for (size_t k_idx = 0; k_idx < K; ++k_idx) {
        const float weight = selection->weights[(b * T + t) * K + k_idx];
        const float *src =
            expert_outputs->buf + (((b * T + t) * K + k_idx) * hidden);
        for (size_t h = 0; h < hidden; ++h) {
          dst[h] += weight * src[h];
        }
      }
    }
  }
  commit_tensor_dtype(output);
}
#endif

void WeightedExpertReduce(Tensor *expert_outputs,
                          const ExpertSelection *selection,
                          const DeviceTokenBatch *tokens, Tensor *output) {
  CHECK_ERROR(expert_outputs->ndim == 4 && output->ndim == 3,
              "WeightedExpertReduce rank mismatch");
  CHECK_ERROR(expert_outputs->dtype == TensorDType::BF16,
              "WeightedExpertReduce expects BF16 expert outputs");
  CHECK_ERROR(output->dtype == TensorDType::BF16,
              "WeightedExpertReduce expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *expert_output_buf =
      (const __nv_bfloat16 *)expert_outputs->buf;
  const int32_t *length_buf = tokens != nullptr ? tokens->lengths : nullptr;
  const float *selection_weight_buf = selection->weights;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t batch_size = output->shape[0];
  const size_t seq_len = output->shape[1];
  const size_t topk = selection->K;
  const size_t hidden = output->shape[2];
  const size_t total = batch_size * seq_len * hidden;

  const dim3 block(kMoeBlockSize);
  const dim3 grid((unsigned int)ceil_div(total, (size_t)block.x));
  weighted_expert_reduce_kernel_bf16xf32_to_bf16<<<grid, block>>>(
      expert_output_buf, length_buf, selection_weight_buf, output_buf,
      batch_size, seq_len, topk, hidden);
  CUDA_LAUNCH_CHECK();
}

#if 0  // Legacy CPU reference retained for kernel-study notes only.
void LMHead(Tensor *input, Tensor *weight, Tensor *output) { Linear(input, weight, output); }
#endif

namespace {

__global__ void linear_kernel_bf16xbf16_to_f32(const __nv_bfloat16 *input,
                                               const __nv_bfloat16 *weight,
                                               float *output, size_t rows,
                                               size_t in_dim, size_t out_dim) {
  const size_t row = blockIdx.y * blockDim.y + threadIdx.y;
  const size_t col = blockIdx.x * blockDim.x + threadIdx.x;
  const bool valid = row < rows && col < out_dim;

  __shared__ __nv_bfloat16 input_tile[kLinearTileSize][kLinearTileSize];
  __shared__ __nv_bfloat16 weight_tile[kLinearTileSize][kLinearTileSize];

  float sum = 0.0f;
  for (size_t k0 = 0; k0 < in_dim; k0 += kLinearTileSize) {
    const size_t input_col = k0 + threadIdx.x;
    const size_t weight_row = k0 + threadIdx.y;
    input_tile[threadIdx.y][threadIdx.x] =
        (row < rows && input_col < in_dim) ? input[row * in_dim + input_col]
                                           : __float2bfloat16_rn(0.0f);
    weight_tile[threadIdx.y][threadIdx.x] =
        (col < out_dim && weight_row < in_dim)
            ? weight[col * in_dim + weight_row]
            : __float2bfloat16_rn(0.0f);
    __syncthreads();

    for (size_t k = 0; k < kLinearTileSize && (k0 + k) < in_dim; ++k) {
      sum = fmaf(__bfloat162float(input_tile[threadIdx.y][k]),
                 __bfloat162float(weight_tile[k][threadIdx.x]), sum);
    }
    __syncthreads();
  }
  if (valid) {
    output[row * out_dim + col] = sum;
  }
}

}  // namespace

namespace {

void LinearToF32(Tensor *input, Tensor *weight, Tensor *output) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "LinearToF32 expects BF16 input");
  CHECK_ERROR(weight->dtype == TensorDType::BF16, "LinearToF32 expects BF16 weight");
  CHECK_ERROR(output->dtype == TensorDType::F32, "LinearToF32 expects F32 output");
  weight->ensure_gpu();
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const __nv_bfloat16 *weight_buf = (const __nv_bfloat16 *)weight->buf;
  float *output_buf = (float *)output->buf;
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  const size_t out_dim = weight->shape[0];

  const dim3 block(kLinearTileSize, kLinearTileSize);
  const dim3 grid((unsigned int)ceil_div(out_dim, (size_t)block.x),
                  (unsigned int)ceil_div(rows, (size_t)block.y));
  linear_kernel_bf16xbf16_to_f32<<<grid, block>>>(
      input_buf, weight_buf, output_buf, rows, in_dim, out_dim);
  CUDA_LAUNCH_CHECK();
}

}  // namespace

void LMHead(Tensor *input, Tensor *weight, Tensor *output) {
  LinearToF32(input, weight, output);
}
