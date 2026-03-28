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

#define CUDA_LAUNCH_CHECK()            \
  do {                                \
    CHECK_CUDA(cudaGetLastError());   \
    CHECK_CUDA(cudaDeviceSynchronize()); \
  } while (0)

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t token_slot = 0; token_slot < num_tokens; ++token_slot) {
  //   const int32_t token_id = token_buf[token_slot];
  //   for (size_t h = 0; h < hidden; ++h) {
  //     const size_t idx = token_slot * hidden + h;
  //     output_buf[idx] =
  //         (token_id < 0 || token_id >= (int32_t)vocab_size)
  //             ? __float2bfloat16_rn(0.0f)
  //             : embedding_buf[(size_t)token_id * hidden + h];
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t row = 0; row < rows; ++row) {
  //   float sum = 0.0f;
  //   for (size_t col = 0; col < cols; ++col) {
  //     const float value = __bfloat162float(input_buf[row * cols + col]);
  //     sum = fmaf(value, value, sum);
  //   }
  //   const float scale = rsqrtf(sum / (float)cols + eps);
  //   for (size_t col = 0; col < cols; ++col) {
  //     const size_t index = row * cols + col;
  //     output_buf[index] = __float2bfloat16_rn(
  //         __bfloat162float(input_buf[index]) * scale *
  //         __bfloat162float(weight_buf[col]));
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t row = 0; row < rows; ++row) {
  //   for (size_t col = 0; col < out_dim; ++col) {
  //     float sum = __bfloat162float(bias_buf[col]);
  //     for (size_t k = 0; k < in_dim; ++k) {
  //       sum = fmaf(__bfloat162float(input_buf[row * in_dim + k]),
  //                  __bfloat162float(weight_buf[col * in_dim + k]), sum);
  //     }
  //     output_buf[row * out_dim + col] = __float2bfloat16_rn(sum);
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t row = 0; row < rows; ++row) {
  //   for (size_t col = 0; col < out_dim; ++col) {
  //     float sum = 0.0f;
  //     for (size_t k = 0; k < in_dim; ++k) {
  //       sum = fmaf(__bfloat162float(input_buf[row * in_dim + k]),
  //                  __bfloat162float(weight_buf[col * in_dim + k]), sum);
  //     }
  //     output_buf[row * out_dim + col] = __float2bfloat16_rn(sum);
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t kv = 0; kv < num_kv_heads; ++kv) {
  //     for (size_t qm = 0; qm < q_per_kv; ++qm) {
  //       for (size_t t = 0; t < seq_len; ++t) {
  //         const size_t src_head = kv * q_per_kv + qm;
  //         const size_t src_base =
  //             (b * seq_len + t) * (num_kv_heads * q_per_kv * head_dim) +
  //             src_head * head_dim;
  //         const size_t dst_base =
  //             ((((b * num_kv_heads + kv) * q_per_kv + qm) * seq_len) + t) *
  //             head_dim;
  //         for (size_t d = 0; d < head_dim; ++d) {
  //           output_buf[dst_base + d] = input_buf[src_base + d];
  //         }
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t kv = 0; kv < num_kv_heads; ++kv) {
  //     for (size_t t = 0; t < seq_len; ++t) {
  //       const size_t src_base =
  //           (b * seq_len + t) * (num_kv_heads * head_dim) + kv * head_dim;
  //       const size_t dst_base =
  //           ((b * num_kv_heads + kv) * seq_len + t) * head_dim;
  //       for (size_t d = 0; d < head_dim; ++d) {
  //         output_buf[dst_base + d] = input_buf[src_base + d];
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace the first launch with a host-side reference loop using q_buf.
  // for (size_t b = 0; b < q_B; ++b) {
  //   for (size_t kv = 0; kv < q_KV; ++kv) {
  //     for (size_t qm = 0; qm < q_QM; ++qm) {
  //       for (size_t t = 0; t < q_T; ++t) {
  //         const size_t half = q_D / 2;
  //         const size_t base = ((((b * q_KV + kv) * q_QM + qm) * q_T + t) * q_D);
  //         for (size_t d = 0; d < half; ++d) {
  //           const float angle = (float)t * rope_inv_freq_device(
  //               config.rope_theta, params.factor, params.low, params.high, q_D, d);
  //           const float c = cosf(angle) * params.attention_factor;
  //           const float s = sinf(angle) * params.attention_factor;
  //           const float x0 = __bfloat162float(q_buf[base + d]);
  //           const float x1 = __bfloat162float(q_buf[base + d + half]);
  //           q_buf[base + d] = __float2bfloat16_rn(x0 * c - x1 * s);
  //           q_buf[base + d + half] = __float2bfloat16_rn(x1 * c + x0 * s);
  //         }
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
  // TODO: replace the second launch with a host-side reference loop using k_buf.
  // for (size_t b = 0; b < k_B; ++b) {
  //   for (size_t kv = 0; kv < k_KV; ++kv) {
  //     for (size_t t = 0; t < k_T; ++t) {
  //       const size_t half = k_D / 2;
  //       const size_t base = (((b * k_KV + kv) * k_T + t) * k_D);
  //       for (size_t d = 0; d < half; ++d) {
  //         const float angle = (float)t * rope_inv_freq_device(
  //             config.rope_theta, params.factor, params.low, params.high, k_D, d);
  //         const float c = cosf(angle) * params.attention_factor;
  //         const float s = sinf(angle) * params.attention_factor;
  //         const float x0 = __bfloat162float(k_buf[base + d]);
  //         const float x1 = __bfloat162float(k_buf[base + d + half]);
  //         k_buf[base + d] = __float2bfloat16_rn(x0 * c - x1 * s);
  //         k_buf[base + d + half] = __float2bfloat16_rn(x1 * c + x0 * s);
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t kv = 0; kv < num_kv_heads; ++kv) {
  //     for (size_t qm = 0; qm < q_per_kv; ++qm) {
  //       for (size_t tq = 0; tq < seq_len; ++tq) {
  //         const size_t q_base =
  //             ((((b * num_kv_heads + kv) * q_per_kv + qm) * seq_len) + tq) *
  //             head_dim;
  //         const size_t score_base =
  //             ((((b * num_kv_heads + kv) * q_per_kv + qm) * seq_len) + tq) *
  //             (seq_len + 1);
  //         for (size_t tk = 0; tk < seq_len; ++tk) {
  //           const size_t k_base = (((b * num_kv_heads + kv) * seq_len + tk) * head_dim);
  //           float sum = 0.0f;
  //           for (size_t d = 0; d < head_dim; ++d) {
  //             sum = fmaf(__bfloat162float(q_buf[q_base + d]),
  //                        __bfloat162float(k_buf[k_base + d]), sum);
  //           }
  //           score_buf[score_base + tk] = sum;
  //         }
  //         score_buf[score_base + seq_len] =
  //             __bfloat162float(sink_buf[kv * q_per_kv + qm]);
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   const size_t valid_t = length_buf != nullptr ? (size_t)length_buf[b] : seq_len;
  //   for (size_t kv = 0; kv < num_kv_heads; ++kv) {
  //     for (size_t qm = 0; qm < q_per_kv; ++qm) {
  //       for (size_t tq = 0; tq < seq_len; ++tq) {
  //         const size_t row_base =
  //             ((((b * num_kv_heads + kv) * q_per_kv + qm) * seq_len) + tq) *
  //             (seq_len + 1);
  //         for (size_t i = 0; i < seq_len + 1; ++i) {
  //           prob_buf[row_base + i] = 0.0f;
  //         }
  //         if (tq >= valid_t) {
  //           continue;
  //         }
  //         size_t tk_begin = 0;
  //         if (sliding_window > 0 && tq + 1 > sliding_window) {
  //           tk_begin = tq + 1 - sliding_window;
  //         }
  //         const size_t tk_end = std::min(tq, valid_t - 1);
  //         const float scale = rsqrtf((float)head_dim);
  //         float row_max = -1.0e30f;
  //         for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
  //           row_max = std::max(row_max, score_buf[row_base + tk] * scale);
  //         }
  //         row_max = std::max(row_max, score_buf[row_base + seq_len]);
  //         float sum = 0.0f;
  //         for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
  //           const float value = expf(score_buf[row_base + tk] * scale - row_max);
  //           prob_buf[row_base + tk] = value;
  //           sum += value;
  //         }
  //         prob_buf[row_base + seq_len] = expf(score_buf[row_base + seq_len] - row_max);
  //         sum += prob_buf[row_base + seq_len];
  //         if (sum > 0.0f) {
  //           const float inv_sum = 1.0f / sum;
  //           for (size_t tk = tk_begin; tk <= tk_end; ++tk) {
  //             prob_buf[row_base + tk] *= inv_sum;
  //           }
  //           prob_buf[row_base + seq_len] *= inv_sum;
  //         }
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t kv = 0; kv < num_kv_heads; ++kv) {
  //     for (size_t qm = 0; qm < q_per_kv; ++qm) {
  //       for (size_t tq = 0; tq < seq_len; ++tq) {
  //         const size_t prob_base =
  //             ((((b * num_kv_heads + kv) * q_per_kv + qm) * seq_len) + tq) *
  //             (seq_len + 1);
  //         const size_t out_base =
  //             ((((b * num_kv_heads + kv) * q_per_kv + qm) * seq_len) + tq) *
  //             head_dim;
  //         for (size_t d = 0; d < head_dim; ++d) {
  //           float sum = 0.0f;
  //           for (size_t tk = 0; tk < seq_len; ++tk) {
  //             const size_t value_base =
  //                 (((b * num_kv_heads + kv) * seq_len + tk) * head_dim);
  //             sum = fmaf(prob_buf[prob_base + tk],
  //                        __bfloat162float(value_buf[value_base + d]), sum);
  //           }
  //           context_buf[out_base + d] = __float2bfloat16_rn(sum);
  //         }
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t kv = 0; kv < num_kv_heads; ++kv) {
  //     for (size_t qm = 0; qm < q_per_kv; ++qm) {
  //       for (size_t t = 0; t < seq_len; ++t) {
  //         const size_t head = kv * q_per_kv + qm;
  //         const size_t src_base =
  //             ((((b * num_kv_heads + kv) * q_per_kv + qm) * seq_len) + t) *
  //             head_dim;
  //         const size_t dst_base =
  //             (b * seq_len + t) * (num_kv_heads * q_per_kv * head_dim) +
  //             head * head_dim;
  //         for (size_t d = 0; d < head_dim; ++d) {
  //           merged_buf[dst_base + d] = context_buf[src_base + d];
  //         }
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "ResidualAdd expects BF16 input");
  CHECK_ERROR(addend->dtype == TensorDType::BF16, "ResidualAdd expects BF16 addend");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "ResidualAdd expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const __nv_bfloat16 *addend_buf = (const __nv_bfloat16 *)addend->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t n = input->num_elem();

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t i = 0; i < n; ++i) {
  //   output_buf[i] = __float2bfloat16_rn(
  //       __bfloat162float(input_buf[i]) + __bfloat162float(addend_buf[i]));
  // }
  CUDA_LAUNCH_CHECK();
}

void TopKExperts(Tensor *router_logits, ExpertSelection *selection) {
  CHECK_ERROR(router_logits->dtype == TensorDType::BF16, "TopKExperts expects BF16 router logits");
  selection->ensure_gpu();

  const __nv_bfloat16 *router_buf = (const __nv_bfloat16 *)router_logits->buf;
  int32_t *index_buf = selection->indices;
  float *weight_buf = selection->weights;
  const size_t rows = router_logits->shape[0] * router_logits->shape[1];
  const size_t experts = router_logits->shape[2];
  const size_t topk = selection->K;

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t row = 0; row < rows; ++row) {
  //   std::vector<std::pair<float, int32_t>> values;
  //   values.reserve(experts);
  //   for (size_t e = 0; e < experts; ++e) {
  //     float value = __bfloat162float(router_buf[row * experts + e]);
  //     if (!std::isfinite(value)) {
  //       value = -1.0e30f;
  //     }
  //     values.push_back({value, (int32_t)e});
  //   }
  //   std::partial_sort(
  //       values.begin(), values.begin() + topk, values.end(),
  //       [](const std::pair<float, int32_t> &lhs, const std::pair<float, int32_t> &rhs) {
  //         if (lhs.first != rhs.first) {
  //           return lhs.first > rhs.first;
  //         }
  //         return lhs.second < rhs.second;
  //       });
  //   float row_max = values[0].first;
  //   float sum = 0.0f;
  //   for (size_t k = 0; k < topk; ++k) {
  //     index_buf[row * topk + k] = values[k].second;
  //     weight_buf[row * topk + k] = expf(values[k].first - row_max);
  //     sum += weight_buf[row * topk + k];
  //   }
  //   const float inv_sum = sum > 0.0f ? 1.0f / sum : 0.0f;
  //   for (size_t k = 0; k < topk; ++k) {
  //     weight_buf[row * topk + k] *= inv_sum;
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t t = 0; t < seq_len; ++t) {
  //     for (size_t k_idx = 0; k_idx < topk; ++k_idx) {
  //       const size_t dst_base = (((b * seq_len + t) * topk + k_idx) * out_dim);
  //       if (length_buf != nullptr && t >= (size_t)length_buf[b]) {
  //         for (size_t row = 0; row < out_dim; ++row) {
  //           output_buf[dst_base + row] = __float2bfloat16_rn(0.0f);
  //         }
  //         continue;
  //       }
  //       const size_t expert_idx = (size_t)selection_index_buf[(b * seq_len + t) * topk + k_idx];
  //       if (expert_idx >= num_experts) {
  //         for (size_t row = 0; row < out_dim; ++row) {
  //           output_buf[dst_base + row] = __float2bfloat16_rn(0.0f);
  //         }
  //         continue;
  //       }
  //       for (size_t row = 0; row < out_dim; ++row) {
  //         const size_t row_offset = expert_idx * out_dim + row;
  //         float sum = __bfloat162float(bias_buf[row_offset]);
  //         for (size_t g = 0; g < groups; ++g) {
  //           const size_t scale_index = row_offset * groups + g;
  //           const int exponent = (int)scale_buf[scale_index].biased_exponent - 127;
  //           const size_t block_offset = scale_index * bytes_per_block;
  //           const size_t input_base = (b * seq_len + t) * hidden + g * bytes_per_block * 2;
  //           for (size_t i = 0; i < bytes_per_block; ++i) {
  //             const uint8_t packed = block_buf[block_offset + i].packed;
  //             sum = fmaf(ldexpf(decode_fp4_nibble(packed & 0x0F), exponent),
  //                        __bfloat162float(input_buf[input_base + 2 * i]), sum);
  //             sum = fmaf(ldexpf(decode_fp4_nibble((packed >> 4) & 0x0F), exponent),
  //                        __bfloat162float(input_buf[input_base + 2 * i + 1]), sum);
  //           }
  //         }
  //         output_buf[dst_base + row] = __float2bfloat16_rn(sum);
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

void SwiGLUClamp(Tensor *input, Tensor *output, float limit) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "SwiGLUClamp expects BF16 input");
  CHECK_ERROR(output->dtype == TensorDType::BF16, "SwiGLUClamp expects BF16 output");
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  __nv_bfloat16 *output_buf = (__nv_bfloat16 *)output->buf;
  const size_t rows = flat_rows(output);
  const size_t out_dim = last_dim(output);
  const size_t total = rows * out_dim;

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t row = 0; row < rows; ++row) {
  //   for (size_t col = 0; col < out_dim; ++col) {
  //     const size_t base = row * out_dim * 2 + col * 2;
  //     float gate = __bfloat162float(input_buf[base]);
  //     float up = __bfloat162float(input_buf[base + 1]);
  //     if (gate > limit) {
  //       gate = limit;
  //     }
  //     if (up > limit) {
  //       up = limit;
  //     } else if (up < -limit) {
  //       up = -limit;
  //     }
  //     output_buf[row * out_dim + col] = __float2bfloat16_rn(
  //         (up + 1.0f) * gate * (1.0f / (1.0f + expf(-1.702f * gate))));
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t t = 0; t < seq_len; ++t) {
  //     for (size_t k_idx = 0; k_idx < topk; ++k_idx) {
  //       const size_t dst_base = (((b * seq_len + t) * topk + k_idx) * out_dim);
  //       if (length_buf != nullptr && t >= (size_t)length_buf[b]) {
  //         for (size_t row = 0; row < out_dim; ++row) {
  //           output_buf[dst_base + row] = __float2bfloat16_rn(0.0f);
  //         }
  //         continue;
  //       }
  //       const size_t expert_idx = (size_t)selection_index_buf[(b * seq_len + t) * topk + k_idx];
  //       if (expert_idx >= num_experts) {
  //         for (size_t row = 0; row < out_dim; ++row) {
  //           output_buf[dst_base + row] = __float2bfloat16_rn(0.0f);
  //         }
  //         continue;
  //       }
  //       for (size_t row = 0; row < out_dim; ++row) {
  //         const size_t row_offset = expert_idx * out_dim + row;
  //         float sum = __bfloat162float(bias_buf[row_offset]);
  //         for (size_t g = 0; g < groups; ++g) {
  //           const size_t scale_index = row_offset * groups + g;
  //           const int exponent = (int)scale_buf[scale_index].biased_exponent - 127;
  //           const size_t block_offset = scale_index * bytes_per_block;
  //           const size_t input_base =
  //               ((b * seq_len + t) * topk + k_idx) * in_dim + g * bytes_per_block * 2;
  //           for (size_t i = 0; i < bytes_per_block; ++i) {
  //             const uint8_t packed = block_buf[block_offset + i].packed;
  //             sum = fmaf(ldexpf(decode_fp4_nibble(packed & 0x0F), exponent),
  //                        __bfloat162float(input_buf[input_base + 2 * i]), sum);
  //             sum = fmaf(ldexpf(decode_fp4_nibble((packed >> 4) & 0x0F), exponent),
  //                        __bfloat162float(input_buf[input_base + 2 * i + 1]), sum);
  //           }
  //         }
  //         output_buf[dst_base + row] = __float2bfloat16_rn(sum);
  //       }
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

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

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t b = 0; b < batch_size; ++b) {
  //   for (size_t t = 0; t < seq_len; ++t) {
  //     const size_t dst_base = (b * seq_len + t) * hidden;
  //     if (length_buf != nullptr && t >= (size_t)length_buf[b]) {
  //       for (size_t h = 0; h < hidden; ++h) {
  //         output_buf[dst_base + h] = __float2bfloat16_rn(0.0f);
  //       }
  //       continue;
  //     }
  //     for (size_t h = 0; h < hidden; ++h) {
  //       float sum = 0.0f;
  //       for (size_t k_idx = 0; k_idx < topk; ++k_idx) {
  //         const size_t src =
  //             (((b * seq_len + t) * topk + k_idx) * hidden) + h;
  //         sum = fmaf(selection_weight_buf[(b * seq_len + t) * topk + k_idx],
  //                    __bfloat162float(expert_output_buf[src]), sum);
  //       }
  //       output_buf[dst_base + h] = __float2bfloat16_rn(sum);
  //     }
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}

void LMHead(Tensor *input, Tensor *weight, Tensor *output) {
  CHECK_ERROR(input->dtype == TensorDType::BF16, "LMHead expects BF16 input");
  CHECK_ERROR(weight->dtype == TensorDType::BF16, "LMHead expects BF16 weight");
  CHECK_ERROR(output->dtype == TensorDType::F32, "LMHead expects F32 output");
  weight->ensure_gpu();
  output->ensure_gpu();

  const __nv_bfloat16 *input_buf = (const __nv_bfloat16 *)input->buf;
  const __nv_bfloat16 *weight_buf = (const __nv_bfloat16 *)weight->buf;
  float *output_buf = (float *)output->buf;
  const size_t rows = flat_rows(input);
  const size_t in_dim = last_dim(input);
  const size_t out_dim = weight->shape[0];

  // TODO: replace with a host-side reference loop using the buffers above.
  // for (size_t row = 0; row < rows; ++row) {
  //   for (size_t col = 0; col < out_dim; ++col) {
  //     float sum = 0.0f;
  //     for (size_t k = 0; k < in_dim; ++k) {
  //       sum = fmaf(__bfloat162float(input_buf[row * in_dim + k]),
  //                  __bfloat162float(weight_buf[col * in_dim + k]), sum);
  //     }
  //     output_buf[row * out_dim + col] = sum;
  //   }
  // }
  CUDA_LAUNCH_CHECK();
}
