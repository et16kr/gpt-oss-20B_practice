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

#include "util.h"

namespace {

inline size_t flat_rows(Tensor *tensor) {
  CHECK_ERROR(tensor->ndim >= 2, "Tensor must have at least rank 2");
  return tensor->num_elem() / tensor->shape[tensor->ndim - 1];
}

inline size_t last_dim(Tensor *tensor) { return tensor->shape[tensor->ndim - 1]; }

std::pair<std::vector<float>, std::vector<float>> build_yarn_cos_sin(
    const GptOssConfig &config, size_t seq_len, size_t dim) {
  constexpr float kPi = 3.14159265358979323846f;
  CHECK_ERROR((dim % 2) == 0, "RoPE head_dim must be even");

  const size_t half_dim = dim / 2;
  std::vector<float> inv_freq(half_dim, 0.0f);
  const float scaling = config.rope_scaling_factor;

  for (size_t idx = 0; idx < half_dim; ++idx) {
    const float exponent = (2.0f * (float)idx) / (float)dim;
    const float freq = powf(config.rope_theta, exponent);
    if (scaling > 1.0f) {
      const float d_half = (float)half_dim;
      const float low =
          d_half *
          logf((float)config.initial_context_length /
               (config.rope_ntk_beta * 2.0f * kPi)) /
          logf(config.rope_theta);
      const float high =
          d_half *
          logf((float)config.initial_context_length /
               (config.rope_ntk_alpha * 2.0f * kPi)) /
          logf(config.rope_theta);
      const float interpolation = 1.0f / (scaling * freq);
      const float extrapolation = 1.0f / freq;
      float ramp = ((float)idx - low) / (high - low);
      if (ramp < 0.0f) {
        ramp = 0.0f;
      }
      if (ramp > 1.0f) {
        ramp = 1.0f;
      }
      const float mask = 1.0f - ramp;
      inv_freq[idx] = interpolation * (1.0f - mask) + extrapolation * mask;
    } else {
      inv_freq[idx] = 1.0f / freq;
    }
  }

  float concentration = 1.0f;
  if (scaling > 1.0f) {
    concentration = 0.1f * logf(scaling) + 1.0f;
  }

  std::vector<float> cos(seq_len * half_dim, 0.0f);
  std::vector<float> sin(seq_len * half_dim, 0.0f);
  for (size_t t = 0; t < seq_len; ++t) {
    for (size_t i = 0; i < half_dim; ++i) {
      const float angle = (float)t * inv_freq[i];
      cos[t * half_dim + i] = cosf(angle) * concentration;
      sin[t * half_dim + i] = sinf(angle) * concentration;
    }
  }
  return {std::move(cos), std::move(sin)};
}

inline float sigmoidf(float x) { return 1.0f / (1.0f + expf(-x)); }

}  // namespace

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
}

void EmbeddingLookup_gpu_i32xbf16_to_bf16(TokenBatch *tokens, Tensor *embedding,
                                          Tensor *output) {
  // GPU dtype note:
  // tokens: int32 ids
  // embedding/output storage: BF16 in current gpt-oss graph
  EmbeddingLookup(tokens, embedding, output);
}

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
}

void RMSNorm_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *weight, Tensor *output,
                                   float eps) {
  // GPU dtype note:
  // input/weight/output storage: BF16 in current gpt-oss graph
  // reduction and rsqrt are expected to accumulate in FP32
  RMSNorm(input, weight, output, eps);
}

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
}

void LinearBias_gpu_bf16xbf16xbf16_to_bf16(Tensor *input, Tensor *weight,
                                           Tensor *bias, Tensor *output) {
  // GPU dtype note:
  // input/output storage: BF16 in current gpt-oss graph
  // weight/bias storage: checkpoint dtype, typically BF16
  // matmul accumulation should be FP32
  LinearBias(input, weight, bias, output);
}

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
}

void Linear_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *weight, Tensor *output) {
  // GPU dtype note:
  // generic linear path usually uses BF16 input/weight/output
  // accumulation should be FP32
  Linear(input, weight, output);
}

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
}

void SplitQHeadsGrouped_gpu_bf16_to_bf16(Tensor *input, Tensor *output,
                                         size_t num_kv_heads, size_t q_per_kv,
                                         size_t head_dim) {
  // GPU dtype note:
  // input/output storage: BF16 in current gpt-oss graph
  // this is a layout transform only
  SplitQHeadsGrouped(input, output, num_kv_heads, q_per_kv, head_dim);
}

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
}

void SplitKVHeads_gpu_bf16_to_bf16(Tensor *input, Tensor *output,
                                   size_t num_kv_heads, size_t head_dim) {
  // GPU dtype note:
  // input/output storage: BF16 in current gpt-oss graph
  // this is a layout transform only
  SplitKVHeads(input, output, num_kv_heads, head_dim);
}

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
}

void ApplyYaRNRoPE_gpu_bf16(Tensor *q, Tensor *k, const GptOssConfig &config) {
  // GPU dtype note:
  // q/k storage: BF16 in current gpt-oss graph
  // trig tables and rotation math are best computed in FP32, then stored back as BF16
  ApplyYaRNRoPE(q, k, config);
}

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

void AttentionScoresWithSink_gpu_bf16xbf16xbf16_to_f32(Tensor *q, Tensor *k,
                                                       Tensor *sinks,
                                                       Tensor *scores) {
  // GPU dtype note:
  // q/k storage: BF16
  // sinks storage: checkpoint dtype, typically BF16
  // scores storage: F32 in current gpt-oss graph
  // dot-product accumulation should be FP32
  AttentionScoresWithSink(q, k, sinks, scores);
}

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

void ScaleMaskSoftmax_gpu_f32_to_f32(Tensor *scores, Tensor *probs,
                                     size_t head_dim, const TokenBatch *tokens,
                                     size_t sliding_window) {
  // GPU dtype note:
  // scores/probs storage: F32 in current gpt-oss graph
  // masking, max-reduction, exp, and normalization stay in FP32
  ScaleMaskSoftmax(scores, probs, head_dim, tokens, sliding_window);
}

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
}

void AttentionContextGrouped_gpu_f32xbf16_to_bf16(Tensor *probs, Tensor *v,
                                                  Tensor *context) {
  // GPU dtype note:
  // probs storage: F32
  // v/context storage: BF16 in current gpt-oss graph
  // weighted sum should accumulate in FP32 before storing to BF16 context
  AttentionContextGrouped(probs, v, context);
}

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
}

void MergeHeadsGrouped_gpu_bf16_to_bf16(Tensor *context, Tensor *merged) {
  // GPU dtype note:
  // context/merged storage: BF16 in current gpt-oss graph
  // this is a layout transform only
  MergeHeadsGrouped(context, merged);
}

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output) {
  CHECK_ERROR(input->num_elem() == addend->num_elem() &&
                  input->num_elem() == output->num_elem(),
              "ResidualAdd shape mismatch");

#pragma omp parallel for
  for (size_t i = 0; i < input->num_elem(); ++i) {
    output->buf[i] = input->buf[i] + addend->buf[i];
  }
}

void ResidualAdd_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *addend,
                                       Tensor *output) {
  // GPU dtype note:
  // input/addend/output storage: BF16 in current gpt-oss graph
  // elementwise add may accumulate in FP32, then cast/store to BF16
  ResidualAdd(input, addend, output);
}

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
        values.push_back({row[e], (int32_t)e});
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

void TopKExperts_gpu_bf16_to_i32f32(Tensor *router_logits,
                                    ExpertSelection *selection) {
  // GPU dtype note:
  // router_logits storage: BF16 in current gpt-oss graph
  // top-k compare and softmax weights should use FP32
  // selection indices: int32, selection weights: FP32
  TopKExperts(router_logits, selection);
}

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
}

void SwiGLUClamp_gpu_bf16_to_bf16(Tensor *input, Tensor *output, float limit) {
  // GPU dtype note:
  // input/output storage: BF16 in current gpt-oss graph
  // clamp, sigmoid, and multiply are best evaluated in FP32, then stored as BF16
  SwiGLUClamp(input, output, limit);
}

void LMHead(Tensor *input, Tensor *weight, Tensor *output) { Linear(input, weight, output); }

void Linear_gpu_bf16xbf16_to_f32(Tensor *input, Tensor *weight, Tensor *output) {
  // GPU dtype note:
  // input storage: BF16 in current gpt-oss graph
  // weight storage: checkpoint dtype, typically BF16
  // output storage: F32
  // matmul accumulation should be FP32
  Linear(input, weight, output);
}

void LMHead_gpu_bf16xbf16_to_f32(Tensor *input, Tensor *weight, Tensor *output) {
  // GPU dtype note:
  // input storage: BF16 in current gpt-oss graph
  // weight storage: checkpoint dtype, typically BF16
  // output logits storage: F32
  // matmul accumulation should be FP32
  Linear_gpu_bf16xbf16_to_f32(input, weight, output);
}
