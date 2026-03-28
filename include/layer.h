#pragma once

#include "gpt_oss_config.h"
#include "tensor.h"

struct QuantizedExpertMatrix;

void EmbeddingLookup(DeviceTokenBatch *tokens, Tensor *embedding, Tensor *output);

void RMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps);

void LinearBias(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output);

void Linear(Tensor *input, Tensor *weight, Tensor *output);

void SplitQHeadsGrouped(Tensor *input, Tensor *output, size_t num_kv_heads,
                        size_t q_per_kv, size_t head_dim);

void SplitKVHeads(Tensor *input, Tensor *output, size_t num_kv_heads,
                  size_t head_dim);

void ApplyYaRNRoPE(Tensor *q, Tensor *k, const GptOssConfig &config);

void AttentionScoresWithSink(Tensor *q, Tensor *k, Tensor *sinks, Tensor *scores);

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const DeviceTokenBatch *tokens, size_t sliding_window);

void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context);

void MergeHeadsGrouped(Tensor *context, Tensor *merged);

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output);

void TopKExperts(Tensor *router_logits, ExpertSelection *selection);

void ExpertGateUp(Tensor *input, const ExpertSelection *selection,
                  const QuantizedExpertMatrix &matrix,
                  const DeviceTokenBatch *tokens, Tensor *output);

void SwiGLUClamp(Tensor *input, Tensor *output, float limit);

void ExpertDown(Tensor *input, const ExpertSelection *selection,
                const QuantizedExpertMatrix &matrix,
                const DeviceTokenBatch *tokens, Tensor *output);

void WeightedExpertReduce(Tensor *expert_outputs,
                          const ExpertSelection *selection,
                          const DeviceTokenBatch *tokens, Tensor *output);

void LMHead(Tensor *input, Tensor *weight, Tensor *output);
