#pragma once

#include "gpt_oss_config.h"
#include "tensor.h"

void EmbeddingLookup(TokenBatch *tokens, Tensor *embedding, Tensor *output);
void EmbeddingLookup_gpu_i32xbf16_to_bf16(TokenBatch *tokens, Tensor *embedding,
                                          Tensor *output);

void RMSNorm(Tensor *input, Tensor *weight, Tensor *output, float eps);
void RMSNorm_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *weight, Tensor *output,
                                   float eps);

void LinearBias(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output);
void LinearBias_gpu_bf16xbf16xbf16_to_bf16(Tensor *input, Tensor *weight,
                                           Tensor *bias, Tensor *output);

void Linear(Tensor *input, Tensor *weight, Tensor *output);
void Linear_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *weight, Tensor *output);
void Linear_gpu_bf16xbf16_to_f32(Tensor *input, Tensor *weight, Tensor *output);

void SplitQHeadsGrouped(Tensor *input, Tensor *output, size_t num_kv_heads,
                        size_t q_per_kv, size_t head_dim);
void SplitQHeadsGrouped_gpu_bf16_to_bf16(Tensor *input, Tensor *output,
                                         size_t num_kv_heads, size_t q_per_kv,
                                         size_t head_dim);

void SplitKVHeads(Tensor *input, Tensor *output, size_t num_kv_heads,
                  size_t head_dim);
void SplitKVHeads_gpu_bf16_to_bf16(Tensor *input, Tensor *output,
                                   size_t num_kv_heads, size_t head_dim);

void ApplyYaRNRoPE(Tensor *q, Tensor *k, const GptOssConfig &config);
void ApplyYaRNRoPE_gpu_bf16(Tensor *q, Tensor *k, const GptOssConfig &config);

void AttentionScoresWithSink(Tensor *q, Tensor *k, Tensor *sinks, Tensor *scores);
void AttentionScoresWithSink_gpu_bf16xbf16xbf16_to_f32(Tensor *q, Tensor *k,
                                                       Tensor *sinks,
                                                       Tensor *scores);

void ScaleMaskSoftmax(Tensor *scores, Tensor *probs, size_t head_dim,
                      const TokenBatch *tokens, size_t sliding_window);
void ScaleMaskSoftmax_gpu_f32_to_f32(Tensor *scores, Tensor *probs,
                                     size_t head_dim, const TokenBatch *tokens,
                                     size_t sliding_window);

void AttentionContextGrouped(Tensor *probs, Tensor *v, Tensor *context);
void AttentionContextGrouped_gpu_f32xbf16_to_bf16(Tensor *probs, Tensor *v,
                                                  Tensor *context);

void MergeHeadsGrouped(Tensor *context, Tensor *merged);
void MergeHeadsGrouped_gpu_bf16_to_bf16(Tensor *context, Tensor *merged);

void ResidualAdd(Tensor *input, Tensor *addend, Tensor *output);
void ResidualAdd_gpu_bf16xbf16_to_bf16(Tensor *input, Tensor *addend,
                                       Tensor *output);

void TopKExperts(Tensor *router_logits, ExpertSelection *selection);
void TopKExperts_gpu_bf16_to_i32f32(Tensor *router_logits,
                                    ExpertSelection *selection);

void SwiGLUClamp(Tensor *input, Tensor *output, float limit);
void SwiGLUClamp_gpu_bf16_to_bf16(Tensor *input, Tensor *output, float limit);

void LMHead(Tensor *input, Tensor *weight, Tensor *output);
void LMHead_gpu_bf16xbf16_to_f32(Tensor *input, Tensor *weight, Tensor *output);
