#pragma once

#include "gpt_oss_config.h"
#include "tensor.h"

TokenBatch load_tokens(const char *path);
void initialize_model(const char *model_dir);
void alloc_activations(size_t batch_size, size_t seq_len);
void gpt_oss_forward(DeviceTokenBatch *tokens, Tensor *logits);
void select_next_tokens(DeviceTokenBatch *tokens, Tensor *logits, int32_t *next_tokens);
void append_next_tokens(DeviceTokenBatch *tokens, const int32_t *next_tokens,
                        int32_t *generated_tokens, int32_t *generated_lengths,
                        uint8_t *finished, size_t max_new_tokens);
void finalize_model();
void free_activations();
const GptOssConfig &model_config();
