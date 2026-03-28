#pragma once

#include <cstddef>
#include <cstdlib>

#include "tensor.h"

#define EXIT(status)                                                         \
  do {                                                                       \
    exit(status);                                                            \
  } while (0)

#define CHECK_ERROR(cond, fmt, ...)                                          \
  do {                                                                       \
    if (!(cond)) {                                                           \
      fprintf(stderr, "[%s:%d] " fmt "\n", __FILE__, __LINE__,               \
              ##__VA_ARGS__);                                                \
      EXIT(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

double get_time();
void *read_binary(const char *filename, size_t *size);
void write_binary(const char *filename, const void *buf, size_t size);
int validate_buffer(const float *output, const float *answer, size_t n,
                    float atol, float rtol);
void print_last_token_topk(const Tensor *logits, size_t batch_size,
                           size_t seq_len, int k);
