#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t status_ = call;                                              \
    if (status_ != cudaSuccess) {                                            \
      fprintf(stderr, "CUDA error (%s:%d): %s:%s\n", __FILE__, __LINE__,     \
              cudaGetErrorName(status_), cudaGetErrorString(status_));       \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

enum class TensorDType : uint8_t {
  F32 = 0,
  F16 = 1,
  BF16 = 2,
  U8 = 3,
};

const char *tensor_dtype_name(TensorDType dtype);
size_t tensor_dtype_size(TensorDType dtype);
TensorDType tensor_dtype_from_safetensors(const std::string &dtype);
bool tensor_dtype_has_fp32_staging(TensorDType dtype);

struct Tensor {
  size_t ndim = 0;
  size_t shape[5] = {1, 1, 1, 1, 1};
  TensorDType dtype = TensorDType::F32;
  void *storage = nullptr;
  float *buf = nullptr;
  mutable void *gpu_buf = nullptr;

  Tensor() = default;
  explicit Tensor(const std::vector<size_t> &shape_,
                  TensorDType dtype_ = TensorDType::F32);
  Tensor(const std::vector<size_t> &shape_, const float *buf_,
         TensorDType dtype_ = TensorDType::F32);
  ~Tensor();

  size_t num_elem() const;
  size_t elem_size() const;
  size_t num_bytes() const;
  void reshape(const std::vector<size_t> &shape_);
  void sync_fp32_from_storage() const;
  void sync_storage_from_fp32() const;
  void ensure_gpu() const;
  void to_gpu() const;
  void to_cpu() const;
  void zero_host() const;
  void zero_device() const;
};

struct TokenBatch {
  size_t B = 0;
  size_t T = 0;
  size_t n_elem = 0;
  int32_t *buf = nullptr;
  mutable int32_t *gpu_buf = nullptr;
  int32_t *lengths = nullptr;
  mutable int32_t *gpu_lengths = nullptr;

  TokenBatch() = default;
  TokenBatch(size_t batch_size, size_t seq_len);
  ~TokenBatch();

  TokenBatch(const TokenBatch &) = delete;
  TokenBatch &operator=(const TokenBatch &) = delete;
  TokenBatch(TokenBatch &&other) noexcept;
  TokenBatch &operator=(TokenBatch &&other) noexcept;

  void ensure_gpu() const;
  void to_gpu() const;
  void to_cpu() const;
};

struct ExpertSelection {
  size_t B = 0;
  size_t T = 0;
  size_t K = 0;
  std::vector<int32_t> indices;
  std::vector<float> weights;

  ExpertSelection() = default;
  ExpertSelection(size_t batch_size, size_t seq_len, size_t topk);

  void resize(size_t batch_size, size_t seq_len, size_t topk);
  size_t num_rows() const { return B * T; }
  int32_t &index(size_t b, size_t t, size_t k);
  float &weight(size_t b, size_t t, size_t k);
  int32_t index(size_t b, size_t t, size_t k) const;
  float weight(size_t b, size_t t, size_t k) const;
};

typedef Tensor Parameter;
typedef Tensor Activation;
