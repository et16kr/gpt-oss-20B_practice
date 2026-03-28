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

struct Tensor {
  size_t ndim = 0;
  size_t shape[5] = {1, 1, 1, 1, 1};
  TensorDType dtype = TensorDType::F32;
  mutable void *buf = nullptr;

  Tensor() = default;
  explicit Tensor(const std::vector<size_t> &shape_,
                  TensorDType dtype_ = TensorDType::F32);
  ~Tensor();

  Tensor(const Tensor &) = delete;
  Tensor &operator=(const Tensor &) = delete;

  size_t num_elem() const;
  size_t elem_size() const;
  size_t num_bytes() const;
  void reshape(const std::vector<size_t> &shape_);
  void ensure_gpu() const;
  void zero_device() const;
};

struct TokenBatch {
  size_t B = 0;
  size_t T = 0;
  size_t n_elem = 0;
  int32_t *buf = nullptr;
  int32_t *lengths = nullptr;

  TokenBatch() = default;
  TokenBatch(size_t batch_size, size_t seq_len);
  ~TokenBatch();

  TokenBatch(const TokenBatch &) = delete;
  TokenBatch &operator=(const TokenBatch &) = delete;
  TokenBatch(TokenBatch &&other) noexcept;
  TokenBatch &operator=(TokenBatch &&other) noexcept;
};

struct DeviceTokenBatch {
  size_t B = 0;
  size_t T = 0;
  size_t n_elem = 0;
  int32_t *buf = nullptr;
  int32_t *lengths = nullptr;

  DeviceTokenBatch() = default;
  DeviceTokenBatch(size_t batch_size, size_t seq_len);
  ~DeviceTokenBatch();

  DeviceTokenBatch(const DeviceTokenBatch &) = delete;
  DeviceTokenBatch &operator=(const DeviceTokenBatch &) = delete;
  DeviceTokenBatch(DeviceTokenBatch &&other) noexcept;
  DeviceTokenBatch &operator=(DeviceTokenBatch &&other) noexcept;

  void resize(size_t batch_size, size_t seq_len);
  void upload(const TokenBatch &host_batch);
  void zero() const;
};

struct ExpertSelection {
  size_t B = 0;
  size_t T = 0;
  size_t K = 0;
  mutable int32_t *indices = nullptr;
  mutable float *weights = nullptr;

  ExpertSelection() = default;
  ExpertSelection(size_t batch_size, size_t seq_len, size_t topk);
  ~ExpertSelection();

  ExpertSelection(const ExpertSelection &) = delete;
  ExpertSelection &operator=(const ExpertSelection &) = delete;

  void resize(size_t batch_size, size_t seq_len, size_t topk);
  size_t num_rows() const { return B * T; }
  void ensure_gpu() const;
};

typedef Tensor Parameter;
typedef Tensor Activation;
