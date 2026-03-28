#include "tensor.h"

#include <cstdlib>

#include "util.h"

namespace {

void free_device(void *ptr) {
  if (ptr != nullptr) {
    CHECK_CUDA(cudaFree(ptr));
  }
}

}  // namespace

const char *tensor_dtype_name(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::F32:
      return "F32";
    case TensorDType::F16:
      return "F16";
    case TensorDType::BF16:
      return "BF16";
    case TensorDType::U8:
      return "U8";
  }
  return "UNKNOWN";
}

size_t tensor_dtype_size(TensorDType dtype) {
  switch (dtype) {
    case TensorDType::F32:
      return sizeof(float);
    case TensorDType::F16:
    case TensorDType::BF16:
      return sizeof(uint16_t);
    case TensorDType::U8:
      return sizeof(uint8_t);
  }
  return 0;
}

TensorDType tensor_dtype_from_safetensors(const std::string &dtype) {
  if (dtype == "F32") {
    return TensorDType::F32;
  }
  if (dtype == "F16") {
    return TensorDType::F16;
  }
  if (dtype == "BF16") {
    return TensorDType::BF16;
  }
  if (dtype == "U8") {
    return TensorDType::U8;
  }
  CHECK_ERROR(false, "Unsupported safetensors dtype %s", dtype.c_str());
  return TensorDType::F32;
}

Tensor::Tensor(const std::vector<size_t> &shape_, TensorDType dtype_) : dtype(dtype_) {
  reshape(shape_);
}

Tensor::~Tensor() { free_device(buf); }

size_t Tensor::num_elem() const {
  size_t n = 1;
  for (size_t i = 0; i < ndim; ++i) {
    n *= shape[i];
  }
  return n;
}

size_t Tensor::elem_size() const { return tensor_dtype_size(dtype); }

size_t Tensor::num_bytes() const { return num_elem() * elem_size(); }

void Tensor::reshape(const std::vector<size_t> &shape_) {
  const size_t old_numel = (buf == nullptr) ? 0 : num_elem();
  ndim = shape_.size();
  CHECK_ERROR(ndim <= 5, "Tensor rank %zu exceeds maximum supported rank 5", ndim);

  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = shape_[i];
  }
  for (size_t i = ndim; i < 5; ++i) {
    shape[i] = 1;
  }

  if (buf == nullptr) {
    ensure_gpu();
    return;
  }

  const size_t new_numel = num_elem();
  CHECK_ERROR(old_numel == new_numel,
              "reshape changes tensor size (%zu -> %zu), which is not allowed",
              old_numel, new_numel);
}

void Tensor::ensure_gpu() const {
  if (buf == nullptr) {
    CHECK_CUDA(cudaMalloc((void **)&buf, num_bytes()));
    CHECK_CUDA(cudaMemset(buf, 0, num_bytes()));
  }
}

void Tensor::zero_device() const {
  ensure_gpu();
  CHECK_CUDA(cudaMemset(buf, 0, num_bytes()));
}

TokenBatch::TokenBatch(size_t batch_size, size_t seq_len)
    : B(batch_size), T(seq_len), n_elem(batch_size * seq_len) {
  buf = (int32_t *)malloc(n_elem * sizeof(int32_t));
  CHECK_ERROR(buf != nullptr, "Failed to allocate host token buffer");
  lengths = (int32_t *)malloc(B * sizeof(int32_t));
  CHECK_ERROR(lengths != nullptr, "Failed to allocate host length buffer");
}

TokenBatch::~TokenBatch() {
  if (buf != nullptr) {
    free(buf);
  }
  if (lengths != nullptr) {
    free(lengths);
  }
}

TokenBatch::TokenBatch(TokenBatch &&other) noexcept {
  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  lengths = other.lengths;
  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.lengths = nullptr;
}

TokenBatch &TokenBatch::operator=(TokenBatch &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (buf != nullptr) {
    free(buf);
  }
  if (lengths != nullptr) {
    free(lengths);
  }

  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  lengths = other.lengths;

  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.lengths = nullptr;
  return *this;
}

DeviceTokenBatch::DeviceTokenBatch(size_t batch_size, size_t seq_len) {
  resize(batch_size, seq_len);
}

DeviceTokenBatch::~DeviceTokenBatch() {
  free_device(buf);
  free_device(lengths);
}

DeviceTokenBatch::DeviceTokenBatch(DeviceTokenBatch &&other) noexcept {
  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  lengths = other.lengths;
  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.lengths = nullptr;
}

DeviceTokenBatch &DeviceTokenBatch::operator=(DeviceTokenBatch &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  free_device(buf);
  free_device(lengths);

  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  lengths = other.lengths;

  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.lengths = nullptr;
  return *this;
}

void DeviceTokenBatch::resize(size_t batch_size, size_t seq_len) {
  if (buf != nullptr || lengths != nullptr) {
    free_device(buf);
    free_device(lengths);
    buf = nullptr;
    lengths = nullptr;
  }

  B = batch_size;
  T = seq_len;
  n_elem = B * T;
  CHECK_CUDA(cudaMalloc((void **)&buf, n_elem * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc((void **)&lengths, B * sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(buf, 0, n_elem * sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(lengths, 0, B * sizeof(int32_t)));
}

void DeviceTokenBatch::upload(const TokenBatch &host_batch) {
  CHECK_ERROR(B == host_batch.B && T == host_batch.T,
              "DeviceTokenBatch upload shape mismatch");
  CHECK_CUDA(cudaMemcpy(buf, host_batch.buf, n_elem * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(lengths, host_batch.lengths, B * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
}

void DeviceTokenBatch::zero() const {
  CHECK_CUDA(cudaMemset(buf, 0, n_elem * sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(lengths, 0, B * sizeof(int32_t)));
}

ExpertSelection::ExpertSelection(size_t batch_size, size_t seq_len, size_t topk) {
  resize(batch_size, seq_len, topk);
}

ExpertSelection::~ExpertSelection() {
  free_device(indices);
  free_device(weights);
}

void ExpertSelection::resize(size_t batch_size, size_t seq_len, size_t topk) {
  free_device(indices);
  free_device(weights);
  indices = nullptr;
  weights = nullptr;
  B = batch_size;
  T = seq_len;
  K = topk;
  ensure_gpu();
}

void ExpertSelection::ensure_gpu() const {
  const size_t n = B * T * K;
  if (indices == nullptr) {
    CHECK_CUDA(cudaMalloc((void **)&indices, n * sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(indices, 0, n * sizeof(int32_t)));
  }
  if (weights == nullptr) {
    CHECK_CUDA(cudaMalloc((void **)&weights, n * sizeof(float)));
    CHECK_CUDA(cudaMemset(weights, 0, n * sizeof(float)));
  }
}
