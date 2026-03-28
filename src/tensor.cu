#include "tensor.h"

#include <cstdlib>
#include <cstring>

#include <cuda_fp16.h>

#include "util.h"

namespace {

float fp16_to_float(uint16_t h) {
  const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
  const uint32_t exp = (h >> 10) & 0x1f;
  const uint32_t mant = h & 0x03ff;
  uint32_t bits = 0;

  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      int e = -14;
      uint32_t m = mant;
      while ((m & 0x0400) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x03ff;
      bits = sign | (uint32_t)(e + 127) << 23 | (m << 13);
    }
  } else if (exp == 0x1f) {
    bits = sign | 0x7f800000 | (mant << 13);
  } else {
    bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
  }

  float value = 0.0f;
  memcpy(&value, &bits, sizeof(float));
  return value;
}

float bf16_to_float(uint16_t value) {
  const uint32_t bits = (uint32_t)value << 16;
  float out = 0.0f;
  memcpy(&out, &bits, sizeof(float));
  return out;
}

uint16_t float_to_bf16(float value) {
  uint32_t bits = 0;
  memcpy(&bits, &value, sizeof(float));
  const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
  return (uint16_t)((bits + rounding_bias) >> 16);
}

uint16_t float_to_fp16(float value) {
  const __half half_value = __float2half_rn(value);
  uint16_t raw = 0;
  memcpy(&raw, &half_value, sizeof(uint16_t));
  return raw;
}

void allocate_host_buffers(Tensor *tensor) {
  const size_t numel = tensor->num_elem();
  const size_t bytes = tensor->num_bytes();
  if (tensor->dtype == TensorDType::F32) {
    tensor->buf = (float *)calloc(numel, sizeof(float));
    CHECK_ERROR(tensor->buf != nullptr, "Failed to allocate F32 tensor host buffer");
    tensor->storage = tensor->buf;
    return;
  }

  tensor->storage = calloc(1, bytes);
  CHECK_ERROR(tensor->storage != nullptr, "Failed to allocate native tensor host buffer");
  if (tensor_dtype_has_fp32_staging(tensor->dtype)) {
    tensor->buf = (float *)calloc(numel, sizeof(float));
    CHECK_ERROR(tensor->buf != nullptr, "Failed to allocate tensor fp32 staging buffer");
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

bool tensor_dtype_has_fp32_staging(TensorDType dtype) {
  return dtype == TensorDType::F32 || dtype == TensorDType::F16 ||
         dtype == TensorDType::BF16;
}

Tensor::Tensor(const std::vector<size_t> &shape_, TensorDType dtype_) : dtype(dtype_) {
  reshape(shape_);
}

Tensor::Tensor(const std::vector<size_t> &shape_, const float *buf_, TensorDType dtype_)
    : dtype(dtype_) {
  reshape(shape_);
  if (buf_ != nullptr && buf != nullptr) {
    memcpy(buf, buf_, num_elem() * sizeof(float));
    if (dtype != TensorDType::F32) {
      sync_storage_from_fp32();
    }
  }
}

Tensor::~Tensor() {
  if (storage != nullptr && storage != buf) {
    free(storage);
  }
  if (buf != nullptr) {
    free(buf);
  }
  if (gpu_buf != nullptr) {
    CHECK_CUDA(cudaFree(gpu_buf));
  }
}

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
  const size_t old_numel =
      (storage == nullptr && buf == nullptr) ? 0 : num_elem();
  ndim = shape_.size();
  CHECK_ERROR(ndim <= 5, "Tensor rank %zu exceeds maximum supported rank 5", ndim);

  for (size_t i = 0; i < ndim; ++i) {
    shape[i] = shape_[i];
  }
  for (size_t i = ndim; i < 5; ++i) {
    shape[i] = 1;
  }

  const size_t new_numel = num_elem();
  if (storage == nullptr && buf == nullptr) {
    allocate_host_buffers(this);
    return;
  }

  CHECK_ERROR(old_numel == new_numel,
              "reshape changes tensor size (%zu -> %zu), which is not allowed",
              old_numel, new_numel);
}

void Tensor::sync_fp32_from_storage() const {
  if (dtype == TensorDType::F32 || buf == nullptr || storage == nullptr) {
    return;
  }

  const size_t numel = num_elem();
  if (dtype == TensorDType::BF16) {
    const uint16_t *src = (const uint16_t *)storage;
    for (size_t i = 0; i < numel; ++i) {
      buf[i] = bf16_to_float(src[i]);
    }
    return;
  }

  if (dtype == TensorDType::F16) {
    const uint16_t *src = (const uint16_t *)storage;
    for (size_t i = 0; i < numel; ++i) {
      buf[i] = fp16_to_float(src[i]);
    }
    return;
  }
}

void Tensor::sync_storage_from_fp32() const {
  if (dtype == TensorDType::F32 || buf == nullptr || storage == nullptr) {
    return;
  }

  const size_t numel = num_elem();
  if (dtype == TensorDType::BF16) {
    uint16_t *dst = (uint16_t *)storage;
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = float_to_bf16(buf[i]);
    }
    return;
  }

  if (dtype == TensorDType::F16) {
    uint16_t *dst = (uint16_t *)storage;
    for (size_t i = 0; i < numel; ++i) {
      dst[i] = float_to_fp16(buf[i]);
    }
    return;
  }
}

void Tensor::ensure_gpu() const {
  CHECK_ERROR(storage != nullptr || buf != nullptr,
              "Tensor host buffer must exist before ensure_gpu()");
  if (gpu_buf == nullptr) {
    CHECK_CUDA(cudaMalloc((void **)&gpu_buf, num_bytes()));
    CHECK_CUDA(cudaMemset(gpu_buf, 0, num_bytes()));
  }
}

void Tensor::to_gpu() const {
  CHECK_ERROR(storage != nullptr || buf != nullptr,
              "Tensor host buffer must exist before to_gpu()");
  if (dtype != TensorDType::F32 && buf != nullptr && storage != nullptr) {
    sync_storage_from_fp32();
  }
  ensure_gpu();
  const void *src = (storage != nullptr) ? storage : (const void *)buf;
  CHECK_CUDA(cudaMemcpy(gpu_buf, src, num_bytes(), cudaMemcpyHostToDevice));
}

void Tensor::to_cpu() const {
  CHECK_ERROR((storage != nullptr || buf != nullptr) && gpu_buf != nullptr,
              "Tensor buffers must be allocated before to_cpu()");
  void *dst = (storage != nullptr) ? storage : (void *)buf;
  CHECK_CUDA(cudaMemcpy(dst, gpu_buf, num_bytes(), cudaMemcpyDeviceToHost));
  if (dtype != TensorDType::F32 && buf != nullptr) {
    sync_fp32_from_storage();
  }
}

void Tensor::zero_host() const {
  if (storage != nullptr) {
    memset(storage, 0, num_bytes());
  }
  if (buf != nullptr && storage != buf) {
    memset(buf, 0, num_elem() * sizeof(float));
  }
}

void Tensor::zero_device() const {
  ensure_gpu();
  CHECK_CUDA(cudaMemset(gpu_buf, 0, num_bytes()));
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
  if (gpu_buf != nullptr) {
    CHECK_CUDA(cudaFree(gpu_buf));
  }
  if (lengths != nullptr) {
    free(lengths);
  }
  if (gpu_lengths != nullptr) {
    CHECK_CUDA(cudaFree(gpu_lengths));
  }
}

TokenBatch::TokenBatch(TokenBatch &&other) noexcept {
  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  gpu_buf = other.gpu_buf;
  lengths = other.lengths;
  gpu_lengths = other.gpu_lengths;
  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.gpu_buf = nullptr;
  other.lengths = nullptr;
  other.gpu_lengths = nullptr;
}

TokenBatch &TokenBatch::operator=(TokenBatch &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  if (buf != nullptr) {
    free(buf);
  }
  if (gpu_buf != nullptr) {
    CHECK_CUDA(cudaFree(gpu_buf));
  }
  if (lengths != nullptr) {
    free(lengths);
  }
  if (gpu_lengths != nullptr) {
    CHECK_CUDA(cudaFree(gpu_lengths));
  }

  B = other.B;
  T = other.T;
  n_elem = other.n_elem;
  buf = other.buf;
  gpu_buf = other.gpu_buf;
  lengths = other.lengths;
  gpu_lengths = other.gpu_lengths;

  other.B = 0;
  other.T = 0;
  other.n_elem = 0;
  other.buf = nullptr;
  other.gpu_buf = nullptr;
  other.lengths = nullptr;
  other.gpu_lengths = nullptr;
  return *this;
}

void TokenBatch::ensure_gpu() const {
  if (gpu_buf == nullptr) {
    CHECK_CUDA(cudaMalloc((void **)&gpu_buf, n_elem * sizeof(int32_t)));
  }
  if (gpu_lengths == nullptr) {
    CHECK_CUDA(cudaMalloc((void **)&gpu_lengths, B * sizeof(int32_t)));
  }
}

void TokenBatch::to_gpu() const {
  ensure_gpu();
  CHECK_CUDA(cudaMemcpy(gpu_buf, buf, n_elem * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(
      cudaMemcpy(gpu_lengths, lengths, B * sizeof(int32_t), cudaMemcpyHostToDevice));
}

void TokenBatch::to_cpu() const {
  CHECK_ERROR(gpu_buf != nullptr && gpu_lengths != nullptr,
              "TokenBatch buffers must be allocated before to_cpu()");
  CHECK_CUDA(cudaMemcpy(buf, gpu_buf, n_elem * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));
  CHECK_CUDA(
      cudaMemcpy(lengths, gpu_lengths, B * sizeof(int32_t), cudaMemcpyDeviceToHost));
}

ExpertSelection::ExpertSelection(size_t batch_size, size_t seq_len, size_t topk) {
  resize(batch_size, seq_len, topk);
}

void ExpertSelection::resize(size_t batch_size, size_t seq_len, size_t topk) {
  B = batch_size;
  T = seq_len;
  K = topk;
  indices.assign(B * T * K, 0);
  weights.assign(B * T * K, 0.0f);
}

int32_t &ExpertSelection::index(size_t b, size_t t, size_t k) {
  return indices[(b * T + t) * K + k];
}

float &ExpertSelection::weight(size_t b, size_t t, size_t k) {
  return weights[(b * T + t) * K + k];
}

int32_t ExpertSelection::index(size_t b, size_t t, size_t k) const {
  return indices[(b * T + t) * K + k];
}

float ExpertSelection::weight(size_t b, size_t t, size_t k) const {
  return weights[(b * T + t) * K + k];
}
