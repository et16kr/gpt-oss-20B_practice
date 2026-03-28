#pragma once

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

#include "tensor.h"

namespace cuda_common {

union FloatBits {
  uint32_t u32;
  float f32;
};

__host__ __device__ inline float bits_to_float(uint32_t bits) {
  FloatBits value;
  value.u32 = bits;
  return value.f32;
}

__host__ __device__ inline uint32_t float_to_bits(float value) {
  FloatBits bits;
  bits.f32 = value;
  return bits.u32;
}

__host__ __device__ inline float bf16_to_float(uint16_t value) {
  return bits_to_float((uint32_t)value << 16);
}

__host__ __device__ inline uint16_t float_to_bf16(float value) {
  const uint32_t bits = float_to_bits(value);
  const uint32_t rounding_bias = 0x7FFFu + ((bits >> 16) & 1u);
  return (uint16_t)((bits + rounding_bias) >> 16);
}

__host__ __device__ inline float fp16_to_float(uint16_t h) {
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
  return bits_to_float(bits);
}

__host__ __device__ inline float sigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ inline float load_tensor_value(const void *base, TensorDType dtype,
                                          size_t index) {
  switch (dtype) {
    case TensorDType::F32:
      return ((const float *)base)[index];
    case TensorDType::F16:
      return fp16_to_float(((const uint16_t *)base)[index]);
    case TensorDType::BF16:
      return bf16_to_float(((const uint16_t *)base)[index]);
    case TensorDType::U8:
      return (float)((const uint8_t *)base)[index];
  }
  return 0.0f;
}

__device__ inline void store_tensor_value(void *base, TensorDType dtype,
                                          size_t index, float value) {
  switch (dtype) {
    case TensorDType::F32:
      ((float *)base)[index] = value;
      return;
    case TensorDType::F16:
      // Activations are not stored as F16 in this project.
      return;
    case TensorDType::BF16:
      ((uint16_t *)base)[index] = float_to_bf16(value);
      return;
    case TensorDType::U8:
      ((uint8_t *)base)[index] = (uint8_t)value;
      return;
  }
}

__host__ __device__ inline size_t ceil_div(size_t n, size_t d) {
  return (n + d - 1) / d;
}

}  // namespace cuda_common

