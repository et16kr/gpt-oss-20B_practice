#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "tensor.h"

struct QuantizedExpertMatrix {
  size_t num_experts = 0;
  size_t out_dim = 0;
  size_t in_dim = 0;
  size_t groups = 0;
  size_t bytes_per_block = 16;
  TensorDType bias_dtype = TensorDType::F32;
  size_t blocks_bytes = 0;
  size_t scales_bytes = 0;
  size_t bias_bytes = 0;
  uint8_t *blocks = nullptr;
  uint8_t *scales = nullptr;
  void *bias = nullptr;

  QuantizedExpertMatrix() = default;
  ~QuantizedExpertMatrix();

  QuantizedExpertMatrix(const QuantizedExpertMatrix &) = delete;
  QuantizedExpertMatrix &operator=(const QuantizedExpertMatrix &) = delete;
  QuantizedExpertMatrix(QuantizedExpertMatrix &&other) noexcept;
  QuantizedExpertMatrix &operator=(QuantizedExpertMatrix &&other) noexcept;

  void free_gpu();
};

class ShardedSafetensorsLoader {
 public:
  struct TensorInfo {
    std::string dtype;
    std::vector<size_t> shape;
    size_t begin = 0;
    size_t end = 0;
  };

  struct FileInfo {
    std::string path;
    std::unordered_map<std::string, TensorInfo> tensors;
    size_t data_base = 0;
  };

  explicit ShardedSafetensorsLoader(const char *model_dir);
  ~ShardedSafetensorsLoader() = default;

  ShardedSafetensorsLoader(const ShardedSafetensorsLoader &) = delete;
  ShardedSafetensorsLoader &operator=(const ShardedSafetensorsLoader &) = delete;

  bool has_tensor(const char *name) const;
  std::vector<size_t> tensor_shape(const char *name) const;
  Parameter *load_parameter(const char *name,
                            const std::vector<size_t> &expected_shape) const;
  QuantizedExpertMatrix load_quantized_expert_matrix(
      const char *blocks_name, const char *scales_name, const char *bias_name,
      size_t num_experts, size_t out_dim, size_t in_dim) const;

 private:
  const TensorInfo &get_tensor_info(const char *name,
                                    const FileInfo **file_info) const;
  void read_bytes(const FileInfo &file_info, const TensorInfo &tensor_info, void *dst,
                  size_t size) const;

  std::string model_dir_;
  std::unordered_map<std::string, std::string> tensor_to_file_;
  std::unordered_map<std::string, FileInfo> files_;
};
