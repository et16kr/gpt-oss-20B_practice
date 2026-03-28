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
  std::vector<uint8_t> blocks;
  std::vector<uint8_t> scales;
  std::vector<uint8_t> bias_bytes;
  std::vector<float> bias;

  const uint8_t *row_blocks(size_t expert_idx, size_t row_idx) const;
  const uint8_t *row_scales(size_t expert_idx, size_t row_idx) const;
  float row_bias(size_t expert_idx, size_t row_idx) const;
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
