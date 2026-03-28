#include "safetensors_loader.h"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>

#include <nlohmann/json.hpp>

#include "util.h"

namespace {
ShardedSafetensorsLoader::TensorInfo parse_tensor_info(const nlohmann::json &tensor) {
  ShardedSafetensorsLoader::TensorInfo info;
  info.dtype = tensor.at("dtype").get<std::string>();
  info.shape = tensor.at("shape").get<std::vector<size_t>>();
  const std::vector<size_t> offsets =
      tensor.at("data_offsets").get<std::vector<size_t>>();
  CHECK_ERROR(offsets.size() == 2, "Invalid data_offsets size in safetensors header");
  info.begin = offsets[0];
  info.end = offsets[1];
  return info;
}

}  // namespace

const uint8_t *QuantizedExpertMatrix::row_blocks(size_t expert_idx, size_t row_idx) const {
  const size_t row_size = groups * bytes_per_block;
  return blocks.data() + ((expert_idx * out_dim) + row_idx) * row_size;
}

const uint8_t *QuantizedExpertMatrix::row_scales(size_t expert_idx, size_t row_idx) const {
  return scales.data() + ((expert_idx * out_dim) + row_idx) * groups;
}

float QuantizedExpertMatrix::row_bias(size_t expert_idx, size_t row_idx) const {
  return bias[(expert_idx * out_dim) + row_idx];
}

ShardedSafetensorsLoader::ShardedSafetensorsLoader(const char *model_dir)
    : model_dir_(model_dir) {
  const std::string index_path = model_dir_ + "/model.safetensors.index.json";
  std::ifstream index_input(index_path);
  CHECK_ERROR(index_input.good(), "Failed to open safetensors index %s",
              index_path.c_str());

  nlohmann::json index_json;
  index_input >> index_json;
  CHECK_ERROR(index_json.contains("weight_map"), "Missing weight_map in %s",
              index_path.c_str());

  const nlohmann::json &weight_map = index_json.at("weight_map");
  std::set<std::string> file_names;
  for (auto it = weight_map.begin(); it != weight_map.end(); ++it) {
    tensor_to_file_[it.key()] = it.value().get<std::string>();
    file_names.insert(it.value().get<std::string>());
  }

  for (const std::string &file_name : file_names) {
    FileInfo info;
    info.path = model_dir_ + "/" + file_name;

    FILE *fp = fopen(info.path.c_str(), "rb");
    CHECK_ERROR(fp != nullptr, "Failed to open model shard %s", info.path.c_str());

    uint64_t header_len = 0;
    CHECK_ERROR(fread(&header_len, sizeof(uint64_t), 1, fp) == 1,
                "Failed to read safetensors header length from %s", info.path.c_str());
    std::string header((size_t)header_len, '\0');
    CHECK_ERROR(fread(&header[0], 1, (size_t)header_len, fp) == header_len,
                "Failed to read safetensors header from %s", info.path.c_str());
    fclose(fp);

    info.data_base = sizeof(uint64_t) + (size_t)header_len;
    nlohmann::json header_json = nlohmann::json::parse(header);
    for (auto it = header_json.begin(); it != header_json.end(); ++it) {
      if (it.key() == "__metadata__") {
        continue;
      }
      info.tensors[it.key()] = parse_tensor_info(it.value());
    }
    files_[file_name] = std::move(info);
  }
}

bool ShardedSafetensorsLoader::has_tensor(const char *name) const {
  return tensor_to_file_.find(name) != tensor_to_file_.end();
}

std::vector<size_t> ShardedSafetensorsLoader::tensor_shape(const char *name) const {
  const FileInfo *file_info = nullptr;
  return get_tensor_info(name, &file_info).shape;
}

const ShardedSafetensorsLoader::TensorInfo &ShardedSafetensorsLoader::get_tensor_info(
    const char *name, const FileInfo **file_info) const {
  auto file_it = tensor_to_file_.find(name);
  CHECK_ERROR(file_it != tensor_to_file_.end(), "Tensor %s not found in checkpoint",
              name);
  auto info_it = files_.find(file_it->second);
  CHECK_ERROR(info_it != files_.end(), "Shard metadata for %s not found", name);
  auto tensor_it = info_it->second.tensors.find(name);
  CHECK_ERROR(tensor_it != info_it->second.tensors.end(),
              "Tensor header for %s not found", name);
  if (file_info != nullptr) {
    *file_info = &info_it->second;
  }
  return tensor_it->second;
}

void ShardedSafetensorsLoader::read_bytes(const FileInfo &file_info,
                                          const TensorInfo &tensor_info, void *dst,
                                          size_t size) const {
  FILE *fp = fopen(file_info.path.c_str(), "rb");
  CHECK_ERROR(fp != nullptr, "Failed to open shard %s", file_info.path.c_str());
  CHECK_ERROR(fseek(fp, (long)(file_info.data_base + tensor_info.begin), SEEK_SET) == 0,
              "Failed to seek tensor in %s", file_info.path.c_str());
  CHECK_ERROR(fread(dst, 1, size, fp) == size, "Failed to read tensor from %s",
              file_info.path.c_str());
  fclose(fp);
}

Parameter *ShardedSafetensorsLoader::load_parameter(
    const char *name, const std::vector<size_t> &expected_shape) const {
  const FileInfo *file_info = nullptr;
  const TensorInfo &info = get_tensor_info(name, &file_info);
  CHECK_ERROR(info.shape == expected_shape,
              "Tensor %s shape mismatch while loading parameter", name);
  CHECK_ERROR(info.dtype == "F32" || info.dtype == "F16" || info.dtype == "BF16",
              "Unsupported dtype %s for %s", info.dtype.c_str(), name);
  const TensorDType dtype = tensor_dtype_from_safetensors(info.dtype);

  size_t numel = 1;
  for (size_t dim : info.shape) {
    numel *= dim;
  }

  Parameter *param = new Parameter(expected_shape, dtype);
  const size_t bytes = numel * tensor_dtype_size(dtype);
  CHECK_ERROR((info.end - info.begin) == bytes, "Tensor %s byte size mismatch", name);
  read_bytes(*file_info, info, param->storage, bytes);
  if (tensor_dtype_has_fp32_staging(dtype)) {
    param->sync_fp32_from_storage();
  }
  return param;
}

QuantizedExpertMatrix ShardedSafetensorsLoader::load_quantized_expert_matrix(
    const char *blocks_name, const char *scales_name, const char *bias_name,
    size_t num_experts, size_t out_dim, size_t in_dim) const {
  const FileInfo *blocks_file = nullptr;
  const FileInfo *scales_file = nullptr;
  const FileInfo *bias_file = nullptr;
  const TensorInfo &blocks_info = get_tensor_info(blocks_name, &blocks_file);
  const TensorInfo &scales_info = get_tensor_info(scales_name, &scales_file);
  const TensorInfo &bias_info = get_tensor_info(bias_name, &bias_file);

  CHECK_ERROR(blocks_info.dtype == "U8", "%s must be U8", blocks_name);
  CHECK_ERROR(scales_info.dtype == "U8", "%s must be U8", scales_name);
  CHECK_ERROR(bias_info.dtype == "BF16" || bias_info.dtype == "F16" || bias_info.dtype == "F32",
              "%s must be floating-point", bias_name);
  CHECK_ERROR(blocks_info.shape.size() == 4, "%s must be rank 4", blocks_name);
  CHECK_ERROR(scales_info.shape.size() == 3, "%s must be rank 3", scales_name);
  CHECK_ERROR(bias_info.shape.size() == 2, "%s must be rank 2", bias_name);
  CHECK_ERROR(blocks_info.shape[0] == num_experts && blocks_info.shape[1] == out_dim,
              "%s shape mismatch", blocks_name);
  CHECK_ERROR(scales_info.shape[0] == num_experts && scales_info.shape[1] == out_dim,
              "%s shape mismatch", scales_name);
  CHECK_ERROR(bias_info.shape[0] == num_experts && bias_info.shape[1] == out_dim,
              "%s shape mismatch", bias_name);
  CHECK_ERROR(blocks_info.shape[2] == scales_info.shape[2],
              "blocks/scales group mismatch for %s", blocks_name);
  CHECK_ERROR(blocks_info.shape[3] == 16, "Expected 16 bytes per MXFP4 block for %s",
              blocks_name);
  CHECK_ERROR(blocks_info.shape[2] * blocks_info.shape[3] * 2 == in_dim,
              "MXFP4 layout mismatch for %s", blocks_name);

  QuantizedExpertMatrix matrix;
  matrix.num_experts = num_experts;
  matrix.out_dim = out_dim;
  matrix.in_dim = in_dim;
  matrix.groups = blocks_info.shape[2];
  matrix.bytes_per_block = blocks_info.shape[3];
  matrix.bias_dtype = tensor_dtype_from_safetensors(bias_info.dtype);
  matrix.blocks.resize(num_experts * out_dim * matrix.groups * matrix.bytes_per_block);
  matrix.scales.resize(num_experts * out_dim * matrix.groups);
  matrix.bias_bytes.resize(num_experts * out_dim * tensor_dtype_size(matrix.bias_dtype));
  matrix.bias.resize(num_experts * out_dim);

  read_bytes(*blocks_file, blocks_info, matrix.blocks.data(), matrix.blocks.size());
  read_bytes(*scales_file, scales_info, matrix.scales.data(), matrix.scales.size());

  size_t bias_numel = num_experts * out_dim;
  read_bytes(*bias_file, bias_info, matrix.bias_bytes.data(), matrix.bias_bytes.size());
  if (matrix.bias_dtype == TensorDType::F32) {
    memcpy(matrix.bias.data(), matrix.bias_bytes.data(), bias_numel * sizeof(float));
  } else {
    const uint16_t *raw = (const uint16_t *)matrix.bias_bytes.data();
    for (size_t i = 0; i < bias_numel; ++i) {
      if (matrix.bias_dtype == TensorDType::BF16) {
        const uint32_t bits = (uint32_t)raw[i] << 16;
        memcpy(&matrix.bias[i], &bits, sizeof(float));
      } else {
        const uint32_t sign = (uint32_t)(raw[i] & 0x8000) << 16;
        const uint32_t exp = (raw[i] >> 10) & 0x1f;
        const uint32_t mant = raw[i] & 0x03ff;
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
        memcpy(&matrix.bias[i], &bits, sizeof(float));
      }
    }
  }
  return matrix;
}
