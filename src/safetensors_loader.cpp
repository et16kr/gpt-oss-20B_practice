#include "safetensors_loader.h"

#include <cstdint>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

void free_device(void *ptr) {
  if (ptr != nullptr) {
    CHECK_CUDA(cudaFree(ptr));
  }
}

}  // namespace

QuantizedExpertMatrix::~QuantizedExpertMatrix() { free_gpu(); }

QuantizedExpertMatrix::QuantizedExpertMatrix(QuantizedExpertMatrix &&other) noexcept {
  num_experts = other.num_experts;
  out_dim = other.out_dim;
  in_dim = other.in_dim;
  groups = other.groups;
  bytes_per_block = other.bytes_per_block;
  blocks_bytes = other.blocks_bytes;
  scales_bytes = other.scales_bytes;
  bias_bytes = other.bias_bytes;
  blocks = other.blocks;
  scales = other.scales;
  bias = other.bias;

  other.num_experts = 0;
  other.out_dim = 0;
  other.in_dim = 0;
  other.groups = 0;
  other.bytes_per_block = 16;
  other.blocks_bytes = 0;
  other.scales_bytes = 0;
  other.bias_bytes = 0;
  other.blocks = nullptr;
  other.scales = nullptr;
  other.bias = nullptr;
}

QuantizedExpertMatrix &QuantizedExpertMatrix::operator=(QuantizedExpertMatrix &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  free_gpu();

  num_experts = other.num_experts;
  out_dim = other.out_dim;
  in_dim = other.in_dim;
  groups = other.groups;
  bytes_per_block = other.bytes_per_block;
  blocks_bytes = other.blocks_bytes;
  scales_bytes = other.scales_bytes;
  bias_bytes = other.bias_bytes;
  blocks = other.blocks;
  scales = other.scales;
  bias = other.bias;

  other.num_experts = 0;
  other.out_dim = 0;
  other.in_dim = 0;
  other.groups = 0;
  other.bytes_per_block = 16;
  other.blocks_bytes = 0;
  other.scales_bytes = 0;
  other.bias_bytes = 0;
  other.blocks = nullptr;
  other.scales = nullptr;
  other.bias = nullptr;
  return *this;
}

void QuantizedExpertMatrix::free_gpu() {
  free_device(blocks);
  free_device(scales);
  free_device(bias);
  blocks = nullptr;
  scales = nullptr;
  bias = nullptr;
  blocks_bytes = 0;
  scales_bytes = 0;
  bias_bytes = 0;
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

  std::vector<uint8_t> scratch(bytes);
  read_bytes(*file_info, info, scratch.data(), bytes);
  CHECK_CUDA(cudaMemcpy(param->buf, scratch.data(), bytes, cudaMemcpyHostToDevice));
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
  CHECK_ERROR(bias_info.dtype == "BF16", "%s must be BF16", bias_name);
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
  matrix.blocks_bytes = num_experts * out_dim * matrix.groups * matrix.bytes_per_block;
  matrix.scales_bytes = num_experts * out_dim * matrix.groups;
  matrix.bias_bytes = num_experts * out_dim * sizeof(uint16_t);

  std::vector<uint8_t> host_blocks(matrix.blocks_bytes);
  std::vector<uint8_t> host_scales(matrix.scales_bytes);
  std::vector<uint16_t> host_bias(num_experts * out_dim);
  read_bytes(*blocks_file, blocks_info, host_blocks.data(), host_blocks.size());
  read_bytes(*scales_file, scales_info, host_scales.data(), host_scales.size());
  read_bytes(*bias_file, bias_info, host_bias.data(), matrix.bias_bytes);

  CHECK_CUDA(cudaMalloc((void **)&matrix.blocks, matrix.blocks_bytes));
  CHECK_CUDA(cudaMalloc((void **)&matrix.scales, matrix.scales_bytes));
  CHECK_CUDA(cudaMalloc((void **)&matrix.bias, matrix.bias_bytes));
  CHECK_CUDA(cudaMemcpy(matrix.blocks, host_blocks.data(), matrix.blocks_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(matrix.scales, host_scales.data(), matrix.scales_bytes,
                        cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(matrix.bias, host_bias.data(), matrix.bias_bytes,
                        cudaMemcpyHostToDevice));
  return matrix;
}
