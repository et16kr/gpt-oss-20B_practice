#include "generation.h"

#include <cuda_profiler_api.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "model.h"
#include "util.h"

namespace {

size_t effective_context_len(const CliOptions &options, const GptOssConfig &cfg) {
  size_t ctx = (options.context_len > 0) ? (size_t)options.context_len : (size_t)512;
  if (ctx > cfg.max_position_embeddings) {
    ctx = cfg.max_position_embeddings;
  }
  return ctx;
}

int pick_pad_token(const GptOssConfig &cfg) {
  if (cfg.pad_token_id >= 0) {
    return cfg.pad_token_id;
  }
  if (cfg.primary_eos_token_id() >= 0) {
    return cfg.primary_eos_token_id();
  }
  return cfg.bos_token_id;
}

std::vector<std::vector<int>> batch_to_sequences(const TokenBatch &batch,
                                                 size_t context_len) {
  std::vector<std::vector<int>> sequences(batch.B);
  for (size_t b = 0; b < batch.B; ++b) {
    const size_t valid = (batch.lengths != nullptr) ? (size_t)batch.lengths[b] : batch.T;
    CHECK_ERROR(valid > 0 && valid <= batch.T, "Invalid sequence length for batch row %zu", b);
    const size_t used = std::min(valid, context_len);
    sequences[b].resize(used);
    const size_t start = valid - used;
    for (size_t t = 0; t < used; ++t) {
      sequences[b][t] = batch.buf[b * batch.T + start + t];
    }
  }
  return sequences;
}

TokenBatch make_context_window_batch(const std::vector<std::vector<int>> &sequences,
                                     size_t context_len, int pad_token_id) {
  CHECK_ERROR(!sequences.empty(), "sequences must not be empty");

  const size_t batch_size = sequences.size();
  TokenBatch batch(batch_size, context_len);
  for (size_t i = 0; i < batch.n_elem; ++i) {
    batch.buf[i] = pad_token_id;
  }
  for (size_t b = 0; b < batch_size; ++b) {
    const size_t used = std::min(sequences[b].size(), context_len);
    CHECK_ERROR(used > 0, "each sequence must contain at least one token");
    const size_t start = sequences[b].size() - used;
    batch.lengths[b] = (int32_t)used;
    for (size_t t = 0; t < used; ++t) {
      batch.buf[b * context_len + t] = sequences[b][start + t];
    }
  }
  return batch;
}

void maybe_warmup_batch(const CliOptions &options,
                        const std::vector<std::vector<int>> &sequences,
                        size_t context_len, int pad_token_id, bool *did_warmup) {
  if (!options.run_warmup || *did_warmup) {
    return;
  }

  TokenBatch batch = make_context_window_batch(sequences, context_len, pad_token_id);
  DeviceTokenBatch device_batch(batch.B, batch.T);
  device_batch.upload(batch);
  alloc_activations(batch.B, batch.T);
  Tensor logits({batch.B, batch.T, model_config().vocab_size});
  gpt_oss_forward(&device_batch, &logits);
  CHECK_CUDA(cudaDeviceSynchronize());
  *did_warmup = true;
}

void maybe_start_cuda_profiler_range(const CliOptions &options) {
  if (!options.cuda_profiler_range) {
    return;
  }
  CHECK_CUDA(cudaProfilerStart());
}

void maybe_stop_cuda_profiler_range(const CliOptions &options) {
  if (!options.cuda_profiler_range) {
    return;
  }
  CHECK_CUDA(cudaProfilerStop());
}

void write_token_sequences(const char *path, const std::vector<std::vector<int>> &sequences,
                           int pad_token_id) {
  std::ofstream output(path, std::ios::binary);
  CHECK_ERROR(output.good(), "failed to open %s for writing", path);

  const int32_t B = (int32_t)sequences.size();
  int32_t T = 0;
  for (const std::vector<int> &sequence : sequences) {
    T = std::max(T, (int32_t)sequence.size());
  }

  output.write(reinterpret_cast<const char *>(&B), sizeof(int32_t));
  output.write(reinterpret_cast<const char *>(&T), sizeof(int32_t));

  std::vector<int32_t> lengths(B, 0);
  std::vector<int32_t> padded((size_t)B * (size_t)std::max(T, 0), pad_token_id);
  for (int32_t b = 0; b < B; ++b) {
    lengths[b] = (int32_t)sequences[(size_t)b].size();
    for (int32_t t = 0; t < lengths[b]; ++t) {
      padded[(size_t)b * (size_t)T + (size_t)t] = sequences[(size_t)b][(size_t)t];
    }
  }

  output.write(reinterpret_cast<const char *>(lengths.data()), sizeof(int32_t) * (size_t)B);
  if (T > 0) {
    output.write(reinterpret_cast<const char *>(padded.data()),
                 sizeof(int32_t) * padded.size());
  }
  CHECK_ERROR(output.good(), "failed to write generated token file %s", path);
}

std::vector<std::vector<int>> download_generated_sequences(const int32_t *generated_tokens,
                                                           const int32_t *generated_lengths,
                                                           size_t batch_size,
                                                           size_t max_new_tokens) {
  std::vector<int32_t> host_lengths(batch_size, 0);
  std::vector<int32_t> host_tokens(batch_size * max_new_tokens, 0);
  CHECK_CUDA(cudaMemcpy(host_lengths.data(), generated_lengths,
                        batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost));
  if (max_new_tokens > 0) {
    CHECK_CUDA(cudaMemcpy(host_tokens.data(), generated_tokens,
                          batch_size * max_new_tokens * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));
  }

  std::vector<std::vector<int>> sequences(batch_size);
  for (size_t b = 0; b < batch_size; ++b) {
    CHECK_ERROR(host_lengths[b] >= 0 && (size_t)host_lengths[b] <= max_new_tokens,
                "Invalid generated length for batch row %zu", b);
    sequences[b].assign(host_tokens.begin() + (ptrdiff_t)(b * max_new_tokens),
                        host_tokens.begin() + (ptrdiff_t)(b * max_new_tokens + host_lengths[b]));
  }
  return sequences;
}

}  // namespace

void run_generation_mode(const CliOptions &options) {
  TokenBatch prompts = load_tokens(options.token_input_path.c_str());
  const size_t context_len = effective_context_len(options, model_config());
  const int pad_token_id = pick_pad_token(model_config());

  std::vector<std::vector<int>> sequences = batch_to_sequences(prompts, context_len);
  bool did_warmup = false;
  maybe_warmup_batch(options, sequences, context_len, pad_token_id, &did_warmup);

  TokenBatch initial_window = make_context_window_batch(sequences, context_len, pad_token_id);
  DeviceTokenBatch device_window(initial_window.B, initial_window.T);
  device_window.upload(initial_window);

  alloc_activations(device_window.B, device_window.T);
  Tensor logits({device_window.B, device_window.T, model_config().vocab_size});

  int32_t *next_tokens = nullptr;
  int32_t *generated_tokens = nullptr;
  int32_t *generated_lengths = nullptr;
  uint8_t *finished = nullptr;
  CHECK_CUDA(cudaMalloc((void **)&next_tokens, device_window.B * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc((void **)&generated_lengths, device_window.B * sizeof(int32_t)));
  CHECK_CUDA(cudaMalloc((void **)&finished, device_window.B * sizeof(uint8_t)));
  CHECK_CUDA(cudaMemset(next_tokens, 0, device_window.B * sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(generated_lengths, 0, device_window.B * sizeof(int32_t)));
  CHECK_CUDA(cudaMemset(finished, 0, device_window.B * sizeof(uint8_t)));
  if (options.max_new_tokens > 0) {
    CHECK_CUDA(cudaMalloc((void **)&generated_tokens,
                          device_window.B * (size_t)options.max_new_tokens *
                              sizeof(int32_t)));
    CHECK_CUDA(cudaMemset(generated_tokens, 0,
                          device_window.B * (size_t)options.max_new_tokens *
                              sizeof(int32_t)));
  }

  maybe_start_cuda_profiler_range(options);
  double total_elapsed = 0.0;
  for (int step = 0; step < options.max_new_tokens; ++step) {
    const double st = get_time();
    gpt_oss_forward(&device_window, &logits);
    select_next_tokens(&device_window, &logits, next_tokens);
    append_next_tokens(&device_window, next_tokens, generated_tokens, generated_lengths,
                       finished, (size_t)options.max_new_tokens);
    CHECK_CUDA(cudaDeviceSynchronize());
    const double et = get_time();
    total_elapsed += et - st;
  }
  maybe_stop_cuda_profiler_range(options);

  std::vector<std::vector<int>> completions = download_generated_sequences(
      generated_tokens, generated_lengths, device_window.B, (size_t)options.max_new_tokens);

  size_t generated = 0;
  for (const std::vector<int> &tokens : completions) {
    generated += tokens.size();
  }
  if (generated > 0 && total_elapsed > 0.0) {
    printf("Generated %zu tokens across %zu prompts in %.6f sec (%.3f tokens/sec)\n",
           generated, completions.size(), total_elapsed, (double)generated / total_elapsed);
  }

  write_token_sequences(options.token_output_path.c_str(), completions, pad_token_id);

  CHECK_CUDA(cudaFree(next_tokens));
  CHECK_CUDA(cudaFree(generated_lengths));
  CHECK_CUDA(cudaFree(finished));
  if (generated_tokens != nullptr) {
    CHECK_CUDA(cudaFree(generated_tokens));
  }
}

void run_forward_only_mode(const CliOptions &options) {
  printf("=============================================\n");
  printf(" gpt-oss-20B Practice (Forward Mode)\n");
  printf("---------------------------------------------\n");
  printf(" Token input      : %s\n", options.token_input_path.c_str());
  printf(" Model dir        : %s\n", options.model_dir.c_str());
  printf(" Logits output    : %s\n", options.logits_output_path.c_str());
  printf(" Warm-up          : %s\n", options.run_warmup ? "ON" : "OFF");
  printf("=============================================\n\n");

  TokenBatch host_tokens = load_tokens(options.token_input_path.c_str());
  DeviceTokenBatch device_tokens(host_tokens.B, host_tokens.T);
  device_tokens.upload(host_tokens);
  alloc_activations(device_tokens.B, device_tokens.T);
  Tensor logits({device_tokens.B, device_tokens.T, model_config().vocab_size});

  if (options.run_warmup) {
    gpt_oss_forward(&device_tokens, &logits);
    CHECK_CUDA(cudaDeviceSynchronize());
  }

  maybe_start_cuda_profiler_range(options);
  const double st = get_time();
  gpt_oss_forward(&device_tokens, &logits);
  CHECK_CUDA(cudaDeviceSynchronize());
  const double et = get_time();
  maybe_stop_cuda_profiler_range(options);

  printf("Elapsed time: %.6f sec\n", et - st);
  printf("Throughput  : %.3f tokens/sec\n",
         (double)(device_tokens.B * device_tokens.T) / (et - st));

  std::vector<float> host_logits(logits.num_elem(), 0.0f);
  CHECK_CUDA(cudaMemcpy(host_logits.data(), logits.buf, logits.num_bytes(),
                        cudaMemcpyDeviceToHost));
  print_last_token_topk(host_logits.data(), device_tokens.B, device_tokens.T,
                        logits.shape[2], 5);
  write_binary(options.logits_output_path.c_str(), host_logits.data(),
               host_logits.size() * sizeof(float));
}
