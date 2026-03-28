#include "util.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <vector>

double get_time() {
  struct timespec tv;
  clock_gettime(CLOCK_MONOTONIC, &tv);
  return tv.tv_sec + tv.tv_nsec * 1e-9;
}

void *read_binary(const char *filename, size_t *size) {
  FILE *f = fopen(filename, "rb");
  CHECK_ERROR(f != nullptr, "Failed to open %s", filename);
  fseek(f, 0, SEEK_END);
  size_t size_ = (size_t)ftell(f);
  rewind(f);

  void *buf = malloc(size_);
  CHECK_ERROR(buf != nullptr, "Failed to allocate %zu bytes for %s", size_,
              filename);

  size_t ret = fread(buf, 1, size_, f);
  fclose(f);
  CHECK_ERROR(ret == size_, "Failed to read %zu bytes from %s", size_,
              filename);

  if (size != nullptr) {
    *size = size_;
  }
  return buf;
}

void write_binary(const char *filename, const void *buf, size_t size) {
  FILE *f = fopen(filename, "wb");
  CHECK_ERROR(f != nullptr, "Failed to open %s for writing", filename);
  size_t ret = fwrite(buf, 1, size, f);
  fclose(f);
  CHECK_ERROR(ret == size, "Failed to write %zu bytes to %s", size, filename);
}

int validate_buffer(const float *output, const float *answer, size_t n,
                    float atol, float rtol) {
  for (size_t i = 0; i < n; ++i) {
    float abs_err = fabsf(output[i] - answer[i]);
    if (std::isnan(output[i]) ||
        abs_err > atol + rtol * fabsf(answer[i])) {
      return (int)i;
    }
  }
  return -1;
}

void print_last_token_topk(const Tensor *logits, size_t batch_size,
                           size_t seq_len, int k) {
  CHECK_ERROR(k > 0, "k must be positive");
  const size_t row = (batch_size - 1) * seq_len + (seq_len - 1);
  const float *ptr = logits->buf + row * logits->shape[2];

  std::vector<std::pair<float, int>> top;
  top.reserve(logits->shape[2]);
  for (size_t i = 0; i < logits->shape[2]; ++i) {
    top.push_back({ptr[i], (int)i});
  }

  if ((size_t)k < top.size()) {
    std::partial_sort(top.begin(), top.begin() + k, top.end(),
                      [](const std::pair<float, int> &a,
                         const std::pair<float, int> &b) {
                        return a.first > b.first;
                      });
  } else {
    std::sort(top.begin(), top.end(),
              [](const std::pair<float, int> &a,
                 const std::pair<float, int> &b) { return a.first > b.first; });
    k = (int)top.size();
  }

  printf("Top-%d predictions at last position:\n", k);
  for (int i = 0; i < k; ++i) {
    printf("  rank %d: token_id=%d logit=%f\n", i + 1, top[i].second,
           top[i].first);
  }
}
