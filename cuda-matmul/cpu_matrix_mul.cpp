#include <omp.h>
#include <chrono>
#include <cstdio>
#include <new>

#include "matrix_mul.hpp"

void cpu_matrix_mul(float* a, float* b, float* c, int n, int k, int m) {
  const int max_threads = omp_get_max_threads();
  const int total_iters = m * n;
  const int chunk_size = total_iters / (max_threads * 8) + 1;

  const auto start = std::chrono::steady_clock::now();

  float* bT = new float[k * n];
  if (!bT) {
    std::fprintf(stderr, "Failed to allocate memory for bT.");
    return;
  }
  {
    const int ti = k * n;
#pragma omp parallel for schedule(static)
    for (int idx = 0; idx < ti; ++idx) {
      const int i = idx / k;
      const int j = idx % k;
      bT[i * k + j] = b[j * n + i];
    }
  }

#pragma omp parallel for schedule(dynamic, chunk_size)
  for (int idx = 0; idx < total_iters; ++idx) {
    const int i = idx / n;
    const int j = idx % n;
    float acc = 0.0f;
    const float* a_row = a + i * k;
    const float* bT_row = bT + j * k;

    for (int h = 0; h < k; ++h) {
      acc += a_row[h] * bT_row[h];
    }
    c[i * n + j] = acc;
  }

  delete[] bT;
  const auto end = std::chrono::steady_clock::now();
  const auto time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count()) / 1e6;
  std::fprintf(stdout, "Time: %g\t%g\n", time, time);
}
