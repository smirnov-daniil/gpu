#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <chrono>

#include "matrix_mul.hpp"
#include "hyperparameters.hpp"

enum class optimization {
  tile,
  vector
};

#define cuda_assert(error)                                   \
  if (error != cudaSuccess) {                                \
    std::fprintf(stderr, "%s\n", cudaGetErrorString(error)); \
    goto Error;                                              \
  }

__global__ void kernel_tile(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  int N, int K, int M)
{
  __shared__ float aTile[TILE_M][TILE_K];
  __shared__ float bTile[TILE_K][TILE_N];

  const int row = blockIdx.y * TILE_M + threadIdx.y;
  const int col = blockIdx.x * TILE_N + threadIdx.x;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int numThreads = blockDim.x * blockDim.y;

  float acc = 0.0f;
  const int phases = (K + TILE_K - 1) / TILE_K;

  for (int ph = 0; ph < phases; ph++) {
    const int baseK = ph * TILE_K;

#pragma unroll
    for (int idx = tid; idx < TILE_M * TILE_K; idx += numThreads) {
      const int y = idx / TILE_K;
      const int k = idx % TILE_K;
      const int gRow = blockIdx.y * TILE_M + y;
      const int gCol = baseK + k;

      if (gRow < M && gCol < K) {
        aTile[y][k] = __ldg(&A[gRow * K + gCol]);
      }
      else {
        aTile[y][k] = 0.0f;
      }
    }
    __syncthreads();


#pragma unroll
    for (int idx = tid; idx < TILE_K * TILE_N; idx += numThreads) {
      const int k = idx / TILE_N;
      const int x = idx % TILE_N;
      const int gRow = baseK + k;
      const int gCol = blockIdx.x * TILE_N + x;

      if (gRow < K && gCol < N) {
        bTile[k][x] = __ldg(&B[gRow * N + gCol]);
      }
      else {
        bTile[k][x] = 0.0f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      acc += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
    }
    __syncthreads();
  }

  if (row < M && col < N) {
    C[row * N + col] = acc;
  }
}


__global__ void kernel_vector(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  int N, int K, int M)
{
  __shared__ VEC_Y_TYPE aTile[TILE_M][TILE_K];
  __shared__ VEC_X_TYPE bTile[TILE_K][TILE_N];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int numThreads = blockDim.x * blockDim.y;

  const int blockRow = blockIdx.y * (TILE_M * VEC_Y);
  const int blockCol = blockIdx.x * (TILE_N * VEC_X);
  const int gRow0 = blockRow + threadIdx.y * VEC_Y;
  const int gCol0 = blockCol + threadIdx.x * VEC_X;

  VEC_X_TYPE acc[VEC_Y];
  for (int i = 0; i < VEC_Y; i++) {
    acc[i] = { 0 };
  }

  const int phases = (K + TILE_K - 1) / TILE_K;
  for (int ph = 0; ph < phases; ph++) {
    const int baseK = ph * TILE_K;

    const int aTileSz = TILE_M * TILE_K;
    for (int idx = tid; idx < aTileSz; idx += numThreads) {
      const int y = idx / TILE_K;
      const int k = idx % TILE_K;
      const int gCol = baseK + k;
      VEC_Y_TYPE aVec = { 0.0f };
      if (gCol < K) {
#pragma unroll
        for (int i = 0; i < VEC_Y; ++i) {
          const int gRow = blockRow + y * VEC_Y + i;
          if (gRow < M)
            reinterpret_cast<float*>(&aVec)[i] = __ldg(&A[gRow * K + gCol]);
        }
      }
      aTile[y][k] = aVec;
    }
    __syncthreads();

    const int bTileSz = TILE_K * TILE_N;
    for (int idx = tid; idx < bTileSz; idx += numThreads) {
      const int k = idx / TILE_N;
      const int x = idx % TILE_N;
      const int gRow = baseK + k;
      VEC_X_TYPE bVec = { 0.0f };
      if (gRow < K) {
#pragma unroll
        for (int j = 0; j < VEC_X; ++j) {
          const int gCol = blockCol + x * VEC_X + j;
          if (gCol < N)
            reinterpret_cast<float*>(&bVec)[j] = __ldg(&B[gRow * N + gCol]);
        }
      }
      bTile[k][x] = bVec;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      const VEC_Y_TYPE aVec = aTile[threadIdx.y][k];
      const VEC_X_TYPE bVec = bTile[k][threadIdx.x];
#pragma unroll
      for (int i = 0; i < VEC_Y; ++i) {
        const float aVal = reinterpret_cast<const float*>(&aVec)[i];
#pragma unroll
        for (int j = 0; j < VEC_X; ++j) {
          reinterpret_cast<float*>(&acc[i])[j] += aVal * reinterpret_cast<const float*>(&bVec)[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < VEC_Y; ++i) {
    const int row = gRow0 + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < VEC_X; ++j) {
        const int col = gCol0 + j;
        if (col < N) {
          C[row * N + col] = reinterpret_cast<float*>(&acc[i])[j];
        }
      }
    }
  }
}

__global__ void kernel_tile_padded(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const int N, const int K, const int M)
{
  __shared__ float aTile[TILE_M][TILE_K];
  __shared__ float bTile[TILE_K][TILE_N];

  const int row = blockIdx.y * TILE_M + threadIdx.y;
  const int col = blockIdx.x * TILE_N + threadIdx.x;

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int numThreads = blockDim.x * blockDim.y;

  float acc = 0.0f;
  const int phases = K / TILE_K;

  for (int ph = 0; ph < phases; ph++) {
    const int baseK = ph * TILE_K;

#pragma unroll
    for (int idx = tid; idx < TILE_M * TILE_K; idx += numThreads) {
      const int y = idx / TILE_K;
      const int k = idx % TILE_K;
      const int gRow = blockIdx.y * TILE_M + y;
      const int gCol = baseK + k;

      aTile[y][k] = __ldg(&A[gRow * K + gCol]);
    }
    __syncthreads();


#pragma unroll
    for (int idx = tid; idx < TILE_K * TILE_N; idx += numThreads) {
      const int k = idx / TILE_N;
      const int x = idx % TILE_N;
      const int gRow = baseK + k;
      const int gCol = blockIdx.x * TILE_N + x;

      bTile[k][x] = __ldg(&B[gRow * N + gCol]);
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; k++) {
      acc += aTile[threadIdx.y][k] * bTile[k][threadIdx.x];
    }
    __syncthreads();
  }

  C[row * N + col] = acc;
}

__global__ void kernel_vector_padded(
  const float* __restrict__ A,
  const float* __restrict__ B,
  float* __restrict__ C,
  const int N, const int K, const int M)
{
  __shared__ VEC_Y_TYPE aTile[TILE_M][TILE_K];
  __shared__ VEC_X_TYPE bTile[TILE_K][TILE_N];

  const int tid = threadIdx.y * blockDim.x + threadIdx.x;
  const int numThreads = blockDim.x * blockDim.y;

  const int blockRow = blockIdx.y * (TILE_M * VEC_Y);
  const int blockCol = blockIdx.x * (TILE_N * VEC_X);
  const int gRow0 = blockRow + threadIdx.y * VEC_Y;
  const int gCol0 = blockCol + threadIdx.x * VEC_X;

  VEC_X_TYPE acc[VEC_Y];
  for (int i = 0; i < VEC_Y; i++) {
    acc[i] = { 0 };
  }

  const int phases = K / TILE_K;
  for (int ph = 0; ph < phases; ph++) {
    const int baseK = ph * TILE_K;

    const int aTileSz = TILE_M * TILE_K;
    for (int idx = tid; idx < aTileSz; idx += numThreads) {
      const int y = idx / TILE_K;
      const int k = idx % TILE_K;
      const int gCol = baseK + k;
      VEC_Y_TYPE aVec = { 0.0f };
#pragma unroll
      for (int i = 0; i < VEC_Y; ++i) {
        const int gRow = blockRow + y * VEC_Y + i;
        reinterpret_cast<float*>(&aVec)[i] = __ldg(&A[gRow * K + gCol]);
      }
      aTile[y][k] = aVec;
    }
    __syncthreads();

    const int bTileSz = TILE_K * TILE_N;
    for (int idx = tid; idx < bTileSz; idx += numThreads) {
      const int k = idx / TILE_N;
      const int x = idx % TILE_N;
      const int gRow = baseK + k;
      VEC_X_TYPE bVec = { 0.0f };
#pragma unroll
      for (int j = 0; j < VEC_X; ++j) {
        const int gCol = blockCol + x * VEC_X + j;
        reinterpret_cast<float*>(&bVec)[j] = __ldg(&B[gRow * N + gCol]);
      }
      bTile[k][x] = bVec;
    }
    __syncthreads();

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      const VEC_Y_TYPE aVec = aTile[threadIdx.y][k];
      const VEC_X_TYPE bVec = bTile[k][threadIdx.x];
#pragma unroll
      for (int i = 0; i < VEC_Y; ++i) {
        const float aVal = reinterpret_cast<const float*>(&aVec)[i];
#pragma unroll
        for (int j = 0; j < VEC_X; ++j) {
          reinterpret_cast<float*>(&acc[i])[j] += aVal * reinterpret_cast<const float*>(&bVec)[j];
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int i = 0; i < VEC_Y; ++i) {
    const int row = gRow0 + i;
#pragma unroll
    for (int j = 0; j < VEC_X; ++j) {
      const int col = gCol0 + j;
      C[row * N + col] = reinterpret_cast<float*>(&acc[i])[j];
    }
  }
}


void gpu_wrapper(optimization opt, bool padded, unsigned int device, float* a, float* b, float* c, int n, int k, int m) {
  cudaError_t cudaStatus;
  int count;
  float gpu_elapsed_time_ms;
  dim3 blockDim;
  cudaEvent_t start, end;
  float time;
  std::chrono::steady_clock::time_point total_start, total_end;
  float* device_a = nullptr, * device_b = nullptr, * device_c = nullptr;
  int vec_x = 1, vec_y = 1;

  cudaStatus = cudaGetDeviceCount(&count);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  device = device >= count ? 0 : device;

  cudaStatus = cudaSetDevice(device);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  {
    cudaDeviceProp prop;
    cudaStatus = cudaGetDeviceProperties(&prop, device);
    if (cudaStatus != cudaSuccess) {
      std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
      goto Error;
    }
    int driverVersion;
    cudaStatus = cudaDriverGetVersion(&driverVersion);
    if (cudaStatus != cudaSuccess) {
      std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
      goto Error;
    }
    std::fprintf(stdout, "Device: %s\tDriver version: %i\n", prop.name, driverVersion);
  }


  cudaStatus = cudaEventCreate(&start);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }
  cudaStatus = cudaEventCreate(&end);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaStatus = cudaMalloc((void**)&device_a, (sizeof(float) * m) * k);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&device_b, (sizeof(float) * k) * n);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }
  cudaStatus = cudaMalloc((void**)&device_c, (sizeof(float) * m) * n);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  total_start = std::chrono::steady_clock::now();

  cudaStatus = cudaMemcpy(device_a, a, (sizeof(float) * m) * k, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaStatus = cudaMemcpy(device_b, b, (sizeof(float) * k) * n, cudaMemcpyHostToDevice);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaStatus = cudaEventRecord(start, 0);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  blockDim = dim3(TILE_N, TILE_M);
  switch (opt) {
  case optimization::tile:
    if (padded) {
      kernel_tile_padded<<<dim3(n / TILE_N, m / TILE_M), blockDim>>>(device_a, device_b, device_c, n, k, m);
    }
    else {
      kernel_tile<<<dim3(n / TILE_N, m / TILE_M), blockDim>>>(device_a, device_b, device_c, n, k, m);
    }
    break;
  case optimization::vector:
    if (padded) {
      kernel_vector_padded<<<dim3(n / (TILE_N * VEC_X), m / (TILE_M * VEC_Y)), blockDim>>>(device_a, device_b, device_c, n, k, m);
    }
    else {
      kernel_vector<<<dim3(n / (TILE_N * VEC_X), m / (TILE_M * VEC_Y)), blockDim>>>(device_a, device_b, device_c, n, k, m);
    }
    break;
  }
  cudaStatus = cudaGetLastError();
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaStatus = cudaEventRecord(end, 0);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaStatus = cudaEventSynchronize(end);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaStatus = cudaEventElapsedTime(&gpu_elapsed_time_ms, start, end);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  cudaStatus = cudaMemcpy(c, device_c, (sizeof(float) * m) * n, cudaMemcpyDeviceToHost);
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

  total_end = std::chrono::steady_clock::now();
  time = static_cast<float>(std::chrono::duration_cast<std::chrono::nanoseconds>(total_end - total_start).count()) / 1e6;

  std::fprintf(stdout, "Time: %g\t%g\n", gpu_elapsed_time_ms, time);
  if (opt == optimization::vector) {
    vec_x = VEC_X;
    vec_y = VEC_Y;
  }
  std::fprintf(stdout, "BLOCK_WORK_SIZE [%i, %i]\nITEM_WORK_SIZE [%i, %i]\n", TILE_M, TILE_N, vec_x, vec_y);

  cudaStatus = cudaDeviceSynchronize();
  if (cudaStatus != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(cudaStatus));
    goto Error;
  }

Error:
  cudaFree(device_a);
  cudaFree(device_b);
  cudaFree(device_c);
}

void gpu_tile_matrix_mul(unsigned int device, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::tile, false, device, a, b, c, n, k, m);
}

void gpu_vector_matrix_mul(unsigned int device, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::vector, false, device, a, b, c, n, k, m);
}

void gpu_tile_matrix_mul_padded(unsigned int device, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::tile, true, device, a, b, c, n, k, m);
}

void gpu_vector_matrix_mul_padded(unsigned int device, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::vector, true, device, a, b, c, n, k, m);
}


