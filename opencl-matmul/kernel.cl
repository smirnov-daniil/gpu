
#ifdef VEC_X
#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a##b

#define VEC_X_TYPE CAT(float, VEC_X)
#define VEC_Y_TYPE CAT(float, VEC_Y)

#define VLOAD_X CAT(vload, VEC_X)
#define VSTORE_X CAT(vstore, VEC_X)
#endif

__kernel void kernel_tile(
  __global const float* restrict A,
  __global const float* restrict B,
  __global       float* restrict C,
  const int      N,
  const int      K,
  const int      M)
{
  __local float aTile[TILE_M][TILE_K];
  __local float bTile[TILE_K][TILE_N];

  const int localRow = get_local_id(1);
  const int localCol = get_local_id(0);
  const int globalRow = get_global_id(1);
  const int globalCol = get_global_id(0);

  const int localSizeX = get_local_size(0);
  const int localSizeY = get_local_size(1);
  const int numThreads = localSizeX * localSizeY;
  const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);

  float acc = 0.0f;
  const int phases = (K + TILE_K - 1) / TILE_K;

  for (int ph = 0; ph < phases; ++ph) {
    const int baseK = ph * TILE_K;

    for (int idx = tid; idx < TILE_M * TILE_K; idx += numThreads) {
      int i = idx / TILE_K;
      int k = idx % TILE_K;
      int r = (get_group_id(1) * TILE_M) + i;
      int c = baseK + k;

      if (r < M && c < K)
        aTile[i][k] = A[r * K + c];
      else
        aTile[i][k] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int idx = tid; idx < TILE_K * TILE_N; idx += numThreads) {
      int k = idx / TILE_N;
      int j = idx % TILE_N;
      int r = baseK + k;
      int c = (get_group_id(0) * TILE_N) + j;

      if (r < K && c < N)
        bTile[k][j] = B[r * N + c];
      else
        bTile[k][j] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      acc += aTile[localRow][k] * bTile[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (globalRow < M && globalCol < N) {
    C[globalRow * N + globalCol] = acc;
  }
}

#ifdef VEC_X
__kernel void kernel_vector(
  __global const float* restrict A,
  __global const float* restrict B,
  __global       float* restrict C,
  const int      N,
  const int      K,
  const int      M)
{
  __local VEC_Y_TYPE aTile[TILE_M][TILE_K];
  __local VEC_X_TYPE bTile[TILE_K][TILE_N];

  const int localX = get_local_id(0);
  const int localY = get_local_id(1);
  const int globalX = get_global_id(0);
  const int globalY = get_global_id(1);

  const int localSizeX = get_local_size(0);
  const int localSizeY = get_local_size(1);
  const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int nThreads = localSizeX * localSizeY;

  const int groupX = get_group_id(0);
  const int groupY = get_group_id(1);

  const int blockRow = groupY * (TILE_M * VEC_Y);
  const int blockCol = groupX * (TILE_N * VEC_X);

  const int gRow0 = get_global_id(1) * VEC_Y;
  const int gCol0 = get_global_id(0) * VEC_X;


  VEC_X_TYPE acc[VEC_Y];
#pragma unroll
  for (int i = 0; i < VEC_Y; ++i)
    acc[i] = (VEC_X_TYPE)(0.0f);

  const int phases = (K + TILE_K - 1) / TILE_K;

  for (int ph = 0; ph < phases; ++ph) {
    const int baseK = ph * TILE_K;

    const int aTileSz = TILE_M * TILE_K;
    for (int idx = tid; idx < aTileSz; idx += nThreads) {
      int y = idx / TILE_K;
      int k = idx % TILE_K;
      int col = baseK + k;

      VEC_Y_TYPE vec = (VEC_Y_TYPE)(0.0f);
      if (col < K) {
#pragma unroll
        for (int i = 0; i < VEC_Y; ++i) {
          int row = blockRow + y * VEC_Y + i;
          if (row < M)
            ((float*)&vec)[i] = A[row * K + col];
        }
      }
      aTile[y][k] = vec;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int bTileSz = TILE_K * TILE_N;
    for (int idx = tid; idx < bTileSz; idx += nThreads) {
      int k = idx / TILE_N;
      int x = idx % TILE_N;
      int row = baseK + k;

      VEC_X_TYPE vec = (VEC_X_TYPE)(0.0f);
      if (row < K) {
#pragma unroll
        for (int j = 0; j < VEC_X; ++j) {
          int col = blockCol + x * VEC_X + j;
          if (col < N)
            ((float*)&vec)[j] = B[row * N + col];
        }
      }
      bTile[k][x] = vec;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      VEC_Y_TYPE aVec = aTile[localY][k];
      VEC_X_TYPE bVec = bTile[k][localX];
#pragma unroll
      for (int i = 0; i < VEC_Y; i++) {
        float aVal = ((float*)&aVec)[i];
        acc[i] += aVal * bVec;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (int i = 0; i < VEC_Y; ++i) {
    int row = gRow0 + i;
    if (row < M) {
#pragma unroll
      for (int j = 0; j < VEC_X; ++j) {
        int col = gCol0 + j;
        if (col < N) {
          C[row * N + col] = ((float*)&(acc[i]))[j];
        }
      }
    }
  }
}
#endif


__kernel void kernel_tile_padded(
  __global const float* restrict A,
  __global const float* restrict B,
  __global       float* restrict C,
  const int      N,
  const int      K,
  const int      M)
{
  __local float aTile[TILE_M][TILE_K];
  __local float bTile[TILE_K][TILE_N];

  const int localRow = get_local_id(1);
  const int localCol = get_local_id(0);
  const int globalRow = get_global_id(1);
  const int globalCol = get_global_id(0);

  const int localSizeX = get_local_size(0);
  const int localSizeY = get_local_size(1);
  const int numThreads = localSizeX * localSizeY;
  const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);

  float acc = 0.0f;
  const int phases = K / TILE_K;

  for (int ph = 0; ph < phases; ++ph) {
    const int baseK = ph * TILE_K;

    for (int idx = tid; idx < TILE_M * TILE_K; idx += numThreads) {
      int i = idx / TILE_K;
      int k = idx % TILE_K;
      int r = (get_group_id(1) * TILE_M) + i;
      int c = baseK + k;

      aTile[i][k] = A[r * K + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int idx = tid; idx < TILE_K * TILE_N; idx += numThreads) {
      int k = idx / TILE_N;
      int j = idx % TILE_N;
      int r = baseK + k;
      int c = (get_group_id(0) * TILE_N) + j;

      bTile[k][j] = B[r * N + c];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      acc += aTile[localRow][k] * bTile[k][localCol];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  C[globalRow * N + globalCol] = acc;
}

#ifdef VEC_X

__kernel void kernel_vector_padded(
  __global const float* restrict A,
  __global const float* restrict B,
  __global       float* restrict C,
  const int      N,
  const int      K,
  const int      M)
{
  __local VEC_Y_TYPE aTile[TILE_M][TILE_K];
  __local VEC_X_TYPE bTile[TILE_K][TILE_N];

  const int localX = get_local_id(0);
  const int localY = get_local_id(1);
  const int globalX = get_global_id(0);
  const int globalY = get_global_id(1);

  const int localSizeX = get_local_size(0);
  const int localSizeY = get_local_size(1);
  const int tid = get_local_id(1) * get_local_size(0) + get_local_id(0);
  const int nThreads = localSizeX * localSizeY;

  const int groupX = get_group_id(0);
  const int groupY = get_group_id(1);

  const int blockRow = groupY * (TILE_M * VEC_Y);
  const int blockCol = groupX * (TILE_N * VEC_X);

  const int gRow0 = get_global_id(1) * VEC_Y;
  const int gCol0 = get_global_id(0) * VEC_X;


  VEC_X_TYPE acc[VEC_Y];
#pragma unroll
  for (int i = 0; i < VEC_Y; ++i)
    acc[i] = (VEC_X_TYPE)(0.0f);

  const int phases = (K + TILE_K - 1) / TILE_K;

  for (int ph = 0; ph < phases; ++ph) {
    const int baseK = ph * TILE_K;

    const int aTileSz = TILE_M * TILE_K;
    for (int idx = tid; idx < aTileSz; idx += nThreads) {
      int y = idx / TILE_K;
      int k = idx % TILE_K;
      int col = baseK + k;

      VEC_Y_TYPE vec = (VEC_Y_TYPE)(0.0f);
#pragma unroll
      for (int i = 0; i < VEC_Y; ++i) {
        int row = blockRow + y * VEC_Y + i;
        ((float*)&vec)[i] = A[row * K + col];
      }
      aTile[y][k] = vec;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const int bTileSz = TILE_K * TILE_N;
    for (int idx = tid; idx < bTileSz; idx += nThreads) {
      int k = idx / TILE_N;
      int x = idx % TILE_N;

      bTile[k][x] = VLOAD_X(0, B + (baseK + k) * N + blockCol + x * VEC_X);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll
    for (int k = 0; k < TILE_K; ++k) {
      VEC_Y_TYPE aVec = aTile[localY][k];
      VEC_X_TYPE bVec = bTile[k][localX];
#pragma unroll
      for (int i = 0; i < VEC_Y; i++) {
        float aVal = ((float*)&aVec)[i];
        acc[i] += aVal * bVec;
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

#pragma unroll
  for (int i = 0; i < VEC_Y; ++i) {
    int row = gRow0 + i;
    VSTORE_X(acc[i], 0, C + row * N + gCol0);
  }
}
#endif
