#ifndef __hyperparameters_hpp_
#define __hyperparameters_hpp_

#include <vector_types.h>

struct __align__(32) float8
{
  float x, y, z, w, a, b, c, d;
};

#ifndef VEC_X
#define VEC_X 4
#endif // !VEC_X

#ifndef VEC_Y
#define VEC_Y 8
#endif // !VEC_Y

#define ALIGMENT (VEC_Y > VEC_X ? VEC_Y : VEC_X) * 4

#define CAT(a, b) CAT_IMPL(a, b)
#define CAT_IMPL(a, b) a##b

#define VEC_X_TYPE CAT(float, VEC_X)
#define VEC_Y_TYPE CAT(float, VEC_Y)

#ifndef TILE_M
#define TILE_M 16
#endif // !TILE_M
#ifndef TILE_N
#define TILE_N 16
#endif // !TILE_N
#ifndef TILE_K
#define TILE_K 32
#endif // !TILE_K

#endif // !__hyperparameters_hpp_
