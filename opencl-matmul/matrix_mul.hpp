#ifndef __matrix_mul_hpp_
#define __matrix_mul_hpp_

enum class device_type : int {
  cpu = 0x1,
  dgpu = 0x2,
  igpu = 0x4,
  gpu = dgpu | igpu,
  all = cpu | dgpu | igpu
};

inline bool matches_type(device_type filter, device_type dt) {
  return (static_cast<int>(filter) & static_cast<int>(dt)) != 0;
}

struct hypers {
  int tile_n, tile_m, tile_k;
  int vec_x, vec_y;
};

enum class optimization { tile, vector };

void get_hyp(hypers* hyp, optimization opt, bool is_padded, size_t* maxItem = nullptr, size_t maxWG = -1);

void cpu_matrix_mul(float* a, float* b, float* v, int n, int k, int m);

void gpu_tile_matrix_mul(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m);

void gpu_vector_matrix_mul(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m);

void gpu_tile_matrix_mul_padded(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m);

void gpu_vector_matrix_mul_padded(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m);

#endif // !__matrix_mul_hpp_
