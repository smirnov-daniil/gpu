#ifndef __matrix_mul_hpp_
#define __matrix_mul_hpp_

void cpu_matrix_mul(float* a, float* b, float* v, int n, int k, int m);

void gpu_tile_matrix_mul(unsigned int device, float* a, float* b, float* c, int n, int k, int m);

void gpu_vector_matrix_mul(unsigned int device, float* a, float* b, float* c, int n, int k, int m);

void gpu_tile_matrix_mul_padded(unsigned int device, float* a, float* b, float* c, int n, int k, int m);

void gpu_vector_matrix_mul_padded(unsigned int device, float* a, float* b, float* c, int n, int k, int m);

#endif // !__matrix_mul_hpp_
