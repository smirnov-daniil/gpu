#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include <cstdio>
#include <cstring>
#include <string_view>
#include <string>
#include <new>

//#include "hyperparameters.hpp"
#include "matrix_mul.hpp"

static int read_matrices(std::FILE* fin, float** host_a, float** host_b, float** host_c, unsigned int* n, unsigned int* k, unsigned int* m) {
  unsigned int nkm[3];

  if (std::fread((void*)nkm, sizeof(unsigned int), 3, fin) != 3) {
    return -1;
  }
  *n = nkm[0];
  *k = nkm[1];
  *m = nkm[2];

  std::size_t size_a = std::size_t(*m) * *k;
  std::size_t size_b = std::size_t(*k) * *n;
  std::size_t size_c = std::size_t(*m) * *n;

  try {
    *host_a = static_cast<float*>(new float[sizeof(float) * (size_a + size_b + size_c)]);
  }
  catch (std::bad_alloc e) {
    return -1;
  }
  if (std::fread((void*)*host_a, sizeof(float), size_a + size_b, fin) != size_a + size_b) {
    delete[] host_a;
    return -1;
  }
  *host_b = *host_a + size_a;
  *host_c = *host_b + size_b;
  return 0;
}

static int read_matrices_padded(
  std::FILE* fin,
  float** host_a_pad, float** host_b_pad, float** host_c_pad,
  unsigned int* n, unsigned int* k, unsigned int* m,
  unsigned int* n_pad, unsigned int* k_pad, unsigned int* m_pad, const hypers* hyp)
{
  unsigned int nkm[3];
  if (std::fread(nkm, sizeof(unsigned int), 3, fin) != 3) {
    return -1;
  }
  *n = nkm[0];
  *k = nkm[1];
  *m = nkm[2];

  *m_pad = ((*m + (hyp->tile_m * hyp->vec_y) - 1) / (hyp->tile_m * hyp->vec_y)) * (hyp->tile_m * hyp->vec_y);
  *k_pad = ((*k + hyp->tile_k - 1) / hyp->tile_k) * hyp->tile_k;
  *n_pad = ((*n + (hyp->tile_n * hyp->vec_x) - 1) / (hyp->tile_n * hyp->vec_x)) * (hyp->tile_n * hyp->vec_x);

  std::size_t size_a = std::size_t(*m_pad) * *k_pad;
  std::size_t size_b = std::size_t(*k_pad) * *n_pad;
  std::size_t size_c = std::size_t(*m_pad) * *n_pad;

  float* buf = nullptr;
  try {
    buf = static_cast<float*>(new float[sizeof(float) * (size_a + size_b + size_c)]);
  }
  catch (std::bad_alloc e) {
    return -1;
  }

  std::memset((void*)buf, 0.0f, (size_a + size_b + size_c) * sizeof(float));

  float* A_pad = buf;
  for (unsigned i = 0; i < *m; i++) {
    if (std::fread((void*)(A_pad + size_t(i) * *k_pad), sizeof(float), *k, fin) != *k) {
      delete[] buf;
      return -1;
    }
  }

  float* B_pad = A_pad + size_a;
  for (unsigned i = 0; i < *k; i++) {
    if (std::fread((void*)(B_pad + size_t(i) * *n_pad), sizeof(float), *n, fin) != *n) {
      delete[] buf;
      return -1;
    }
  }

  *host_a_pad = A_pad;
  *host_b_pad = B_pad;
  *host_c_pad = B_pad + size_b;

  return 0;
}

static int write_matrix(std::FILE* fout, float* host_c, unsigned int n, unsigned int m) {
  std::fwrite((void*)&n, sizeof(unsigned int), 1, fout);
  std::fwrite((void*)&m, sizeof(unsigned int), 1, fout);
  std::fwrite((void*)host_c, sizeof(float), n * m, fout);
  return 0;
}

static int write_matrix_padded(std::FILE* fout, float* host_c, unsigned int n, unsigned int m, unsigned int n_pad) {
  std::fwrite((void*)&n, sizeof(unsigned int), 1, fout);
  std::fwrite((void*)&m, sizeof(unsigned int), 1, fout);
  for (int i = 0; i < m; i++)
    std::fwrite((void*)(host_c + i * n_pad), sizeof(float), n, fout);
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc == 1) {
    std::fprintf(stderr, "No arguments provided.");
    return 1;
  }
  if (argc == 2 && std::strcmp(argv[1], "--help") == 0) {
    std::fprintf(stdout,
      "executable < --input file_name >\n"
      "           < --output file_name >\n"
      "           [ --device-type { dgpu | igpu | gpu | cpu | all } ]"
      "           [ --device-index index ]\n"
      "           [ --realization index ]");
    return 0;
  }
  if (argc % 2 != 1) {
    std::fprintf(stderr, "Invalid arguments provided.");
    return 1;
  }

  std::string_view input, output, device_index = "-1", dev_type = "all", realization = "0";
  for (int i = 1; i < argc; i += 2) {
    if (std::strcmp(argv[i], "--input") == 0) {
      input = argv[i + 1];
      continue;
    }
    if (std::strcmp(argv[i], "--output") == 0) {
      output = argv[i + 1];
      continue;
    }
    if (std::strcmp(argv[i], "--device-type") == 0) {
      dev_type = argv[i + 1];
      continue;
    }
    if (std::strcmp(argv[i], "--device-index") == 0) {
      device_index = argv[i + 1];
      continue;
    }
    if (std::strcmp(argv[i], "--realization") == 0) {
      realization = argv[i + 1];
      continue;
    }
    break;
  }

  std::FILE* fin, * fout;

  if (!(fin = std::fopen(input.data(), "rb"))) {
    std::perror("Invalid input file");
    return 1;
  }

  if (!(fout = std::fopen(output.data(), "wb"))) {
    std::fclose(fin);
    std::perror("Invalid output file");
    return 1;
  }

  device_type dtype = device_type::all;
  switch (dev_type.front()) {
  case 'c':
    dtype = device_type::cpu;
    break;
  case 'd':
    dtype = device_type::dgpu;
    break;
  case 'i':
    dtype = device_type::igpu;
    break;
  case 'a':
    dtype = device_type::all;
    break;
  }

  unsigned int
    device = std::stoi(device_index.data()),
    program = std::stoi(realization.data());
  float* host_a, * host_b, * host_c;
  unsigned int n, k, m, n_pad, k_pad, m_pad;

  hypers hyp;
  get_hyp(&hyp, (program == 1 || program == 3) ? optimization::tile : optimization::vector, (program == 1 || program == 2));

  if ((program == 1 || program == 2)
    ? read_matrices_padded(fin, &host_a, &host_b, &host_c, &n, &k, &m, &n_pad, &k_pad, &m_pad, &hyp)
    : read_matrices(fin, &host_a, &host_b, &host_c, &n, &k, &m)) {
    std::fclose(fin);
    std::fclose(fout);
    std::fprintf(stderr, "Failed to read matrices.");
    return 1;
  }
  std::fclose(fin);

  switch (program) {
  case 0:
    cpu_matrix_mul(host_a, host_b, host_c, n, k, m);
    break;
  case 1:
    gpu_tile_matrix_mul_padded(device, dtype, host_a, host_b, host_c, n_pad, k_pad, m_pad);
    break;
  case 2:
    gpu_vector_matrix_mul_padded(device, dtype, host_a, host_b, host_c, n_pad, k_pad, m_pad);
    break;
  case 3:
    gpu_tile_matrix_mul(device, dtype, host_a, host_b, host_c, n, k, m);
    break;
  case 4:
    gpu_vector_matrix_mul(device, dtype, host_a, host_b, host_c, n, k, m);
    break;
  }

  if ((program == 1 || program == 2)) {
    write_matrix_padded(fout, host_c, n, m, n_pad);
  }
  else {
    write_matrix(fout, host_c, n, m);
  }
  std::fclose(fout);
  delete[] host_a;
  return 0;
}
