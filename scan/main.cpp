#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <string>
#include <new>
#include <numeric>

#include "scan.hpp"

static std::uint32_t round(std::uint32_t n) {
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;
  return n;
}

static int read_array(std::FILE* fin, std::uint32_t** array, std::uint32_t* n) {
  if (std::fread((void*)n, sizeof(std::uint32_t), 1, fin) != 1) {
    return -1;
  }

  try {
    *array = static_cast<std::uint32_t*>(new std::uint32_t[sizeof(std::uint32_t) * round(*n)]);
  }
  catch (std::bad_alloc e) {
    return -1;
  }
  if (std::fread((void*)*array, sizeof(std::uint32_t), *n, fin) != *n) {
    return -1;
  }
  return 0;
}

static int write_array(std::FILE* fout, std::uint32_t* array, std::uint32_t n) {
  std::fwrite((void*)&n, sizeof(std::uint32_t), 1, fout);
  std::fwrite((void*)array, sizeof(std::uint32_t), n, fout);
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
      "           [ --device-index index ]");
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

  std::uint32_t
    device = std::stoi(device_index.data());
  std::uint32_t* array;
  std::uint32_t n;

  if (read_array(fin, &array, &n)) {
    std::fclose(fin);
    std::fclose(fout);
    std::fprintf(stderr, "Failed to read matrices.");
    return 1;
  }
  std::fclose(fin);

  scan(dtype, device, array, round(n));

  write_array(fout, array, n);

  std::fclose(fout);
  delete[] array;
  return 0;
}
