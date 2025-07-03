#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <cstdio>
#include <vector>
#include <chrono>
#include <algorithm>
#include <string>
#include <ranges>
#include <cmath>

#include "matrix_mul.hpp"
//#include "hyperparameters.hpp"

static cl_int load_source(const char* file, char** source_str, size_t* len) {
  FILE* f;

  if (!(f = std::fopen(file, "rb"))) {
    std::fprintf(stderr, "Failed to load kernel\n");
    return 1;
  }

  std::fseek(f, 0, SEEK_END);
  *len = std::ftell(f);
  std::rewind(f);

  if (!(*source_str = new char[*len + 1])) {
    std::fclose(f);
    std::fprintf(stderr, "Failed to allocate memory for kernel source\n");
    return 1;
  }
  (*source_str)[*len] = '\0';
  if (std::fread(*source_str, sizeof(char), *len, f) != *len) {
    std::fclose(f);
    delete[] * source_str;
    std::fprintf(stderr, "Failed to read kernel source\n");
    return 1;
  }
  std::fclose(f);
  return 0;
}

void get_hyp(hypers* hyp, optimization opt, bool is_padded, size_t* maxItem, size_t maxWG) {
  switch (opt) {
  case optimization::tile:
    if (is_padded) {
      *hyp = {
        .tile_n = 32,
        .tile_m = 16,
        .tile_k = 64,
        .vec_x = 1,
        .vec_y = 1
      };
    }
    else {
      *hyp = {
        .tile_n = 16,
        .tile_m = 16,
        .tile_k = 48,
        .vec_x = 1,
        .vec_y = 1
      };
    }
    break;
  case optimization::vector:
    if (is_padded) {
      *hyp = {
        .tile_n = 16,
        .tile_m = 8,
        .tile_k = 32,
        .vec_x = 4,
        .vec_y = 8
      };
    }
    else {
      *hyp = {
        .tile_n = 32,
        .tile_m = 8,
        .tile_k = 16,
        .vec_x = 4,
        .vec_y = 16
      };
    }
    break;
  }

  if (maxItem) {
    hyp->tile_n = std::min(maxItem[0], (size_t)hyp->tile_n);
    hyp->tile_m = std::min(maxItem[1], (size_t)hyp->tile_m);
    if (hyp->tile_n * hyp->tile_m > maxWG) {
      hyp->tile_m = maxWG / hyp->tile_n;
    }
  }
}

static void get_global_size(size_t* global, int n, int k, int m, const hypers* hyp) {
  global[0] = ((n + (hyp->tile_n * hyp->vec_x) - 1) / (hyp->tile_n * hyp->vec_x)) * hyp->tile_n;
  global[1] = ((m + (hyp->tile_m * hyp->vec_y) - 1) / (hyp->tile_m * hyp->vec_y)) * hyp->tile_m;
}

static void get_local_size(size_t* local, const hypers* hyp) {
  local[0] = hyp->tile_n;
  local[1] = hyp->tile_m;
}

static std::string get_options(const hypers* hyp) {
  return std::string(
    " -DTILE_N=") + std::to_string(hyp->tile_n) +
    " -DTILE_M=" + std::to_string(hyp->tile_m) +
    " -DTILE_K=" + std::to_string(hyp->tile_k) +
    ((hyp->vec_x * hyp->vec_y != 1)
      ? std::string(" -DVEC_X=") + std::to_string(hyp->vec_x) + " -DVEC_Y=" + std::to_string(hyp->vec_y)
      : "");
}

void gpu_wrapper(
  optimization opt,
  bool padded,
  device_type dtype,
  int device_index,
  float* a, float* b, float* c,
  int N, int K, int M)
{
  cl_int err;
  cl_platform_id* platforms = nullptr;
  cl_device_id* devices = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel kernel = nullptr;
  cl_mem bufA = nullptr, bufB = nullptr, bufC = nullptr;
  cl_event evA = nullptr, evB = nullptr, evK = nullptr, evC = nullptr;
  cl_uint numPlatforms = 0;
  struct DevInfo { cl_platform_id plat; cl_device_id dev; device_type type; };
  DevInfo choice;
  size_t sizeA = sizeof(float) * M * K;
  size_t sizeB = sizeof(float) * K * N;
  size_t sizeC = sizeof(float) * M * N;
  size_t kernel_size = 0;
  char* kernel_source = nullptr;
  hypers hyp;
  size_t local[2];
  size_t global[2];
  size_t maxWG, maxItem[2];
  std::string options;

#define CHECK_ERR(call, msg) do { err = (call); if (err != CL_SUCCESS) { std::fprintf(stderr, "%s: %d\n", msg, err); goto Error; } } while(0)

  {
    CHECK_ERR(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs failed");
    if (numPlatforms == 0) { std::fprintf(stderr, "No OpenCL platforms found\n"); goto Error; }
    platforms = new cl_platform_id[numPlatforms];
    CHECK_ERR(clGetPlatformIDs(numPlatforms, platforms, nullptr), "clGetPlatformIDs enumeration failed");

    std::vector<DevInfo> all_devs;
    for (cl_uint i = 0; i < numPlatforms; ++i) {
      cl_uint count = 0;
      CHECK_ERR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &count), "clGetDeviceIDs failed");
      if (count == 0) continue;
      devices = new cl_device_id[count];
      CHECK_ERR(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, count, devices, nullptr), "clGetDeviceIDs enumeration failed");
      for (cl_uint j = 0; j < count; ++j) {
        cl_device_type t;
        CHECK_ERR(clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(t), &t, nullptr), "clGetDeviceInfo TYPE failed");
        cl_bool unified = CL_FALSE;
        CHECK_ERR(clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(unified), &unified, nullptr), "clGetDeviceInfo UNIFIED failed");
        device_type dt;
        if (t & CL_DEVICE_TYPE_CPU) dt = device_type::cpu;
        else if (t & CL_DEVICE_TYPE_GPU) dt = unified ? device_type::igpu : device_type::dgpu;
        else dt = device_type::all;
        all_devs.emplace_back(platforms[i], devices[j], dt);
      }
      delete[] devices;
      devices = nullptr;
    }

    std::vector<DevInfo> dev_list;
    for (auto& di : all_devs | std::views::filter([&](auto& di) { return matches_type(dtype, di.type); })) {
      dev_list.push_back(di);
    }
    if (dev_list.empty()) { std::fprintf(stderr, "No matching devices\n"); goto Error; }
    int idx = device_index % static_cast<int>(dev_list.size());
    choice = dev_list[idx];

    char devName[128] = { 0 }, platName[128] = { 0 };
    CHECK_ERR(clGetDeviceInfo(choice.dev, CL_DEVICE_NAME, sizeof(devName), devName, nullptr), "clGetDeviceInfo NAME failed");
    CHECK_ERR(clGetPlatformInfo(choice.plat, CL_PLATFORM_NAME, sizeof(platName), platName, nullptr), "clGetPlatformInfo NAME failed");
    std::printf("Device: %s\tPlatform: %s\n", devName, platName);
  }
  clGetDeviceInfo(choice.dev, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(maxItem), maxItem, nullptr);
  clGetDeviceInfo(choice.dev, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWG), &maxWG, nullptr);


  get_hyp(&hyp, opt, padded, maxItem, maxWG);
  get_local_size(local, &hyp);
  get_global_size(global, N, K, M, &hyp);
  options = get_options(&hyp);

  context = clCreateContext(nullptr, 1, &choice.dev, nullptr, nullptr, &err);
  CHECK_ERR(err, "clCreateContext returned error");
  queue = clCreateCommandQueue(context, choice.dev, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_ERR(err, "clCreateCommandQueue failed");

  if (load_source("kernel.cl", &kernel_source, &kernel_size))
    goto Error;
  {
    const size_t sizes[] = { kernel_size };
    const char* kernel_sources[] = { kernel_source };
    program = clCreateProgramWithSource(context, 1, kernel_sources, sizes, &err);
  }
  CHECK_ERR(err, "clCreateProgramWithSource failed");
  {
    //#define STRINGIFY_IMPL(x)   #x
    //#define STRINGIFY(x)        STRINGIFY_IMPL(x)
    //#define OPTION_D(def)       "-D" #def "=" STRINGIFY(def) " "
    //    const char* options =
    //
    //      OPTION_D(TILE_N)
    //      OPTION_D(TILE_M)
    //      OPTION_D(TILE_K)
    //      OPTION_D(VEC_X)
    //      OPTION_D(VEC_Y)
    //      ;
    //#undef OPTION_D
    //#undef STRINGIFY
    //#undef STRINGIFY_IMPL
    err = clBuildProgram(program, 1, &choice.dev, options.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize;
      clGetProgramBuildInfo(program, choice.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      std::string log(logSize, '\0');
      clGetProgramBuildInfo(program, choice.dev, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
      std::fprintf(stderr, "Build error:\n%s\n", log.c_str());
      goto Error;
    }
  }

  switch (opt) {
  case optimization::tile:
    kernel = clCreateKernel(program, padded ? "kernel_tile_padded" : "kernel_tile", &err);
    break;
  case optimization::vector:
    kernel = clCreateKernel(program, padded ? "kernel_vector_padded" : "kernel_vector", &err);
    break;
  }
  CHECK_ERR(err, "clCreateKernel failed");

  bufA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeA, nullptr, &err);
  CHECK_ERR(err, "clCreateBuffer A failed");
  bufB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeB, nullptr, &err);
  CHECK_ERR(err, "clCreateBuffer B failed");
  bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeC, nullptr, &err);
  CHECK_ERR(err, "clCreateBuffer C failed");

  CHECK_ERR(clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeA, a, 0, nullptr, &evA), "clEnqueueWriteBuffer A failed");
  CHECK_ERR(clEnqueueWriteBuffer(queue, bufB, CL_FALSE, 0, sizeB, b, 0, nullptr, &evB), "clEnqueueWriteBuffer B failed");

  CHECK_ERR(clSetKernelArg(kernel, 0, sizeof(bufA), &bufA), "clSetKernelArg A failed");
  CHECK_ERR(clSetKernelArg(kernel, 1, sizeof(bufB), &bufB), "clSetKernelArg B failed");
  CHECK_ERR(clSetKernelArg(kernel, 2, sizeof(bufC), &bufC), "clSetKernelArg C failed");
  CHECK_ERR(clSetKernelArg(kernel, 3, sizeof(int), &N), "clSetKernelArg N failed");
  CHECK_ERR(clSetKernelArg(kernel, 4, sizeof(int), &K), "clSetKernelArg K failed");
  CHECK_ERR(clSetKernelArg(kernel, 5, sizeof(int), &M), "clSetKernelArg M failed");

  {
    cl_event evs[] = { evA, evB };
    CHECK_ERR(clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, global, local, 2, evs, &evK), "clEnqueueNDRangeKernel failed");
  }

  CHECK_ERR(clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, sizeC, c, 1, &evK, &evC), "clEnqueueReadBuffer C failed");

  CHECK_ERR(clFinish(queue), "clFinish failed");

  {
    auto measure = [&](cl_event e) { cl_ulong s, t; clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr); clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(t), &t, nullptr); return (t - s) * 1e-6; };
    std::printf("Time: %g\t%g\n", measure(evK), measure(evA) + measure(evB) + measure(evK) + measure(evC));
  }

Error:
  if (evA) clReleaseEvent(evA);
  if (evB) clReleaseEvent(evB);
  if (evK) clReleaseEvent(evK);
  if (evC) clReleaseEvent(evC);
  if (bufA) clReleaseMemObject(bufA);
  if (bufB) clReleaseMemObject(bufB);
  if (bufC) clReleaseMemObject(bufC);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  if (kernel_source)
    delete[] kernel_source;
  if (platforms)
    delete[] platforms;

#undef CHECK_ERR
}

void gpu_tile_matrix_mul(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::tile, false, dtype, device, a, b, c, n, k, m);
}

void gpu_vector_matrix_mul(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::vector, false, dtype, device, a, b, c, n, k, m);
}

void gpu_tile_matrix_mul_padded(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::tile, true, dtype, device, a, b, c, n, k, m);
}

void gpu_vector_matrix_mul_padded(unsigned int device, device_type dtype, float* a, float* b, float* c, int n, int k, int m) {
  gpu_wrapper(optimization::vector, true, dtype, device, a, b, c, n, k, m);
}
