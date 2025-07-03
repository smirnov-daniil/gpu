#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "scan.hpp"


#include <cstdio>
#include <vector>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <string>
#include <ranges>
#include <cmath>

#ifndef WORKGROUP
#define WORKGROUP 64
#endif

static cl_int load_source(const char* file, char** source_str, size_t* len) {
  std::FILE* f;

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

cl_int recursive(cl_context context, cl_command_queue queue, cl_kernel sum_block, cl_kernel sum_add,
  cl_mem bufA, std::uint32_t n, std::vector<cl_event>& events) {
  cl_int err = CL_SUCCESS;
  std::uint32_t N = std::max(n, std::uint32_t(WORKGROUP * 2));
  cl_mem bufB;

  bufB = clCreateBuffer(
    context, CL_MEM_READ_WRITE, sizeof(cl_int) * (N / (WORKGROUP * 2)),
    nullptr, &err);
  if (err != CL_SUCCESS) return err;

  std::size_t global[] = { N / 2 };
  std::size_t local[] = { WORKGROUP };

  events.emplace_back();
  err |= clSetKernelArg(sum_block, 0, sizeof(cl_mem), &bufA);
  err |= clSetKernelArg(sum_block, 1, sizeof(cl_mem), &bufB);
  err |= clSetKernelArg(sum_block, 2, sizeof(cl_uint), &n);
  err |= clEnqueueNDRangeKernel(queue, sum_block, 1, nullptr,
    global, local, 0, nullptr,
    &events.back());
  if (err != CL_SUCCESS) {
    clReleaseMemObject(bufB);
    return err;
  }

  if (N > WORKGROUP * 2) {
    err = recursive(
      context, queue, sum_block, sum_add, bufB,
      N / (WORKGROUP * 2), events);
    if (err != CL_SUCCESS) {
      clReleaseMemObject(bufB);
      return err;
    };

    events.emplace_back();
    err |= clSetKernelArg(sum_add, 0, sizeof(cl_mem), &bufA);
    err |= clSetKernelArg(sum_add, 1, sizeof(cl_mem), &bufB);
    err |= clSetKernelArg(sum_add, 2, sizeof(cl_uint), &n);
    err = clEnqueueNDRangeKernel(queue, sum_add, 1, nullptr,
      global, local, 0, nullptr,
      &events.back());
    if (err != CL_SUCCESS) {
      clReleaseMemObject(bufB);
      return err;
    }
  }

  clReleaseMemObject(bufB);
  return CL_SUCCESS;
};

void scan(device_type dtype, std::uint32_t device_index, std::uint32_t* a, std::uint32_t n) {
  cl_int err;
  cl_platform_id* platforms = nullptr;
  cl_device_id* devices = nullptr;
  cl_context context = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel sum_block = nullptr, sum_add = nullptr;
  cl_mem bufA = nullptr;
  cl_uint numPlatforms = 0;
  struct DevInfo { cl_platform_id plat; cl_device_id dev; device_type type; };
  DevInfo choice;
  size_t kernel_size = 0;
  char* kernel_source = nullptr;
  std::vector<cl_event> events;
  events.reserve(66);

#define CHECK_ERR(call, msg) do { err = (call); if (err != CL_SUCCESS) { std::fprintf(stderr, "%s: %d\n", msg, err); goto Error; } } while(0)

  {
    CHECK_ERR(clGetPlatformIDs(0, nullptr, &numPlatforms), "clGetPlatformIDs failed");
    if (numPlatforms == 0) {
      std::fprintf(stderr, "No OpenCL platforms found\n");
      goto Error;
    }
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
      dev_list.emplace_back(di);
    }
    if (dev_list.empty()) {
      std::fprintf(stderr, "No matching devices\n");
      goto Error;
    }
    int idx = device_index % static_cast<int>(dev_list.size());
    choice = dev_list[idx];

    char devName[128] = { 0 }, platName[128] = { 0 };
    CHECK_ERR(clGetDeviceInfo(choice.dev, CL_DEVICE_NAME, sizeof(devName), devName, nullptr), "clGetDeviceInfo NAME failed");
    CHECK_ERR(clGetPlatformInfo(choice.plat, CL_PLATFORM_NAME, sizeof(platName), platName, nullptr), "clGetPlatformInfo NAME failed");
    std::printf("Device: %s\tPlatform: %s\n", devName, platName);
  }
  context = clCreateContext(NULL, 1, &choice.dev, NULL, NULL, NULL);

  if (load_source("kernel.cl", &kernel_source, &kernel_size))
    goto Error;
  {
    const size_t sizes[] = { kernel_size };
    const char* kernel_sources[] = { kernel_source };
    program = clCreateProgramWithSource(context, 1, kernel_sources, sizes, &err);
  }
  CHECK_ERR(err, "clCreateProgramWithSource failed");
  {
    err = clBuildProgram(program, 1, &choice.dev, (std::string(" -DWORKGROUP=") + std::to_string(WORKGROUP)).c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
      size_t logSize;
      clGetProgramBuildInfo(program, choice.dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
      std::string log(logSize, '\0');
      clGetProgramBuildInfo(program, choice.dev, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr);
      std::fprintf(stderr, "Build err:\n%s\n", log.c_str());
      goto Error;
    }
  }

  queue = clCreateCommandQueue(context, choice.dev, CL_QUEUE_PROFILING_ENABLE, &err);
  CHECK_ERR(err, "clCreateCommandQueue failed");

  sum_block = clCreateKernel(program, "sum_block", &err);
  CHECK_ERR(err, "clCreateKernel failed");

  sum_add = clCreateKernel(program, "sum_add", &err);
  CHECK_ERR(err, "clCreateKernel failed");

  bufA = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uint) * n, nullptr, &err);
  CHECK_ERR(err, "ClCreateBuffer failed");

  events.emplace_back();
  CHECK_ERR(clEnqueueWriteBuffer(queue, bufA, CL_FALSE, 0, sizeof(cl_uint) * n, a, 0, nullptr, &events.back()), "clEnqueueWriteBuffer failed");

  CHECK_ERR(recursive(context, queue, sum_block, sum_add, bufA, n, events), "Failed MAMA PAPA YA");

  events.emplace_back();
  CHECK_ERR(clEnqueueReadBuffer(queue, bufA, CL_TRUE, 0, sizeof(cl_uint) * n, a, 0, nullptr, &events.back()), "clEnqueueReadBuffer failed");

  CHECK_ERR(clFinish(queue), "clFinish failed");

  {
    auto measure = [&](cl_event e) { cl_ulong s, t; clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START, sizeof(s), &s, nullptr); clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END, sizeof(t), &t, nullptr); return (t - s) * 1e-6; };
    std::vector<double> timez;
    std::transform(events.begin(), events.end(), std::back_inserter(timez), measure);
    std::fprintf(stdout, "Time: %g\t%g\n",
      std::accumulate(timez.begin() + 1, timez.end() - 1, 0.0),
      std::accumulate(timez.begin(), timez.end(), 0.0));
  }

Error:
  for (auto event : events)
    if (event) clReleaseEvent(event);
  if (bufA) clReleaseMemObject(bufA);
  if (program) clReleaseProgram(program);
  if (queue) clReleaseCommandQueue(queue);
  if (context) clReleaseContext(context);
  if (sum_block) clReleaseKernel(sum_block);
  if (sum_add) clReleaseKernel(sum_add);
  if (kernel_source)
    delete[] kernel_source;
  if (platforms)
    delete[] platforms;
#undef CHECK_ERR
}
