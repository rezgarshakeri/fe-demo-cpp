#include "cuda/device-info.hpp"
#include <cuda_runtime_api.h>
#include <sstream>
#include <stdexcept>
#include "cuda/cuda-check.hpp"

namespace fem::cuda {

/**
  @brief Number of CUDA devices visible to this process (0 if there is no usable driver/GPU)
**/
int device_count() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return 0;
  }
  return count;
}

/**
  @brief Query the limits of a device, plus its currently free global memory

  @param[in] device Device id, 0 <= device < device_count()

  @ref cudaGetDeviceProperties, cudaMemGetInfo
**/
DeviceInfo device_info(int device) {
  if (device < 0 || device >= device_count()) throw std::invalid_argument("device_info: no such CUDA device");
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

  DeviceInfo info;
  info.id = device;
  info.name = prop.name;
  info.cc_major = prop.major;
  info.cc_minor = prop.minor;
  info.sm_count = prop.multiProcessorCount;
  info.warp_size = prop.warpSize;
  info.max_threads_per_block = prop.maxThreadsPerBlock;
  info.regs_per_block = prop.regsPerBlock;
  info.shared_mem_per_block = prop.sharedMemPerBlock;
  info.shared_mem_per_block_optin = prop.sharedMemPerBlockOptin;
  info.shared_mem_per_sm = prop.sharedMemPerMultiprocessor;
  info.total_global_mem = prop.totalGlobalMem;

  // cudaMemGetInfo reports on the *current* device.
  int previous = 0;
  CUDA_CHECK(cudaGetDevice(&previous));
  CUDA_CHECK(cudaSetDevice(device));
  std::size_t free_b = 0, total_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
  CUDA_CHECK(cudaSetDevice(previous));
  info.free_global_mem = free_b;
  return info;
}

std::string to_string(const DeviceInfo& info) {
  constexpr double KiB = 1024.0, MiB = 1024.0 * 1024.0;
  std::ostringstream s;
  s << "device " << info.id << ": " << info.name << " (compute capability " << info.cc_major << "." << info.cc_minor
    << ")\n"
    << "  SMs: " << info.sm_count << ", warp size: " << info.warp_size
    << ", max threads/block: " << info.max_threads_per_block << ", registers/block: " << info.regs_per_block << "\n"
    << "  shared memory: " << info.shared_mem_per_block / KiB << " KiB/block (opt-in "
    << info.shared_mem_per_block_optin / KiB << "), " << info.shared_mem_per_sm / KiB << " KiB/SM\n"
    << "  global memory: " << info.free_global_mem / MiB << " MiB free of " << info.total_global_mem / MiB << " MiB";
  return s.str();
}

} // namespace fem::cuda
