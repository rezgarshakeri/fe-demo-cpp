#pragma once
#include <cstddef>
#include <string>

namespace fem::cuda {

// The device limits that decide launch configurations and memory budgets
struct DeviceInfo {
  int id = 0;
  std::string name;
  int cc_major = 0, cc_minor = 0;  // compute capability, e.g. 12.0 for Blackwell RTX
  int sm_count = 0;                // streaming multiprocessors
  int warp_size = 0;
  int max_threads_per_block = 0;
  int regs_per_block = 0;                     // 32-bit registers
  std::size_t shared_mem_per_block = 0;       // default limit per block (48 KB)
  std::size_t shared_mem_per_block_optin = 0; // max with cudaFuncSetAttribute(MaxDynamicSharedMemorySize)
  std::size_t shared_mem_per_sm = 0;
  std::size_t total_global_mem = 0;  // bytes
  std::size_t free_global_mem = 0;   // bytes, at the time of the query (cudaMemGetInfo)
};

int device_count();
DeviceInfo device_info(int device = 0);
std::string to_string(const DeviceInfo& info);

} // namespace fem::cuda
