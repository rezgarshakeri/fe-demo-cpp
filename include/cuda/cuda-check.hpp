#pragma once
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <string>

namespace fem::cuda {

/**
  @brief Throw std::runtime_error if a CUDA runtime call failed; use through CUDA_CHECK

  Also calls cudaGetLastError() to clear the error. A non-sticky error such as
  cudaErrorMemoryAllocation is otherwise still reported by the next
  cudaGetLastError(), so a later CUDA_CHECK_LAUNCH() would blame the wrong call.
  (Sticky errors, e.g. an illegal address inside a kernel, corrupt the context
  and can't be cleared: every later call fails.)
**/
inline void check(cudaError_t err, const char* expr, const char* file, int line) {
  if (err == cudaSuccess) return;
  (void)cudaGetLastError();
  throw std::runtime_error(std::string(file) + ":" + std::to_string(line) + ": " + expr + " failed: " +
                           cudaGetErrorName(err) + " (" + cudaGetErrorString(err) + ")");
}

} // namespace fem::cuda

// Wrap every CUDA runtime call: CUDA_CHECK(cudaMalloc(&p, bytes));
#define CUDA_CHECK(expr) ::fem::cuda::check((expr), #expr, __FILE__, __LINE__)

// After a kernel launch: catches launch errors (bad grid/block size, too much
// shared memory). Errors *inside* the kernel only show up at the next
// synchronizing call, e.g. CUDA_CHECK(cudaDeviceSynchronize()).
#define CUDA_CHECK_LAUNCH() CUDA_CHECK(cudaGetLastError())
