#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "cuda/cuda-check.hpp"
#include "cuda/device-array.hpp"
#include "cuda/device-info.hpp"

using fem::cuda::DeviceArray;

namespace {

// Toolchain smoke test only: x[i] = a * x[i] + i.
// Grid-stride loop: correct for any grid size, including grids smaller than n
// (the pattern libCEED's restriction kernels use).
__global__ void scale_add_index(double* x, double a, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    x[i] = a * x[i] + i;
  }
}

}  // namespace

TEST_CASE("a CUDA device is visible, with sane limits", "[cuda][device-info]") {
  REQUIRE(fem::cuda::device_count() >= 1);
  const auto info = fem::cuda::device_info(0);
  std::cout << fem::cuda::to_string(info) << "\n";

  REQUIRE(!info.name.empty());
  REQUIRE(info.cc_major >= 1);
  REQUIRE(info.sm_count >= 1);
  REQUIRE(info.warp_size == 32);
  REQUIRE(info.max_threads_per_block >= 1024);
  REQUIRE(info.shared_mem_per_block >= 48 * 1024);
  REQUIRE(info.shared_mem_per_block_optin >= info.shared_mem_per_block);
  REQUIRE(info.total_global_mem > 0);
  REQUIRE(info.free_global_mem <= info.total_global_mem);

  REQUIRE_THROWS_AS(fem::cuda::device_info(-1), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::cuda::device_info(fem::cuda::device_count()), std::invalid_argument);
}

TEST_CASE("DeviceArray: host -> device -> host round trip", "[cuda][device-array]") {
  const std::vector<double> host = {1.5, -2.0, 3.25, 0.0, 1e-300, 7.0};
  DeviceArray<double> d(host);
  REQUIRE(d.size() == host.size());
  REQUIRE(d.bytes() == host.size() * sizeof(double));
  REQUIRE(d.data() != nullptr);
  REQUIRE(d.to_host() == host);

  std::vector<double> back(2, 99.0);  // wrong size: copy_to_host resizes
  d.copy_to_host(back);
  REQUIRE(back == host);

  const std::vector<int> ints = {0, 1, 1, 2};
  REQUIRE(DeviceArray<int>(ints).to_host() == ints);
}

TEST_CASE("DeviceArray: zero, resize, size checks", "[cuda][device-array]") {
  DeviceArray<double> d(std::vector<double>(5, 3.0));
  d.zero();
  REQUIRE(d.to_host() == std::vector<double>(5, 0.0));

  REQUIRE_THROWS_AS(d.copy_from_host(std::vector<double>(4)), std::invalid_argument);

  const double* before = d.data();
  d.resize(5);  // same size: keeps the allocation
  REQUIRE(d.data() == before);
  d.resize(8);
  REQUIRE(d.size() == 8);
  d.copy_from_host(std::vector<double>(8, 1.0));
  REQUIRE(d.to_host() == std::vector<double>(8, 1.0));
}

TEST_CASE("DeviceArray: empty arrays allocate nothing", "[cuda][device-array]") {
  DeviceArray<double> a;
  REQUIRE(a.empty());
  REQUIRE(a.data() == nullptr);
  DeviceArray<double> b(std::vector<double>{});
  REQUIRE(b.data() == nullptr);
  b.zero();  // no-op, no CUDA call
  REQUIRE(b.to_host().empty());
}

TEST_CASE("DeviceArray: move transfers ownership", "[cuda][device-array]") {
  DeviceArray<double> a(std::vector<double>{1, 2, 3});
  const double* p = a.data();

  DeviceArray<double> b(std::move(a));
  REQUIRE(b.data() == p);
  REQUIRE(b.size() == 3);
  REQUIRE(a.data() == nullptr);  // NOLINT(bugprone-use-after-move): checking the moved-from state
  REQUIRE(a.size() == 0);

  DeviceArray<double> c(std::vector<double>{9});
  c = std::move(b);  // c's old allocation is freed
  REQUIRE(c.data() == p);
  REQUIRE(c.to_host() == std::vector<double>{1, 2, 3});
  REQUIRE(b.data() == nullptr);  // NOLINT(bugprone-use-after-move)
}

TEST_CASE("CUDA_CHECK throws on failure and clears the error", "[cuda][cuda-check]") {
  // Far more than any GPU has: cudaMalloc fails with cudaErrorMemoryAllocation.
  const std::size_t too_many = std::size_t(1) << 50;
  try {
    DeviceArray<double> huge(too_many);
    FAIL("allocating 8 PiB should have thrown");
  } catch (const std::runtime_error& e) {
    REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("cudaErrorMemoryAllocation"));
    REQUIRE_THAT(e.what(), Catch::Matchers::ContainsSubstring("cudaMalloc"));
  }
  // The failed allocation must not be reported again by the next check.
  REQUIRE_NOTHROW(CUDA_CHECK_LAUNCH());

  // A launch with too many threads per block fails at launch time.
  DeviceArray<double> x(std::vector<double>(4, 0.0));
  scale_add_index<<<1, 4096>>>(x.data(), 1.0, 4);
  REQUIRE_THROWS_AS(CUDA_CHECK_LAUNCH(), std::runtime_error);
  REQUIRE_NOTHROW(CUDA_CHECK_LAUNCH());
}

TEST_CASE("smoke test: a kernel runs and its result comes back", "[cuda][smoke]") {
  const int n = 100000;
  std::vector<double> host(n);
  for (int i = 0; i < n; ++i) host[i] = 0.5 * i;
  DeviceArray<double> x(host);

  const int threads = 256;
  const int blocks = 64;  // deliberately fewer than n / threads: the grid-stride loop covers the rest
  scale_add_index<<<blocks, threads>>>(x.data(), 2.0, n);
  CUDA_CHECK_LAUNCH();
  CUDA_CHECK(cudaDeviceSynchronize());

  const std::vector<double> result = x.to_host();
  for (int i = 0; i < n; ++i) {
    REQUIRE(result[i] == 2.0 * host[i] + i);  // exact: small integers and halves
  }
}
