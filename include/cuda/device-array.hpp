#pragma once
#include <cstddef>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include "cuda/cuda-check.hpp"

namespace fem::cuda {

// An array in GPU global memory that owns its allocation (RAII): cudaMalloc
// in the constructor, cudaFree in the destructor. Move-only, like
// std::unique_ptr, so an allocation always has exactly one owner and is freed
// exactly once. Copies between host and device are explicit, so every PCIe
// transfer is visible in the code.
//
// Copies are synchronous (cudaMemcpy): simple and safe for now. Streams and
// async copies can come later if profiling shows they matter.
template <class T>
class DeviceArray {
  static_assert(std::is_trivially_copyable_v<T>, "DeviceArray elements are copied as raw bytes");

 public:
  DeviceArray() = default;
  explicit DeviceArray(std::size_t n) { allocate(n); }
  explicit DeviceArray(const std::vector<T>& host) : DeviceArray(host.size()) { copy_from_host(host); }
  ~DeviceArray() { release(); }

  DeviceArray(const DeviceArray&) = delete;
  DeviceArray& operator=(const DeviceArray&) = delete;
  DeviceArray(DeviceArray&& other) noexcept : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
  }
  DeviceArray& operator=(DeviceArray&& other) noexcept {
    if (this != &other) {
      release();
      data_ = other.data_;
      size_ = other.size_;
      other.data_ = nullptr;
      other.size_ = 0;
    }
    return *this;
  }

  // Device pointer: pass it to kernels, never dereference it on the host.
  T* data() { return data_; }
  const T* data() const { return data_; }
  std::size_t size() const { return size_; }
  std::size_t bytes() const { return size_ * sizeof(T); }
  bool empty() const { return size_ == 0; }

  // Reallocate to n entries; the contents are NOT kept. No-op if the size is unchanged.
  void resize(std::size_t n) {
    if (n == size_) return;
    release();
    allocate(n);
  }

  // Set every byte to zero (0.0 for double, 0 for int).
  void zero() {
    if (size_ > 0) CUDA_CHECK(cudaMemset(data_, 0, bytes()));
  }

  // host -> device. host.size() must equal size().
  void copy_from_host(const std::vector<T>& host) {
    if (host.size() != size_) throw std::invalid_argument("DeviceArray::copy_from_host: size mismatch");
    if (size_ > 0) CUDA_CHECK(cudaMemcpy(data_, host.data(), bytes(), cudaMemcpyHostToDevice));
  }

  // device -> host. host is resized to size() (library convention: outputs are resized).
  void copy_to_host(std::vector<T>& host) const {
    host.resize(size_);
    if (size_ > 0) CUDA_CHECK(cudaMemcpy(host.data(), data_, bytes(), cudaMemcpyDeviceToHost));
  }
  std::vector<T> to_host() const {
    std::vector<T> host;
    copy_to_host(host);
    return host;
  }

 private:
  void allocate(std::size_t n) {
    if (n == 0) return;  // no allocation for an empty array; data() stays nullptr
    CUDA_CHECK(cudaMalloc(&data_, n * sizeof(T)));
    size_ = n;
  }
  void release() noexcept {
    if (data_) (void)cudaFree(data_);  // a destructor must not throw; ignore the error
    data_ = nullptr;
    size_ = 0;
  }

  T* data_ = nullptr;
  std::size_t size_ = 0;
};

} // namespace fem::cuda
