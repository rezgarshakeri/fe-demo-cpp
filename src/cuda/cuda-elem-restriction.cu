#include "cuda/cuda-elem-restriction.hpp"
#include <stdexcept>
#include "cuda/cuda-check.hpp"

namespace fem::cuda {

// =============================================================================
// Kernels
//
// Layouts (same as the CPU version, src/elem-restriction.cpp), for node i,
// component k, element e:
//   E-vector:            E[(k * elem_size + i) * num_elem + e]   (element index innermost)
//   L-vector (Offset):   L[offsets[i + e * elem_size] + k * comp_stride]
//   L-vector (Strided):  L[i * strides[0] + k * strides[1] + e * strides[2]]
//
// Every kernel is a grid-stride loop over its work items, so it is correct for
// any grid size (see scale_add_index in tests/cuda/test-device.cu).
//
// Thread -> (e, i): consecutive threads take consecutive nodes of the same
// element (e = n / elem_size, i = n % elem_size). Reads of the offsets are then
// contiguous and an element's L-vector nodes are close together; the E-vector
// writes are num_elem apart, but neighbouring warps fill the same cache lines
// and L2 merges them. Measured on an RTX PRO 1000 (Blackwell) for 512k
// quadratic hexes: ~3x faster than the mapping that makes the writes
// contiguous (e = n % num_elem), whose offsets reads are elem_size apart.
//
// libCEED reference: include/ceed/jit-source/cuda/cuda-ref-restriction-offset.h
//                    include/ceed/jit-source/cuda/cuda-ref-restriction-strided.h
// (libCEED's E-vector has the local node innermost instead of the element.)
// =============================================================================

namespace {

// Offset gather (NoTranspose): v = E-vector, u = L-vector.
// One thread per E-vector node, looping over the components. No race: each
// E-vector entry is written by exactly one thread.
__global__ void offset_gather_kernel(int num_elem, int elem_size, int num_comp, int comp_stride,
                                     const int* __restrict__ offsets, const double* __restrict__ u,
                                     double* __restrict__ v) {
  const int num_items = num_elem * elem_size;
  for (int node = blockIdx.x * blockDim.x + threadIdx.x; node < num_items; node += blockDim.x * gridDim.x) {
    const int elem = node / elem_size;
    const int local_node = node % elem_size;
    const int offset = offsets[local_node + elem * elem_size];
    for (int comp = 0; comp < num_comp; ++comp) {
      v[(comp * elem_size + local_node) * num_elem + elem] = u[offset + comp * comp_stride];
    }
  }
}

// Offset scatter-add (Transpose), atomic: u = E-vector, v = L-vector.
// One thread per E-vector node. Elements sharing an L-vector node add into it
// at the same time, so the addition is atomic; the order of the additions (and
// so the last bits of the result) varies from run to run.
__global__ void offset_scatter_atomic_kernel(int num_elem, int elem_size, int num_comp, int comp_stride,
                                             const int* __restrict__ offsets, const double* __restrict__ u,
                                             double* __restrict__ v) {
  const int num_items = num_elem * elem_size;
  for (int node = blockIdx.x * blockDim.x + threadIdx.x; node < num_items; node += blockDim.x * gridDim.x) {
    const int elem = node / elem_size;
    const int local_node = node % elem_size;
    const int offset = offsets[local_node + elem * elem_size];
    for (int comp = 0; comp < num_comp; ++comp) {
      atomicAdd(&v[offset + comp * comp_stride], u[(comp * elem_size + local_node) * num_elem + elem]);
    }
  }
}

// Offset scatter-add (Transpose), deterministic: u = E-vector, v = L-vector.
// One thread per used L-vector node n. It sums, in a register, the E-vector
// entries of the E-nodes t_indices[t_offsets[n] .. t_offsets[n+1]) (each stored
// as local_node + elem * elem_size), then adds the sum to v once. Each thread
// owns its L-vector node: no race, no atomics, and a fixed summation order, so
// the result is bitwise identical on every run.
__global__ void offset_scatter_deterministic_kernel(int num_nodes, int num_elem, int elem_size, int num_comp,
                                                    int comp_stride, const int* __restrict__ l_vec_indices,
                                                    const int* __restrict__ t_offsets,
                                                    const int* __restrict__ t_indices, const double* __restrict__ u,
                                                    double* __restrict__ v) {
  for (int node = blockIdx.x * blockDim.x + threadIdx.x; node < num_nodes; node += blockDim.x * gridDim.x) {
    const int l_index = l_vec_indices[node];
    const int t_start = t_offsets[node];
    const int t_end = t_offsets[node + 1];
    for (int comp = 0; comp < num_comp; ++comp) {
      double sum = 0.0;
      for (int t = t_start; t < t_end; ++t) {
        const int e_node = t_indices[t];
        const int elem = e_node / elem_size;
        const int local_node = e_node % elem_size;
        sum += u[(comp * elem_size + local_node) * num_elem + elem];
      }
      v[l_index + comp * comp_stride] += sum;  // add, don't overwrite: Transpose accumulates into v
    }
  }
}

// Strided gather (NoTranspose): v = E-vector, u = L-vector.
// One thread per E-vector node, looping over the components. With backend
// strides {num_elem, elem_size * num_elem, 1} the two indices are equal: a copy.
__global__ void strided_gather_kernel(int num_elem, int elem_size, int num_comp, int stride_node, int stride_comp,
                                      int stride_elem, const double* __restrict__ u, double* __restrict__ v) {
  const int num_items = num_elem * elem_size;
  for (int node = blockIdx.x * blockDim.x + threadIdx.x; node < num_items; node += blockDim.x * gridDim.x) {
    const int elem = node / elem_size;
    const int local_node = node % elem_size;
    for (int comp = 0; comp < num_comp; ++comp) {
      v[(comp * elem_size + local_node) * num_elem + elem] =
          u[local_node * stride_node + comp * stride_comp + elem * stride_elem];
    }
  }
}

// Strided scatter-add (Transpose): u = E-vector, v = L-vector.
// Nothing is shared between elements, so each L-vector entry receives exactly
// one contribution and a plain += has no race (as in libCEED).
__global__ void strided_scatter_kernel(int num_elem, int elem_size, int num_comp, int stride_node, int stride_comp,
                                       int stride_elem, const double* __restrict__ u, double* __restrict__ v) {
  const int num_items = num_elem * elem_size;
  for (int node = blockIdx.x * blockDim.x + threadIdx.x; node < num_items; node += blockDim.x * gridDim.x) {
    const int elem = node / elem_size;
    const int local_node = node % elem_size;
    for (int comp = 0; comp < num_comp; ++comp) {
      v[local_node * stride_node + comp * stride_comp + elem * stride_elem] +=
          u[(comp * elem_size + local_node) * num_elem + elem];
    }
  }
}

// Launch configuration: 256 threads per block (a multiple of the warp size, 32),
// enough blocks to give every work item its own thread. The grid-stride loop
// in the kernels would also be correct with fewer blocks.
constexpr int kThreadsPerBlock = 256;
int num_blocks(int num_items) { return (num_items + kThreadsPerBlock - 1) / kThreadsPerBlock; }

}  // namespace

// =============================================================================
// Host side: setup and dispatch
// =============================================================================

/**
  @brief Build the inverse of the offsets (CSR), for the deterministic scatter

  @param[in]  restriction An Offset restriction
  @param[out] map         See TransposeMap in cuda/cuda-elem-restriction.hpp

  @ref CeedElemRestrictionOffset_Cuda (backends/cuda-ref/ceed-cuda-ref-restriction.c)
**/
void build_transpose_map(const fem::ElemRestriction& restriction, TransposeMap& map) {
  if (restriction.type != RestrictionType::Offset) {
    throw std::invalid_argument("build_transpose_map: only for Offset restrictions");
  }
  const int num_e_nodes = restriction.num_elem * restriction.elem_size;
  const std::vector<int>& offsets = restriction.offsets;

  // 1. Which L-vector nodes are used, numbered in increasing L-vector order.
  std::vector<int> node_of_l_index(restriction.l_size, -1);
  for (int t = 0; t < num_e_nodes; ++t) node_of_l_index[offsets[t]] = 0;
  map.l_vec_indices.clear();
  for (int l = 0; l < restriction.l_size; ++l) {
    if (node_of_l_index[l] == 0) {
      node_of_l_index[l] = static_cast<int>(map.l_vec_indices.size());
      map.l_vec_indices.push_back(l);
    }
  }
  const int num_nodes = static_cast<int>(map.l_vec_indices.size());

  // 2. Count the E-nodes per L-node, prefix sum -> CSR row starts.
  map.t_offsets.assign(num_nodes + 1, 0);
  for (int t = 0; t < num_e_nodes; ++t) ++map.t_offsets[node_of_l_index[offsets[t]] + 1];
  for (int n = 0; n < num_nodes; ++n) map.t_offsets[n + 1] += map.t_offsets[n];

  // 3. Fill each row; t increases, so every row comes out sorted.
  map.t_indices.assign(num_e_nodes, 0);
  std::vector<int> next(map.t_offsets.begin(), map.t_offsets.end() - 1);
  for (int t = 0; t < num_e_nodes; ++t) map.t_indices[next[node_of_l_index[offsets[t]]]++] = t;
}

/**
  @brief Copy a CPU `ElemRestriction` to the device; build the transpose map for Deterministic

  @param[in]  host         A restriction made by fem::elem_restriction_create or _create_strided
  @param[in]  scatter_mode How Transpose handles shared nodes (Offset only)
  @param[out] restriction  The device restriction; its index arrays are copied here, once

  @ref CeedElemRestrictionCreate_Cuda (backends/cuda-ref/ceed-cuda-ref-restriction.c)
**/
void elem_restriction_create(const fem::ElemRestriction& host, ScatterMode scatter_mode, ElemRestriction& restriction) {
  ElemRestriction r;
  r.type = host.type;
  r.num_elem = host.num_elem;
  r.elem_size = host.elem_size;
  r.num_comp = host.num_comp;
  r.comp_stride = host.comp_stride;
  r.l_size = host.l_size;
  r.e_size = host.e_size;
  r.strides = host.strides;
  r.scatter_mode = scatter_mode;

  if (host.type == RestrictionType::Offset) {
    if (static_cast<int>(host.offsets.size()) != host.num_elem * host.elem_size) {
      throw std::invalid_argument("cuda::elem_restriction_create: offsets size != num_elem * elem_size");
    }
    r.offsets = DeviceArray<int>(host.offsets);
    if (scatter_mode == ScatterMode::Deterministic) {
      TransposeMap map;
      build_transpose_map(host, map);
      r.num_nodes = static_cast<int>(map.l_vec_indices.size());
      r.l_vec_indices = DeviceArray<int>(map.l_vec_indices);
      r.t_offsets = DeviceArray<int>(map.t_offsets);
      r.t_indices = DeviceArray<int>(map.t_indices);
    }
  }
  restriction = std::move(r);  // only replace the output once everything succeeded
}

/**
  @brief Gather (NoTranspose) or scatter-add (Transpose) on the device

  Asynchronous: launches the kernel and returns; the result is visible after the
  next synchronizing call (e.g. out.copy_to_host).

  @param[in]     restriction The device restriction
  @param[in]     t_mode      NoTranspose: L -> E.  Transpose: E -> L, added into out
  @param[in]     in          NoTranspose: l_size entries.  Transpose: e_size entries
  @param[in,out] out         NoTranspose: resized to e_size.  Transpose: must have l_size entries

  @ref CeedElemRestrictionApply_Cuda (backends/cuda-ref/ceed-cuda-ref-restriction.c)
**/
void elem_restriction_apply(const ElemRestriction& restriction, ContractMode t_mode, const DeviceArray<double>& in,
                            DeviceArray<double>& out) {
  const ElemRestriction& r = restriction;
  const bool transpose = (t_mode == ContractMode::Transpose);
  const int in_size = transpose ? r.e_size : r.l_size;
  if (static_cast<int>(in.size()) != in_size) {
    throw std::invalid_argument("cuda::elem_restriction_apply: in has the wrong size");
  }
  if (transpose) {
    if (static_cast<int>(out.size()) != r.l_size) {
      throw std::invalid_argument(
          "cuda::elem_restriction_apply: Transpose adds into out, which must already have l_size entries");
    }
  } else {
    out.resize(r.e_size);  // fully overwritten, so no need to zero it
  }

  const int num_e_nodes = r.num_elem * r.elem_size;
  if (num_e_nodes == 0) return;
  const double* u = in.data();
  double* v = out.data();
  const auto& s = r.strides;

  if (r.type == RestrictionType::Strided) {
    if (!transpose) {
      strided_gather_kernel<<<num_blocks(num_e_nodes), kThreadsPerBlock>>>(r.num_elem, r.elem_size, r.num_comp,
                                                                           s[0], s[1], s[2], u, v);
    } else {
      strided_scatter_kernel<<<num_blocks(num_e_nodes), kThreadsPerBlock>>>(r.num_elem, r.elem_size, r.num_comp,
                                                                            s[0], s[1], s[2], u, v);
    }
  } else if (!transpose) {
    offset_gather_kernel<<<num_blocks(num_e_nodes), kThreadsPerBlock>>>(r.num_elem, r.elem_size, r.num_comp,
                                                                        r.comp_stride, r.offsets.data(), u, v);
  } else if (r.scatter_mode == ScatterMode::Atomic) {
    offset_scatter_atomic_kernel<<<num_blocks(num_e_nodes), kThreadsPerBlock>>>(
        r.num_elem, r.elem_size, r.num_comp, r.comp_stride, r.offsets.data(), u, v);
  } else {
    offset_scatter_deterministic_kernel<<<num_blocks(r.num_nodes), kThreadsPerBlock>>>(
        r.num_nodes, r.num_elem, r.elem_size, r.num_comp, r.comp_stride, r.l_vec_indices.data(),
        r.t_offsets.data(), r.t_indices.data(), u, v);
  }
  CUDA_CHECK_LAUNCH();
}

} // namespace fem::cuda
