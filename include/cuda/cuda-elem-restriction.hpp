#pragma once
#include <array>
#include <vector>
#include "cuda/device-array.hpp"
#include "elem-restriction.hpp"  // fem::ElemRestriction, fem::RestrictionType
#include "transpose-mode.hpp"

namespace fem::cuda {

// How the Transpose (scatter-add, E -> L) resolves the race on L-vector
// entries shared by several elements:
//   Atomic:        one thread per E-vector node, atomicAdd into the L-vector.
//                  Fast, but the order of the additions varies from run to
//                  run, so results can differ in the last bits.
//   Deterministic: one thread per L-vector node, which sums everything that
//                  maps to it (through the transpose map below) in a fixed
//                  order and adds the sum once. Bitwise identical on every run.
// Mirrors libCEED's cuda-ref backend (USE_DETERMINISTIC in
// cuda-ref-restriction-offset.h). Only used by Offset restrictions: a strided
// restriction shares nothing, so its Transpose has no race.
enum class ScatterMode { Atomic, Deterministic };

// The inverse of the offsets, in CSR form, for the deterministic scatter.
// For the n-th L-vector node that appears in the offsets:
//   l_vec_indices[n]                      its L-vector index (component 0)
//   t_indices[t_offsets[n] .. t_offsets[n+1])
//                                         the E-vector nodes that map to it,
//                                         as i + e*elem_size, in increasing order
// Nodes that no element touches are left out.
// Example, offsets {0,1, 1,2} (2 elements, elem_size 2):
//   l_vec_indices {0, 1, 2}, t_offsets {0, 1, 3, 4}, t_indices {0, 1, 2, 3}
// Mirrors CeedElemRestrictionOffset_Cuda (backends/cuda-ref/ceed-cuda-ref-restriction.c).
struct TransposeMap {
  std::vector<int> l_vec_indices;
  std::vector<int> t_offsets;  // size l_vec_indices.size() + 1
  std::vector<int> t_indices;  // size num_elem * elem_size
};

// Built on the host, at create time. Offset restrictions only.
void build_transpose_map(const fem::ElemRestriction& restriction, TransposeMap& map);

// The GPU version of fem::ElemRestriction: the same sizes, with the index
// arrays in device memory. Created from a CPU restriction, which does the
// validation. Move-only (it owns device memory, see DeviceArray).
// Mirrors CeedElemRestriction_Cuda (backends/cuda-ref/ceed-cuda-ref.h).
struct ElemRestriction {
  RestrictionType type = RestrictionType::Offset;
  int num_elem = 0, elem_size = 0, num_comp = 0, comp_stride = 0;
  int l_size = 0, e_size = 0;
  std::array<int, 3> strides{};  // Strided only
  ScatterMode scatter_mode = ScatterMode::Atomic;

  DeviceArray<int> offsets;  // Offset only (empty for Strided): copy of the host offsets, [num_elem][elem_size]

  // Offset + Deterministic only (empty otherwise): the TransposeMap on the device
  int num_nodes = 0;  // l_vec_indices.size()
  DeviceArray<int> l_vec_indices, t_offsets, t_indices;
};

// Copy a CPU restriction to the device: its sizes, and its index arrays, once,
// here, never per apply. With ScatterMode::Deterministic, also builds the
// TransposeMap on the host and copies it. scatter_mode is ignored for Strided
// restrictions. Throws std::invalid_argument if the offsets don't have
// num_elem * elem_size entries, std::runtime_error if a CUDA call fails;
// `restriction` is only replaced on success.
void elem_restriction_create(const fem::ElemRestriction& host, ScatterMode scatter_mode, ElemRestriction& restriction);

// Same as fem::elem_restriction_apply, on device arrays:
//   NoTranspose: in = L-vector (l_size), out = E-vector, resized to e_size and overwritten.
//   Transpose:   in = E-vector (e_size), out = L-vector, ADDED INTO, not zeroed.
//                out must already have l_size entries (throws otherwise): a
//                freshly allocated device array holds garbage, not zeros
//                (unlike std::vector::resize), so it can't be resized here.
// Asynchronous: this only launches the kernel and returns before the GPU is
// done. The result is visible after the next synchronizing call (e.g.
// out.copy_to_host, or cudaDeviceSynchronize), and an error inside the kernel
// is reported there, not here. Launch errors are reported here.
void elem_restriction_apply(const ElemRestriction& restriction, ContractMode t_mode, const DeviceArray<double>& in,
                            DeviceArray<double>& out);

} // namespace fem::cuda
