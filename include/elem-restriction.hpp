#pragma once
#include <array>
#include <vector>
#include "transpose-mode.hpp"  // ContractMode, as t_mode below

namespace fem {

// Maps between an L-vector (the global solution vector, one entry per
// mesh DOF) and an E-vector (element vector: each element's own private
// copy of its local DOFs -- nodes shared between elements are
// duplicated, once per element that touches them).
//
// Two kinds, mirroring libCEED's CeedElemRestriction (no blocking, no
// orientations/curl-conforming):
//   Offset:  an explicit offsets array, for nodes shared between elements.
//   Strided: L-vector index = i*strides[0] + k*strides[1] + e*strides[2]
//            (node i, component k, element e); nothing is shared. Used for
//            quadrature-point data (qdata). "Backend strides" means our own
//            E/Q-vector layout, so gather/scatter is a plain copy -- the
//            layout of anything stored that way belongs to the backend, and
//            is only ever read or written through the restriction.
enum class RestrictionType { Offset, Strided };

struct ElemRestriction {
  RestrictionType type = RestrictionType::Offset;
  int num_elem;     // number of elements described by offsets
  int elem_size;    // local nodes per element (e.g. P_1d^dim)
  int num_comp;     // field components per node
  int comp_stride;  // stride between components for the same L-vector node
  int l_size;       // size of the L-vector
  int e_size;       // size of the E-vector: num_comp * elem_size * num_elem

  // Row-major [num_elem][elem_size]: offsets[i + e*elem_size] is the
  // L-vector index of local node i of element e. All values must be in
  // [0, l_size).
  std::vector<int> offsets;  // Offset only

  // Strided only: (node, component, element) strides into the L-vector.
  std::array<int, 3> strides{};
  bool has_backend_strides = false;  // strides == our own E/Q-vector layout
};

void elem_restriction_apply(const ElemRestriction& restriction, ContractMode t_mode,
                             const std::vector<double>& in, std::vector<double>& out);

void elem_restriction_create(const int num_elem, const int elem_size, const int num_comp, const int comp_stride,
                             const int l_size, const std::vector<int>& offsets, ElemRestriction& restriction);

// Backend strides: {num_elem, elem_size * num_elem, 1}, i.e. the E-vector layout
// E[(k * elem_size + i) * num_elem + e] -- so the L-vector is laid out exactly like
// the Q-vectors tensor_basis_apply_* produce, and like QFunction inputs/outputs.
void elem_restriction_create_strided(const int num_elem, const int elem_size, const int num_comp, const int l_size,
                                     ElemRestriction& restriction);

// User-provided (node, component, element) strides, e.g. to wrap data laid out elsewhere.
void elem_restriction_create_strided(const int num_elem, const int elem_size, const int num_comp, const int l_size,
                                     const std::array<int, 3>& strides, ElemRestriction& restriction);

void elem_restriction_get_multiplicity(const ElemRestriction& restriction, std::vector<double>& multiplicity);

} // namespace fem
