#pragma once
#include <vector>
#include "transpose-mode.hpp"  // ContractMode, as t_mode below

namespace fem {

// Maps between an L-vector (the global solution vector, one entry per
// mesh DOF) and an E-vector (element vector: each element's own private
// copy of its local DOFs -- nodes shared between elements are
// duplicated, once per element that touches them).
//
// Mirrors libCEED's CeedElemRestriction (libCEED/interface/
// ceed-elemrestriction.c), scoped down: offset-based only (no blocking,
// no orientations/curl-conforming, no "strided" identity case yet).
struct ElemRestriction {
  int num_elem;     // number of elements described by offsets
  int elem_size;    // local nodes per element (e.g. P_1d^dim)
  int num_comp;     // field components per node
  int comp_stride;  // stride between components for the same L-vector node
  int l_size;       // size of the L-vector

  // Row-major [num_elem][elem_size]: offsets[i + e*elem_size] is the
  // L-vector index of local node i of element e. All values must be in
  // [0, l_size).
  std::vector<int> offsets;
};

void elem_restriction_apply(const ElemRestriction& restriction, ContractMode t_mode,
                             const std::vector<double>& in, std::vector<double>& out);

} // namespace fem
