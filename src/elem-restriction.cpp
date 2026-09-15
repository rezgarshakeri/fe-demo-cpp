#include "elem-restriction.hpp"
#include <stdexcept>

namespace fem {

/**
  @brief Gather an L-vector into an E-vector, or scatter-add an E-vector back into an L-vector

  @param[in]  restriction The offsets/sizing that define the mapping (see elem-restriction.hpp)
  @param[in]  t_mode      NoTranspose: gather, out = E-vector. Transpose: scatter-add, out = L-vector
  @param[in]  in          NoTranspose: L-vector (size restriction.l_size).
                          Transpose: E-vector (size num_comp*elem_size*num_elem)
  @param[out] out         NoTranspose: resized to num_comp*elem_size*num_elem and overwritten.
                          Transpose: resized to restriction.l_size if not already that size, but
                          NOT zeroed -- results accumulate into whatever it already holds (multiple
                          elements sharing a node must sum, not overwrite; zero it yourself first
                          for a fresh sum)

  All num_elem elements are gathered/scattered in this one call, with num_elem interleaved
  (innermost) in the E-vector layout, matching tensor_basis_apply_interp/_grad's own
  convention, so this restriction's output plugs straight into those. This deliberately differs
  from libCEED's CPU ref/serial backend, which restricts the whole mesh in one call but then
  calls CeedBasisApply once *per element* (num_elem=1 every time). See ceed-ref-operator.c.
  We batch every element through one apply_interp/_grad call instead, matching how libCEED's CUDA
  backends batch (CeedBasisApply(basis, num_elem, ...) once,
  true total) -- the layout choice here is made with that CUDA-style port in mind, once there's a
  working CPU (parallel, e.g. OpenMP/MPI) version to translate.

  @ref libCEED's CeedElemRestrictionApply (interface/ceed-elemrestriction.c),
       CeedElemRestrictionApplyOffsetNoTranspose_Ref_Core / ...OffsetTranspose_Ref_Core
       (backends/ref/ceed-ref-restriction.c) -- ported with block_size fixed to 1 and the
       E-vector layout adjusted to match our own num_elem-batched basis apply, not libCEED's.
**/
void elem_restriction_apply(const ElemRestriction& restriction, ContractMode t_mode,
                             const std::vector<double>& in, std::vector<double>& out) {
  int num_comp = restriction.num_comp, elem_size = restriction.elem_size, num_elem = restriction.num_elem;
  int comp_stride = restriction.comp_stride;
  int l_size = restriction.l_size;

  if (t_mode == ContractMode::NoTranspose) {
    out.resize(num_comp * elem_size * num_elem);
    for (int e = 0; e < num_elem; e++) {
      for (int k = 0; k < num_comp; k++) {
        for (int i = 0; i < elem_size; i++) {
          out[(k * elem_size + i) * num_elem + e] = in[restriction.offsets[i + e * elem_size] + k * comp_stride];
        }
      }
    }
  } else {
    out.resize(l_size);   // not zeroed -- caller's responsibility, same as before
    for (int e = 0; e < num_elem; e++) {
      for (int k = 0; k < num_comp; k++) {
        for (int i = 0; i < elem_size; i++) {
          out[restriction.offsets[i + e * elem_size] + k * comp_stride] += in[(k * elem_size + i) * num_elem + e];
        }
      }
    }
  }
}

} // namespace fem
