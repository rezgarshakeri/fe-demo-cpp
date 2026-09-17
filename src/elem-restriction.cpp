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

/**
  @brief Create an `ElemRestriction` from the parameters describing the mapping between an L-vector and an E-vector

  @param[in]  num_elem    Number of elements
  @param[in]  elem_size   Size (number of "nodes") per element
  @param[in]  num_comp    Number of field components per interpolation node (1 for scalar fields)
  @param[in]  comp_stride Stride between components for the same L-vector "node".
                            Data for node `i`, component `j`, element `k` can be found in the L-vector at index `offsets[i + k*elem_size] + j*comp_stride`.
  @param[in]  l_size      The size of the L-vector.
                            This vector may be larger than the elements and fields given by this restriction.
  @param[in]  offsets     Array of shape `[num_elem, elem_size]`.
                            Row `i` holds the ordered list of the offsets for the unknowns corresponding to element `i`, where 0 <= i < @a num_elem.
                            All offsets must be in the range `[0, l_size - 1]`.
  @param[out] restriction  The `ElemRestriction` to fill in
  
  @ref CeedElemRestrictionCreate (interface/ceed-elemrestriction.c)
**/
void elem_restriction_create(const int num_elem, const int elem_size, const int num_comp, const int comp_stride,
                             const int l_size, const std::vector<int>& offsets, ElemRestriction& restriction) {
  if (num_elem <= 0) {
    throw std::invalid_argument("elem_restriction_create: num_elem must be positive");
  }
  if (elem_size <= 0) {
    throw std::invalid_argument("elem_restriction_create: elem_size must be at least 1");
  }
  if (num_comp <= 0) {
    throw std::invalid_argument("elem_restriction_create: num_comp must be at least 1");
  }
  if (num_comp > 1 && comp_stride <= 0) { 
    throw std::invalid_argument("elem_restriction_create: comp_stride must be at least 1 when num_comp > 1");
  }
  if (l_size <= 0) {
    throw std::invalid_argument("elem_restriction_create: l_size must be positive");
  }
  if (offsets.size() != static_cast<size_t>(num_elem * elem_size)) {
    throw std::invalid_argument("elem_restriction_create: offsets size must be num_elem * elem_size");
  }
  for (int offset : offsets) {
    // apply reads offset + k*comp_stride for every component k, so the last component must fit too
    if (offset < 0 || offset + (num_comp - 1) * comp_stride >= l_size) {
      throw std::invalid_argument("elem_restriction_create: offsets (including component offsets) must be in the range [0, l_size - 1]");
    }
  }
  restriction.num_elem = num_elem;
  restriction.elem_size = elem_size;
  restriction.num_comp = num_comp;
  restriction.comp_stride = comp_stride;
  restriction.l_size = l_size;
  restriction.e_size = num_comp * elem_size * num_elem;
  restriction.offsets = offsets;
}

/**
  @brief Get the multiplicity of each L-vector entry in an `ElemRestriction`, computed as E^T (E 1)

  @param[in]  restriction  `ElemRestriction` to compute the multiplicity of
  @param[out] multiplicity Resized to `l_size`; entry `g` is the number of (element, node, component)
                           slots that map to L-vector entry `g` (0 for entries no element touches)

  @ref CeedElemRestrictionGetMultiplicity (interface/ceed-elemrestriction.c)
**/
void elem_restriction_get_multiplicity(const ElemRestriction& restriction, std::vector<double>& multiplicity) {
  std::vector<double> e_vec;

  // Create e_vec to hold intermediate computation in E^T (E 1)
  e_vec.resize(restriction.e_size);
  // Compute e_vec = E * 1
  multiplicity.assign(restriction.l_size, 1.0);  // multiplicity = 1
  elem_restriction_apply(restriction, ContractMode::NoTranspose, multiplicity, e_vec);
  
  // Compute multiplicity, multiplicity = E^T * e_vec = E^T (E 1)
  multiplicity.assign(restriction.l_size, 0.0);
  elem_restriction_apply(restriction, ContractMode::Transpose, e_vec, multiplicity);
}

} // namespace fem
