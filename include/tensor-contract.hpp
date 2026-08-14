#pragma once
#include <vector>
#include "basis.hpp"

namespace fem {

enum class ContractMode { NoTranspose, Transpose };

void tensor_contract_apply(int A, int B, int C, int J,
                            const std::vector<double>& t,
                            ContractMode mode, bool add,
                            const std::vector<double>& u,
                            std::vector<double>& v);

// Interpolate nodal values to quadrature-point values for a single
// scalar field on a single element, generalizing basis.interp_1d to
// basis.dim dimensions via tensor_contract_apply, once per axis --
// mirrors libCEED's CeedBasisApplyCore_Ref pre/post driving loop
// (~/RATEL/libCEED/backends/ref/ceed-ref-basis.c:57-83, CEED_EVAL_INTERP
// case), specialized to num_comp=1, num_elem=1 (no batching yet).
//
// Node/quadrature-point ordering is row-major with axis 0 (x) fastest:
//   u[iz*P_1d*P_1d + iy*P_1d + ix]   (dim=3), u[iy*P_1d + ix] (dim=2),
//   u[ix] (dim=1) -- P_1d^dim entries total.
// v follows the same convention with Q_1d in place of P_1d, Q_1d^dim
// entries total.
void tensor_basis_apply_interp(const TensorBasis& basis,
                                const std::vector<double>& u,
                                std::vector<double>& v);

// Gradient at quadrature points for a single scalar field on a single
// element. v is `dim` blocks of size Q_1d^dim (same layout as above):
//   v[d_axis * Q_1d^dim + q] = d/dx_{d_axis} of the interpolated field
//                              at quadrature point q
// Block d_axis is computed with basis.grad_1d applied on axis d_axis and
// basis.interp_1d on every other axis (dim tensor_contract_apply calls
// per block, dim*dim calls total) -- see CEED_EVAL_GRAD in the same
// libCEED reference above.
void tensor_basis_apply_grad(const TensorBasis& basis,
                              const std::vector<double>& u,
                              std::vector<double>& v);

} // namespace fem
