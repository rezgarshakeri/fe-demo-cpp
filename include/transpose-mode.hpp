#pragma once

namespace fem {

// Which direction a linear map runs: forward (NoTranspose) or its
// adjoint (Transpose) e.g. nodes -> quadrature points vs. quadrature
// points -> nodes for tensor_contract_apply/tensor_basis_apply_interp/
// _grad, or gather vs. scatter-add for elem_restriction_apply.
//
// Mirrors libCEED's own CeedTransposeMode, defined once in a foundational
// header (ceed/types.h) and reused independently by CeedBasisApply,
// CeedTensorContractApply, and CeedElemRestrictionApply.
enum class ContractMode { NoTranspose, Transpose };

} // namespace fem
