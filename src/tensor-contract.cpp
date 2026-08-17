#include "tensor-contract.hpp"
#include <stdexcept>
#include <algorithm>

namespace fem {

  static int int_pow(int base, int exp) {
    int result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
  }

/**
  @brief Batched contraction along one tensor mode
    The contraction is:
      v[(a*J + j)*C + c] += t(j, b) * u[(a*B + b)*C + c]     summed over b

    where t(j, b) = t[j*B + b] if mode == NoTranspose (read `t` as given),
       or t(j, b) = t[b*J + j] if mode == Transpose (read the *same* array as if it were transposed).

  @param[in]  A    Number of outer batches (axes not yet contracted)
  @param[in]  B    Input size of the contracted axis
  @param[in]  C    Number of inner batches (axes already contracted, e.g. num_elem)
  @param[in]  J    Output size of the contracted axis
  @param[in]  t    J x B matrix, row-major (t[j * B + b]), e.g. interp_1d/grad_1d
  @param[in]  mode NoTranspose reads t as given; Transpose reads the same data as if
                   it were B x J (no separate transposed copy needed)
  @param[in]  add  If false, v is resized to A*J*C and overwritten; if true, v must
                   already be sized A*J*C and results accumulate into it
  @param[in]  u    Flattened, row-major [A][B][C] tensor
  @param[out] v    Flattened, row-major [A][J][C] tensor

  @ref libCEED's CeedTensorContractApply (ceed-ref-tensor.c)
**/
void tensor_contract_apply(int A, int B, int C, int J,
                            const std::vector<double>& t,
                            ContractMode mode, bool add,
                            const std::vector<double>& u,
                            std::vector<double>& v) {
  int t_stride_0 = (mode == ContractMode::NoTranspose) ? B : 1;
  int t_stride_1 = (mode == ContractMode::NoTranspose) ? 1 : J;

  v.resize(static_cast<size_t>(A * J * C));
  if (!add) {
    std::fill(v.begin(), v.end(), 0.0);
  }

  for (int a = 0; a < A; ++a) {
    for (int j = 0; j < J; ++j) {
      for (int c = 0; c < C; ++c) {
        double sum = 0.0;
        for (int b = 0; b < B; ++b) {
          sum += t[j * t_stride_0 + b * t_stride_1] * u[(a*B + b) * C + c];
        }
        v[(a*J + j) * C + c] += sum;
      }
    }
  }
}

/**
  @brief Interpolate nodal values to quadrature-point values

  @param[in]  basis    The TensorBasis to apply (uses dim, P_1d, Q_1d, num_comp, interp_1d)
  @param[in]  num_elem Number of elements batched together (default 1)
  @param[in]  u        Nodal values -- see tensor-contract.hpp for the exact
                       (component, node, element) layout
  @param[out] v        Quadrature-point values, same layout convention

  @ref libCEED's CeedBasisApplyCore_Ref, CEED_EVAL_INTERP case
       (libCEED/backends/ref/ceed-ref-basis.c)
**/
void tensor_basis_apply_interp(const TensorBasis& basis, int num_elem, const std::vector<double>& u,
                               std::vector<double>& v) {
  int P = basis.P_1d, Q = basis.Q_1d, dim = basis.dim, num_comp = basis.num_comp;
  int pre = num_comp * int_pow(P, dim - 1), post = num_elem;
  std::vector<double> tmp[2];

  for (int d = 0; d < dim; d++) {
    tensor_contract_apply(pre, P, post, Q, basis.interp_1d, fem::ContractMode::NoTranspose, false, d == 0 ? u : tmp[d % 2],
                                            d == dim - 1 ? v : tmp[(d + 1) % 2]);
    pre /= P;
    post *= Q;
  }
}

/**
  @brief Gradient at quadrature points

  @param[in]  basis    The TensorBasis to apply (uses dim, P_1d, Q_1d, num_comp, interp_1d, grad_1d)
  @param[in]  num_elem Number of elements batched together (default 1)
  @param[in]  u        Nodal values -- see tensor-contract.hpp for the exact
                       (component, node, element) layout
  @param[out] v        dim blocks; within a block, layout matches tensor_basis_apply_interp

  @ref libCEED's CeedBasisApplyCore_Ref, CEED_EVAL_GRAD case
       (libCEED/backends/ref/ceed-ref-basis.c)
**/
void tensor_basis_apply_grad(const TensorBasis& basis, int num_elem, const std::vector<double>& u,
                             std::vector<double>& v) {
  int P = basis.P_1d, Q = basis.Q_1d, dim = basis.dim, num_comp = basis.num_comp;
  int Qdim = int_pow(Q, dim);
  v.resize(dim * num_comp * Qdim * num_elem);

  for (int d_axis = 0; d_axis < dim; ++d_axis) {
    int pre = num_comp * int_pow(P, dim - 1), post = num_elem;
    std::vector<double> tmp[2], component;
    for (int d = 0; d < dim; ++d) {
      const auto& bb = (d == d_axis) ? basis.grad_1d : basis.interp_1d;
      tensor_contract_apply(pre, P, post, Q, bb, ContractMode::NoTranspose, false,
                             d == 0 ? u : tmp[d % 2],
                             d == dim - 1 ? component : tmp[(d + 1) % 2]);
      pre /= P;
      post *= Q;
    }
    // component now holds this axis's derivative, Qdim entries
    // copy into its slice of v, since tensor_contract_apply would otherwise
    // resize/overwrite all of v rather than just this block.
    std::copy(component.begin(), component.end(), v.begin() + d_axis * num_comp * Qdim * num_elem);
  }
}

/**
  @brief Dispatch to tensor_basis_apply_interp or tensor_basis_apply_grad by eval_mode

  @param[in]  basis     The TensorBasis to apply
  @param[in]  num_elem  Number of elements batched together
  @param[in]  eval_mode EvalMode::Interp or EvalMode::Grad
  @param[in]  u         Nodal values
  @param[out] v         Result

  @ref libCEED's CeedBasisApply (interface/ceed-basis.c)
**/
void tensor_basis_apply(const TensorBasis& basis, int num_elem,EvalMode eval_mode,
                         const std::vector<double>& u, std::vector<double>& v) {
  switch (eval_mode) {
    case EvalMode::Interp:
      tensor_basis_apply_interp(basis, num_elem, u, v);
      break;
    case EvalMode::Grad:
      tensor_basis_apply_grad(basis, num_elem, u, v);
      break;
    default:
      throw std::invalid_argument("tensor_basis_apply: unknown eval_mode");
  }
}

}// namespace fem
