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
  @brief Interpolate nodal values to quadrature-point values (num_comp=1, num_elem=1)

  @param[in]  basis The TensorBasis to apply (uses dim, P_1d, Q_1d, interp_1d)
  @param[in]  u     Nodal values, P_1d^dim entries, row-major with axis 0 fastest
  @param[out] v     Quadrature-point values, Q_1d^dim entries, same convention

  @ref libCEED's CeedBasisApplyCore_Ref, CEED_EVAL_INTERP case
       (libCEED/backends/ref/ceed-ref-basis.c)
**/
void tensor_basis_apply_interp(const TensorBasis& basis, const std::vector<double>& u,
                               std::vector<double>& v) {
  int P = basis.P_1d, Q = basis.Q_1d, dim = basis.dim;
  int pre = 1 * int_pow(P, dim - 1), post = 1;
  std::vector<double> tmp[2];

  for (int d = 0; d < dim; d++) {
    tensor_contract_apply(pre, P, post, Q, basis.interp_1d, fem::ContractMode::NoTranspose, false, d == 0 ? u : tmp[d % 2],
                                            d == dim - 1 ? v : tmp[(d + 1) % 2]);
    pre /= P;
    post *= Q;
  }
}

/**
  @brief Gradient at quadrature points (num_comp=1, num_elem=1)

  @param[in]  basis The TensorBasis to apply (uses dim, P_1d, Q_1d, interp_1d, grad_1d)
  @param[in]  u     Nodal values, P_1d^dim entries, row-major with axis 0 fastest
  @param[out] v     dim blocks of Q_1d^dim entries each; block d_axis is d/dx_{d_axis}

  @ref libCEED's CeedBasisApplyCore_Ref, CEED_EVAL_GRAD case
       (libCEED/backends/ref/ceed-ref-basis.c)
**/
void tensor_basis_apply_grad(const TensorBasis& basis, const std::vector<double>& u,
                             std::vector<double>& v) {
  int P = basis.P_1d, Q = basis.Q_1d, dim = basis.dim;
  int Qdim = int_pow(Q, dim);
  v.resize(dim * Qdim);

  for (int d_axis = 0; d_axis < dim; ++d_axis) {
    int pre = int_pow(P, dim - 1), post = 1;
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
    std::copy(component.begin(), component.end(), v.begin() + d_axis * Qdim);
  }
} 

}// namespace fem
