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

    where t(j, b) = t[j*B + b] if t_mode == NoTranspose (read `t` as given),
       or t(j, b) = t[b*J + j] if t_mode == Transpose (read the *same* array as if it were transposed).

  @param[in]  A      Number of outer batches (axes not yet contracted)
  @param[in]  B      Input size of the contracted axis
  @param[in]  C      Number of inner batches (axes already contracted, e.g. num_elem)
  @param[in]  J      Output size of the contracted axis
  @param[in]  t      J x B matrix, row-major (t[j * B + b]), e.g. interp_1d/grad_1d
  @param[in]  t_mode NoTranspose reads t as given; Transpose reads the same data as if
                     it were B x J (no separate transposed copy needed)
  @param[in]  add    If false, v is resized to A*J*C and overwritten; if true, v must
                     already be sized A*J*C and results accumulate into it
  @param[in]  u      Flattened, row-major [A][B][C] tensor
  @param[out] v      Flattened, row-major [A][J][C] tensor

  @ref libCEED's CeedTensorContractApply (ceed-ref-tensor.c)
**/
void tensor_contract_apply(int A, int B, int C, int J,
                            const std::vector<double>& t,
                            ContractMode t_mode, bool add,
                            const std::vector<double>& u,
                            std::vector<double>& v) {
  int t_stride_0 = (t_mode == ContractMode::NoTranspose) ? B : 1;
  int t_stride_1 = (t_mode == ContractMode::NoTranspose) ? 1 : J;

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

// Core of interp: apply_add = false overwrites v, true adds into it (v must
// already have the output size). Only the LAST contraction pass writes v, so
// only it adds; the intermediate passes always overwrite their scratch arrays.
//
// @ref libCEED's CeedBasisApplyCore_Ref (backends/ref/ceed-ref-basis.c), whose
//      apply_add flag plays the same role
static void basis_apply_interp_core(const TensorBasis& basis, bool apply_add, int num_elem, ContractMode t_mode,
                                    const std::vector<double>& u, std::vector<double>& v) {
  int P = basis.P_1d, Q = basis.Q_1d, dim = basis.dim, num_comp = basis.num_comp;
  if (t_mode == ContractMode::Transpose) {
    P = basis.Q_1d;
    Q = basis.P_1d;
  }
  int pre = num_comp * int_pow(P, dim - 1), post = num_elem;
  std::vector<double> tmp[2];

  for (int d = 0; d < dim; d++) {
    const bool is_last = (d == dim - 1);
    tensor_contract_apply(pre, P, post, Q, basis.interp_1d, t_mode, is_last && apply_add, d == 0 ? u : tmp[d % 2],
                          is_last ? v : tmp[(d + 1) % 2]);
    pre /= P;
    post *= Q;
  }
}

// Core of grad, same apply_add meaning (apply_add is only used with Transpose).
//   NoTranspose: v = dim blocks of num_comp * Q_1d^dim * num_elem, block d_axis
//                = derivative along axis d_axis (grad_1d on that axis, interp_1d
//                on the others).
//   Transpose:   u = those dim blocks; every block's transpose is summed into
//                the same v (nodal values). So v is zeroed first unless
//                apply_add, and every d_axis pass adds its last contraction.
//
// TODO(perf): the u_slice copy (Transpose, one per d_axis) is only there because
// tensor_contract_apply takes const std::vector<double>&, which can't point
// partway into an existing array. Raw pointers (libCEED's CeedScalar* + offset
// style) would make it zero-copy; switch if profiling shows it matters.
//
// @ref libCEED's CeedBasisApplyCore_Ref (backends/ref/ceed-ref-basis.c), CEED_EVAL_GRAD
static void basis_apply_grad_core(const TensorBasis& basis, bool apply_add, int num_elem, ContractMode t_mode,
                                  const std::vector<double>& u, std::vector<double>& v) {
  const bool transpose = (t_mode == ContractMode::Transpose);
  const int dim = basis.dim, num_comp = basis.num_comp;
  const int P = transpose ? basis.Q_1d : basis.P_1d;  // contraction input size
  const int Q = transpose ? basis.P_1d : basis.Q_1d;  // contraction output size
  const int block = num_comp * int_pow(basis.Q_1d, dim) * num_elem;  // one derivative block (Q-vector side)

  if (transpose) {
    if (!apply_add) v.assign(num_comp * int_pow(basis.P_1d, dim) * num_elem, 0.0);
  } else {
    v.resize(dim * block);
  }

  std::vector<double> tmp[2], component, u_slice;
  for (int d_axis = 0; d_axis < dim; ++d_axis) {
    int pre = num_comp * int_pow(P, dim - 1), post = num_elem;
    if (transpose) u_slice.assign(u.begin() + d_axis * block, u.begin() + (d_axis + 1) * block);

    for (int d = 0; d < dim; ++d) {
      const std::vector<double>& t_1d = (d == d_axis) ? basis.grad_1d : basis.interp_1d;
      const bool is_last = (d == dim - 1);
      const std::vector<double>& in = (d == 0) ? (transpose ? u_slice : u) : tmp[d % 2];
      std::vector<double>& out = is_last ? (transpose ? v : component) : tmp[(d + 1) % 2];
      tensor_contract_apply(pre, P, post, Q, t_1d, t_mode, is_last && transpose, in, out);
      pre /= P;
      post *= Q;
    }
    // NoTranspose: tensor_contract_apply resizes its output, so it can't write
    // into one block of v directly; copy this axis's derivative into its block.
    if (!transpose) std::copy(component.begin(), component.end(), v.begin() + d_axis * block);
  }
}

/**
  @brief Interpolate nodal values to quadrature-point values (or the transpose); overwrites v

  @param[in]  basis    The TensorBasis to apply (uses dim, P_1d, Q_1d, num_comp, interp_1d)
  @param[in]  num_elem Number of elements batched together
  @param[in]  t_mode   ContractMode::NoTranspose or ContractMode::Transpose
  @param[in]  u        Nodal values (quadrature-point values for Transpose)
  @param[out] v        Quadrature-point values (nodal values for Transpose), overwritten

  @ref libCEED's CeedBasisApplyCore_Ref, CEED_EVAL_INTERP case
       (libCEED/backends/ref/ceed-ref-basis.c)
**/
void tensor_basis_apply_interp(const TensorBasis& basis, int num_elem, ContractMode t_mode,
                               const std::vector<double>& u, std::vector<double>& v) {
  basis_apply_interp_core(basis, false, num_elem, t_mode, u, v);
}

/**
  @brief Gradient at quadrature points (or the transpose); overwrites v

  @param[in]  basis    The TensorBasis to apply (uses dim, P_1d, Q_1d, num_comp, interp_1d, grad_1d)
  @param[in]  num_elem Number of elements batched together
  @param[in]  t_mode   ContractMode::NoTranspose or ContractMode::Transpose
  @param[in]  u        Nodal values (for Transpose: dim blocks of quadrature-point values)
  @param[out] v        dim blocks; within a block, layout matches tensor_basis_apply_interp
                       (for Transpose: nodal values, the sum over the dim blocks), overwritten

  @ref libCEED's CeedBasisApplyCore_Ref, CEED_EVAL_GRAD case
       (libCEED/backends/ref/ceed-ref-basis.c)
**/
void tensor_basis_apply_grad(const TensorBasis& basis, int num_elem, ContractMode t_mode,
                             const std::vector<double>& u, std::vector<double>& v) {
  basis_apply_grad_core(basis, false, num_elem, t_mode, u, v);
}

/**
  @brief Compute tensor-product quadrature weights

  @param[in]  basis    The TensorBasis to apply (uses dim, Q_1d, q_weight_1d)
  @param[in]  num_elem Number of elements batched together
  @param[in]  t_mode   Must be ContractMode::NoTranspose
  @param[out] v        Resized to Q_1d^dim * num_elem; v[q * num_elem + e] = prod_d q_weight_1d[q_d]

  @ref libCEED's CeedBasisApplyCore_Ref, CEED_EVAL_WEIGHT case
       (libCEED/backends/ref/ceed-ref-basis.c)
**/
void tensor_basis_apply_weight(const TensorBasis& basis, int num_elem, ContractMode t_mode,
                               std::vector<double>& v) {
  int dim = basis.dim;
  int Q = basis.Q_1d;

  if (t_mode == ContractMode::Transpose) {
    throw std::invalid_argument("EvalMode::Weight is incompatible with ContractMode::Transpose");
  }

  v.resize(num_elem * int_pow(Q, dim));

  for (int d = 0; d < dim; d++) {
    int pre = int_pow(Q, dim - d - 1), post = int_pow(Q, d);

    for (int i = 0; i < pre; i++) {
      for (int j = 0; j < Q; j++) {
        for (int k = 0; k < post; k++) {
          const double w = basis.q_weight_1d[j] * (d == 0 ? 1 : v[((i * Q + j) * post + k) * num_elem]);
          for (int e = 0; e < num_elem; e++) v[((i * Q + j) * post + k) * num_elem + e] = w;
        }
      }
    }
  }
}

/**
  @brief Dispatch to tensor_basis_apply_interp, _grad or _weight by eval_mode

  @param[in]  basis     The TensorBasis to apply
  @param[in]  num_elem  Number of elements batched together
  @param[in]  eval_mode EvalMode::Interp or EvalMode::Grad or EvalMode::Weight
  @param[in]  t_mode    ContractMode::NoTranspose or ContractMode::Transpose
  @param[in]  u         Input (nodal values, or quadrature-point values for Transpose);
                        ignored for EvalMode::Weight
  @param[out] v         Result

  @ref libCEED's CeedBasisApply (interface/ceed-basis.c)
**/
void tensor_basis_apply(const TensorBasis& basis, int num_elem, ContractMode t_mode, EvalMode eval_mode,
                         const std::vector<double>& u, std::vector<double>& v) {
  switch (eval_mode) {
    case EvalMode::Interp:
      tensor_basis_apply_interp(basis, num_elem, t_mode, u, v);
      break;
    case EvalMode::Grad:
      tensor_basis_apply_grad(basis, num_elem, t_mode, u, v);
      break;
    case EvalMode::Weight:
      tensor_basis_apply_weight(basis, num_elem, t_mode, v);
      break;
    default:
      throw std::invalid_argument("tensor_basis_apply: unknown eval_mode");
  }
}

/**
  @brief Transpose basis evaluation, added into v: v += B^T u (Interp or Grad)

  Lets several output fields that share one E-vector (e.g. an Interp and a Grad
  output of the same field) sum in element space before a single scatter.

  @param[in]     basis     The TensorBasis to apply
  @param[in]     num_elem  Number of elements batched together
  @param[in]     t_mode    Must be ContractMode::Transpose
  @param[in]     eval_mode EvalMode::Interp or EvalMode::Grad
  @param[in]     u         Quadrature-point values (Grad: dim blocks)
  @param[in,out] v         Nodal values, num_comp * P_1d^dim * num_elem entries; added into

  @ref libCEED's CeedBasisApplyAdd (interface/ceed-basis.c)
**/
void tensor_basis_apply_add(const TensorBasis& basis, int num_elem, ContractMode t_mode, EvalMode eval_mode,
                            const std::vector<double>& u, std::vector<double>& v) {
  if (t_mode != ContractMode::Transpose) {
    throw std::invalid_argument("tensor_basis_apply_add: only ContractMode::Transpose is supported");
  }
  const size_t v_size = static_cast<size_t>(basis.num_comp) * int_pow(basis.P_1d, basis.dim) * num_elem;
  if (v.size() != v_size) {
    throw std::invalid_argument("tensor_basis_apply_add: v must already have num_comp * P_1d^dim * num_elem entries");
  }
  switch (eval_mode) {
    case EvalMode::Interp:
      basis_apply_interp_core(basis, true, num_elem, t_mode, u, v);
      break;
    case EvalMode::Grad:
      basis_apply_grad_core(basis, true, num_elem, t_mode, u, v);
      break;
    default:
      throw std::invalid_argument("tensor_basis_apply_add: only EvalMode::Interp and EvalMode::Grad");
  }
}

}  // namespace fem
