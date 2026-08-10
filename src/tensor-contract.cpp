#include "tensor-contract.hpp"
#include <stdexcept>
#include <algorithm>

namespace fem {

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

} // namespace fem
