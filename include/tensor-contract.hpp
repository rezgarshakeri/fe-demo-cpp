#pragma once
#include <vector>

namespace fem {

enum class ContractMode { NoTranspose, Transpose };

void tensor_contract_apply(int A, int B, int C, int J,
                            const std::vector<double>& t,
                            ContractMode mode, bool add,
                            const std::vector<double>& u,
                            std::vector<double>& v);

} // namespace fem
