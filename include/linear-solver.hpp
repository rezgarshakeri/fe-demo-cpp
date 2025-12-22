#pragma once
#include "operator.hpp"
#include <vector>
#include <functional>

namespace fem1d {

struct CGOptions {
    int    max_iters = 1000;
    double rtol      = 1e-8;
};

struct CGResult {
    int    iters;
    double final_res_norm;
};

CGResult conjugate_gradient(const Operator& A,
                            const std::vector<double>& b,
                            std::vector<double>&       x,
                            const CGOptions&           opts = {});

} // namespace fem1d
