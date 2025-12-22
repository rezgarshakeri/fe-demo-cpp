#pragma once
#include "mesh.hpp"
#include "operator.hpp"
#include <vector>

namespace fem1d {

// super simple dense/CSR for now
struct CSRMatrix {
    int n;
    std::vector<int> rowptr;
    std::vector<int> colind;
    std::vector<double> values;

    void matvec(const std::vector<double>& x,
                std::vector<double>&       y) const;
};

class AssembledPoisson1D : public Operator {
public:
    AssembledPoisson1D(const Mesh1D& mesh, double E);

    void apply(const std::vector<double>& x,
               std::vector<double>&       y) const override;

    bool can_assemble() const override { return true; }

private:
    CSRMatrix A_;
};

} // namespace fem1d
