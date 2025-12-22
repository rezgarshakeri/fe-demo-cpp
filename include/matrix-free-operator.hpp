#pragma once
#include "mesh.hpp"
#include "operator.hpp"

namespace fem1d {

class MatrixFreePoisson1D : public Operator {
public:
    MatrixFreePoisson1D(const Mesh1D& mesh, double E);

    void apply(const std::vector<double>& x,
               std::vector<double>&       y) const override;

private:
    Mesh1D mesh_;
    double E_;
};

} // namespace fem1d
