#pragma once
#include "mpi/mpi-operator.hpp"
#include "mesh.hpp"

namespace fem1d::mpi {

class MpiMatrixFreePoisson1D : public MpiOperator {
public:
  MpiMatrixFreePoisson1D(const Mesh1D& mesh, const Dist1D& dist, double E);

  void apply(MpiVector& x, MpiVector& y) const override;

private:
  const Mesh1D& mesh_;
  const Dist1D& dist_;
  double E_;
};

} // namespace fem1d::mpi
