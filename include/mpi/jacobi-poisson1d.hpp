#pragma once
#include "mpi/mpi-preconditioner.hpp"
#include "mesh.hpp"
#include "mpi/dist1d.hpp"
#include <vector>

namespace fem1d::mpi {

class JacobiPoisson1D final : public Preconditioner {
public:
  JacobiPoisson1D(const Mesh1D& mesh, const MpiVector& layout, double E);
  void apply(const MpiVector& r, MpiVector& z) const override;

private:
  const Dist1D* dist_;
  std::vector<double> inv_diag_owned_;
};

} // namespace fem1d::mpi
