#pragma once
#include "mpi/mpi-vector.hpp"

namespace fem1d::mpi {

class Preconditioner {
public:
  virtual ~Preconditioner() = default;

  // z = M^{-1} r
  virtual void apply(const MpiVector& r, MpiVector& z) const = 0;
};

} // namespace fem1d::mpi
