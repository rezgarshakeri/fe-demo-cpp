#pragma once
#include "mpi/mpi-vector.hpp"

namespace fem1d::mpi {

class MpiOperator {
public:
  virtual ~MpiOperator() = default;

  // y = A x  (both distributed vectors)
  virtual void apply(MpiVector& x, MpiVector& y) const = 0;
};

} // namespace fem1d::mpi
