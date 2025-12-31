#pragma once

#include "mpi/mpi-operator.hpp"
#include "mpi/mpi-vector.hpp"
#include "linear-solver.hpp"   // reuse CGOptions/CGResult

namespace fem1d::mpi {

// MPI CG: solves A x = b for distributed vectors
CGResult conjugate_gradient(const MpiOperator& A,
                            const MpiVector&  b,
                            MpiVector&        x,
                            const CGOptions&  opts = {});

} // namespace fem1d::mpi
