#pragma once

#include "mpi/mpi-operator.hpp"
#include "mpi/mpi-vector.hpp"
#include "linear-solver.hpp"   // reuse CGOptions/CGResult
#include "mpi/mpi-preconditioner.hpp"

namespace fem1d::mpi {

// MPI CG: solves A x = b for distributed vectors
CGResult conjugate_gradient(const MpiOperator& A,
                            const MpiVector&  b,
                            MpiVector&        x,
                            const CGOptions&  opts = {});
// MPI PCG: with jacobi pc
CGResult pcg(const MpiOperator& A,
             const Preconditioner& M,
             const MpiVector& b,
             MpiVector& x,
             const CGOptions& opts = {});

} // namespace fem1d::mpi
