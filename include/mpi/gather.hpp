#pragma once

#ifdef MINIFEM_USE_MPI
  #include <mpi.h>
#else
  #error "MINIFEM_USE_MPI not defined."
#endif

#include "mpi/mpi-vector.hpp"
#include <vector>

namespace fem1d::mpi {

// Gather owned portions of v to root into global_out (size = N_global).
// On non-root ranks, global_out is left unchanged.
void gather_to_root_global(const MpiVector& v, std::vector<double>& global_out, int root = 0);

} // namespace fem1d::mpi
