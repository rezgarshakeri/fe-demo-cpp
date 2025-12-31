#pragma once
#include <mpi.h>
#include "mpi/mpi-vector.hpp"

namespace fem1d::mpi {

// Update ghost entries of x using nearest-neighbor exchange (block width = ghost_width, clamped).
void halo_update(MpiVector& x);

} // namespace fem1d::mpi
