#include "mpi/halo.hpp"
#include "mpi/timer.hpp"
#include <algorithm>

namespace fem1d::mpi {

extern TimerDB g_timers;   // declaration only (no fem1d::mpi:: prefix needed inside namespace)

void halo_update(MpiVector& x) {
  ScopedTimer tt(g_timers, "halo_update");   // ✅ inside function scope

  const Dist1D& d = *x.dist;

  if (x.n_ghost_left == 0 && x.n_ghost_right == 0) return;

  const int left  = d.left_rank;
  const int right = d.right_rank;

  MPI_Status st;

  if (x.n_ghost_left > 0) {
    MPI_Sendrecv(
      (x.n_owned > 0 ? x.left_send_ptr() : nullptr),
      x.n_ghost_left, MPI_DOUBLE, left, 100,
      x.left_ghost_ptr(),
      x.n_ghost_left, MPI_DOUBLE, left, 200,
      d.comm, &st
    );
  }

  if (x.n_ghost_right > 0) {
    const double* right_send = (x.n_owned > 0) ? (x.data.data() + (x.n_owned - x.n_ghost_right)) : nullptr;
    MPI_Sendrecv(
      right_send,
      x.n_ghost_right, MPI_DOUBLE, right, 200,
      x.right_ghost_ptr(),
      x.n_ghost_right, MPI_DOUBLE, right, 100,
      d.comm, &st
    );
  }
}

} // namespace fem1d::mpi
