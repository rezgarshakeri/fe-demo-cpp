#include "mpi/halo.hpp"
#include <algorithm>

namespace fem1d::mpi {

void halo_update(MpiVector& x) {
  const Dist1D& d = *x.dist;

  // Nothing to do if no ghosts
  if (x.n_ghost_left == 0 && x.n_ghost_right == 0) return;

  // Neighbor ranks
  const int left  = d.left_rank;
  const int right = d.right_rank;

  // We exchange blocks of size = number of ghost entries on each side.
  // Convention:
  // - To fill my LEFT ghosts, I receive from LEFT neighbor's RIGHT boundary owned block.
  // - To fill my RIGHT ghosts, I receive from RIGHT neighbor's LEFT boundary owned block.

  MPI_Status st;

  // Exchange with left neighbor (my left ghosts <-> their right boundary)
  if (x.n_ghost_left > 0) {
    // Receive my left ghosts from left neighbor
    // Send my left boundary owned block to left neighbor (they use as right ghosts)
    MPI_Sendrecv(
      /*sendbuf*/ (x.n_owned > 0 ? x.left_send_ptr() : nullptr),
      /*sendcount*/ x.n_ghost_left,
      /*sendtype*/ MPI_DOUBLE,
      /*dest*/ left,
      /*sendtag*/ 100,
      /*recvbuf*/ x.left_ghost_ptr(),
      /*recvcount*/ x.n_ghost_left,
      /*recvtype*/ MPI_DOUBLE,
      /*source*/ left,
      /*recvtag*/ 200,
      d.comm,
      &st
    );
  } else {
    // Still send to left if they need right ghosts? In 1D with our ownership, boundary ranks won't.
    // Keep it simple: do nothing.
  }

  // Exchange with right neighbor (my right ghosts <-> their left boundary)
  if (x.n_ghost_right > 0) {
    // Receive my right ghosts from right neighbor
    // Send my right boundary owned block to right neighbor (they use as left ghosts)
    const double* right_send = (x.n_owned > 0) ? (x.data.data() + (x.n_owned - x.n_ghost_right)) : nullptr;

    MPI_Sendrecv(
      /*sendbuf*/ right_send,
      /*sendcount*/ x.n_ghost_right,
      /*sendtype*/ MPI_DOUBLE,
      /*dest*/ right,
      /*sendtag*/ 200,
      /*recvbuf*/ x.right_ghost_ptr(),
      /*recvcount*/ x.n_ghost_right,
      /*recvtype*/ MPI_DOUBLE,
      /*source*/ right,
      /*recvtag*/ 100,
      d.comm,
      &st
    );
  }
}

} // namespace fem1d::mpi
