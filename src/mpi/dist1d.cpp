#include "mpi/dist1d.hpp"

#include <algorithm> // std::min, std::max

namespace fem1d::mpi {

static int clamp_int(int x, int lo, int hi) {
  return std::max(lo, std::min(x, hi));
}

Dist1D Dist1D::create(MPI_Comm comm_in, int Ne_global_in, int ghost_width_in) {
  Dist1D d;
  d.comm = comm_in;

  MPI_Comm_rank(d.comm, &d.rank);
  MPI_Comm_size(d.comm, &d.size);

  d.Ne_global   = Ne_global_in;
  d.N_global    = d.Ne_global + 1;
  d.ghost_width = std::max(0, ghost_width_in);

  // ----- Partition elements contiguously and as evenly as possible -----
  // q = base elements per rank, r = remainder
  const int q = d.Ne_global / d.size;
  const int r = d.Ne_global % d.size;

  // Ranks [0, r) get (q+1), ranks [r, size) get q
  const int my_count = (d.rank < r) ? (q + 1) : q;

  // e0 = sum of counts of previous ranks
  const int e0 = (d.rank < r) ? d.rank * (q + 1) : r * (q + 1) + (d.rank - r) * q;
  const int e1 = e0 + my_count;

  d.e0 = e0;
  d.e1 = e1;
  d.Ne_owned = my_count;

  // Owned nodes for 1D linear mesh: owning elements [e0,e1) implies owning nodes [e0,e1]
  // Special case: if a rank owns zero elements, it owns zero nodes.
  if (d.Ne_owned > 0) {
    d.i0 = d.e0;
    d.i1 = d.e1; // inclusive
    d.n_owned = d.i1 - d.i0 + 1;
  } else {
    d.i0 = 0;
    d.i1 = -1;
    d.n_owned = 0;
  }

  // Neighbors (for contiguous ranks)
  d.left_rank  = (d.rank > 0) ? (d.rank - 1) : MPI_PROC_NULL;
  d.right_rank = (d.rank < d.size - 1) ? (d.rank + 1) : MPI_PROC_NULL;

  // ----- Ghost counts (clamped to domain bounds) -----
  // Left ghosts are global nodes [i0 - gw, i0 - 1]
  // Right ghosts are global nodes [i1 + 1, i1 + gw]
  if (d.n_owned > 0) {
    d.n_ghost_left  = std::min(d.ghost_width, d.i0 - 0);
    d.n_ghost_right = std::min(d.ghost_width, (d.N_global - 1) - d.i1);
  } else {
    d.n_ghost_left = 0;
    d.n_ghost_right = 0;
  }

  d.left_ghost_begin  = d.n_owned;
  d.right_ghost_begin = d.n_owned + d.n_ghost_left;
  d.n_local = d.n_owned + d.n_ghost_left + d.n_ghost_right;

  return d;
}

int Dist1D::left_ghost_global(int k) const {
  // k = 0..n_ghost_left-1, closest-first: i0-1, i0-2, ...
  // We'll store ghosts in increasing global order for simple indexing:
  // left ghosts global range is [i0 - n_ghost_left, i0 - 1]
  // k=0 corresponds to (i0 - n_ghost_left)
  return (i0 - n_ghost_left) + k;
}

int Dist1D::right_ghost_global(int k) const {
  // right ghosts global range is [i1 + 1, i1 + n_ghost_right]
  return (i1 + 1) + k;
}

int Dist1D::local_index_from_global(int g) const {
  if (n_owned == 0) return -1;

  // Owned
  if (g >= i0 && g <= i1) return g - i0;

  // Left ghosts: [i0 - n_ghost_left, i0 - 1]
  const int gl0 = i0 - n_ghost_left;
  const int gl1 = i0 - 1;
  if (n_ghost_left > 0 && g >= gl0 && g <= gl1) {
    return left_ghost_begin + (g - gl0);
  }

  // Right ghosts: [i1 + 1, i1 + n_ghost_right]
  const int gr0 = i1 + 1;
  const int gr1 = i1 + n_ghost_right;
  if (n_ghost_right > 0 && g >= gr0 && g <= gr1) {
    return right_ghost_begin + (g - gr0);
  }

  return -1;
}

} // namespace fem1d::mpi
