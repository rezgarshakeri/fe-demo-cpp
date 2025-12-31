#pragma once
#include <mpi.h>

namespace fem1d::mpi {

struct Dist1D {
  MPI_Comm comm = MPI_COMM_NULL;
  int rank = 0;
  int size = 1;

  // Global mesh sizes
  int Ne_global = 0;  // number of elements
  int N_global  = 0;  // number of nodes = Ne_global + 1

  // Ghost width (points); P1 -> 1, P2 -> 2, ...
  int ghost_width = 1;

  // Owned elements [e0, e1)
  int e0 = 0;
  int e1 = 0;
  int Ne_owned = 0;

  // Owned nodes [i0, i1] (global node indices)
  int i0 = 0;
  int i1 = 0;
  int n_owned = 0;

  // Neighbor ranks
  int left_rank  = MPI_PROC_NULL;
  int right_rank = MPI_PROC_NULL;

  // Ghost counts
  int n_ghost_left  = 0;
  int n_ghost_right = 0;
  int n_local = 0; // owned + ghosts

  // Local indices of ghost blocks in local storage:
  // owned: [0, n_owned)
  // left ghosts:  [left_ghost_begin, left_ghost_begin + n_ghost_left)
  // right ghosts: [right_ghost_begin, right_ghost_begin + n_ghost_right)
  int left_ghost_begin  = -1;
  int right_ghost_begin = -1;

  // Factory
  static Dist1D create(MPI_Comm comm_in, int Ne_global_in, int ghost_width_in);

  // Queries
  bool owns_global_node(int g) const { return (g >= i0 && g <= i1); }

  // Owned local <-> global
  int global_from_owned_lid(int lid) const { return i0 + lid; }
  int owned_lid_from_global(int g) const { return g - i0; } // only valid if owns_global_node(g)

  // Map a global node index to local storage index (owned or ghost), else -1
  int local_index_from_global(int g) const;

  // Convenience: global indices for ghost nodes (if present)
  int left_ghost_global(int k) const;   // k=0..n_ghost_left-1 (closest-first)
  int right_ghost_global(int k) const;  // k=0..n_ghost_right-1
};

} // namespace fem1d::mpi
