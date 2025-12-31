#pragma once
#include <mpi.h>

#include "mpi/dist1d.hpp"
#include <vector>
#include <cassert>

namespace fem1d::mpi {

struct MpiVector {
  const Dist1D* dist = nullptr;

  // Global owned range [g0_owned, g1_owned] inclusive
  int g0_owned = 0;
  int g1_owned = -1;

  // Sizes
  int n_owned = 0;
  int n_ghost_left = 0;
  int n_ghost_right = 0;
  int n_local = 0;

  // Layout in data:
  // owned: [0, n_owned)
  // left ghosts:  [left_ghost_begin, left_ghost_begin + n_ghost_left)
  // right ghosts: [right_ghost_begin, right_ghost_begin + n_ghost_right)
  int left_ghost_begin = -1;
  int right_ghost_begin = -1;

  std::vector<double> data;

  MpiVector() = default;

  explicit MpiVector(const Dist1D& d, double init_owned = 0.0, double init_ghost = 0.0) {
    reset(d, init_owned, init_ghost);
  }

  void reset(const Dist1D& d, double init_owned = 0.0, double init_ghost = 0.0) {
    dist = &d;

    // Dist1D gives the geometric node span implied by element partition: [d.i0, d.i1]
    // Enforce unique ownership:
    // - rank 0 owns [i0..i1]
    // - rank>0 owns [i0+1..i1] so the shared left interface node is owned by the left rank.
    if (d.n_owned == 0) {
      g0_owned = 0;
      g1_owned = -1;
      n_owned = 0;
    } else {
      g0_owned = d.i0 + ((d.rank > 0) ? 1 : 0);
      g1_owned = d.i1;
      if (g0_owned > g1_owned) { // can happen if rank owns 1 node geometrically and we drop it
        n_owned = 0;
      } else {
        n_owned = g1_owned - g0_owned + 1;
      }
    }

    // Ghost layers based on ghost_width, clamped to global bounds
    const int gw = d.ghost_width;
    const int Nlast = d.N_global - 1;

    n_ghost_left  = (n_owned > 0) ? std::min(gw, g0_owned - 0) : 0;
    n_ghost_right = (n_owned > 0) ? std::min(gw, Nlast - g1_owned) : 0;

    left_ghost_begin  = n_owned;
    right_ghost_begin = n_owned + n_ghost_left;

    n_local = n_owned + n_ghost_left + n_ghost_right;
    data.assign(n_local, init_ghost);
    for (int i = 0; i < n_owned; ++i) data[i] = init_owned;
  }

  // Owned access
  double& owned(int i) {
    assert(i >= 0 && i < n_owned);
    return data[i];
  }
  const double& owned(int i) const {
    assert(i >= 0 && i < n_owned);
    return data[i];
  }

  // Ghost pointers
  double* left_ghost_ptr()  { return (n_ghost_left  > 0) ? data.data() + left_ghost_begin  : nullptr; }
  double* right_ghost_ptr() { return (n_ghost_right > 0) ? data.data() + right_ghost_begin : nullptr; }
  const double* left_ghost_ptr()  const { return (n_ghost_left  > 0) ? data.data() + left_ghost_begin  : nullptr; }
  const double* right_ghost_ptr() const { return (n_ghost_right > 0) ? data.data() + right_ghost_begin : nullptr; }

  // Boundary owned blocks (contiguous)
  const double* left_send_ptr() const { return (n_owned > 0) ? data.data() : nullptr; }
  const double* right_send_ptr() const { return (n_owned > 0) ? data.data() + (n_owned - n_ghost_right) : nullptr; }

  // Global index of owned local i
  int global_owned(int i) const { return g0_owned + i; }

  int local_index_from_global(int g) const {
    if (n_owned == 0) return -1;

    // Owned global range [g0_owned, g1_owned]
    if (g >= g0_owned && g <= g1_owned) return g - g0_owned;
  
    // Left ghosts: [g0_owned - n_ghost_left, g0_owned - 1]
    const int gl0 = g0_owned - n_ghost_left;
    const int gl1 = g0_owned - 1;
    if (n_ghost_left > 0 && g >= gl0 && g <= gl1) {
      return left_ghost_begin + (g - gl0);
    }
  
    // Right ghosts: [g1_owned + 1, g1_owned + n_ghost_right]
    const int gr0 = g1_owned + 1;
    const int gr1 = g1_owned + n_ghost_right;
    if (n_ghost_right > 0 && g >= gr0 && g <= gr1) {
      return right_ghost_begin + (g - gr0);
    }
  
    return -1;
  }
};

} // namespace fem1d::mpi
