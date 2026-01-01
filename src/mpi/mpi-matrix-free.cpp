#include "mpi/mpi-matrix-free.hpp"
#include "mpi/halo.hpp"
#include "mpi/timer.hpp"

#include <cassert>

namespace fem1d::mpi {
extern TimerDB g_timers;

MpiMatrixFreePoisson1D::MpiMatrixFreePoisson1D(const Mesh1D& mesh, const Dist1D& dist, double E)
  : mesh_(mesh), dist_(dist), E_(E) {}

void MpiMatrixFreePoisson1D::apply(MpiVector& x, MpiVector& y) const {
  assert(x.dist == &dist_);
  assert(y.dist == &dist_);

  // Update ghost values so we can safely read x at interface neighbors
  ScopedTimer t(g_timers, "apply_total");
  halo_update(x);

  // Zero y (owned + ghosts storage; we will only write owned entries)
  y.data.assign(y.n_local, 0.0);

  // We want to reproduce the serial loop:
  // for e=0..Ne-1:
  //  i0=e, i1=e+1
  //  y[i0]+=c(u0-u1), y[i1]+=c(-u0+u1)
  //
  // In MPI, we loop over elements that can contribute to our owned rows.
  // With P1, owned rows [g0_owned..g1_owned] are influenced by elements:
  //   e in [g0_owned-1 .. g1_owned] (clamped)
  const int Ne = mesh_.num_elements;
  if (y.n_owned == 0) {
    return;
  }

  int e_begin = y.g0_owned - 1;
  int e_end   = y.g1_owned;     // inclusive
  if (e_begin < 0) e_begin = 0;
  if (e_end > Ne - 1) e_end = Ne - 1;
  {
    ScopedTimer t2(g_timers, "apply_element_loop");
    for (int e = e_begin; e <= e_end; ++e) {
      const int i0g = e;
      const int i1g = e + 1;
  
      // Map global nodes to local indices in x/y storage
      const int i0l = x.local_index_from_global(i0g);
      const int i1l = x.local_index_from_global(i1g);
      // i0l/i1l must exist either owned or ghost for the stencil to be computable
      assert(i0l >= 0 && i1l >= 0);
  
      const double h = mesh_.nodes[i1g] - mesh_.nodes[i0g];
      const double c = E_ / h;
  
      const double u0 = x.data[i0l];
      const double u1 = x.data[i1l];
  
      // Scatter to y only if the row is owned on this rank
      if (i0g >= y.g0_owned && i0g <= y.g1_owned) {
        const int row = i0g - y.g0_owned; // owned local
        y.data[row] += c * ( u0 - u1);
      }
      if (i1g >= y.g0_owned && i1g <= y.g1_owned) {
        const int row = i1g - y.g0_owned;
        y.data[row] += c * (-u0 + u1);
      }
    }
  }

  // Dirichlet rows as identity: y[0]=x[0], y[N]=x[N]
  {
    ScopedTimer t3(g_timers, "apply_dirichlet");
    const int Nlast = dist_.N_global - 1;
    if (y.g0_owned == 0) {
      // global 0 is owned here, so it's owned local 0
      y.data[0] = x.data[x.local_index_from_global(0)];
    }
    if (y.g1_owned == Nlast) {
      // global Nlast is owned here, so local index is (Nlast - g0_owned)
      const int row = Nlast - y.g0_owned;
      y.data[row] = x.data[x.local_index_from_global(Nlast)];
    }
  }
}

} // namespace fem1d::mpi
