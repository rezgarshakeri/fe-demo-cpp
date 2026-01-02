#include "mpi/jacobi-poisson1d.hpp"
#include <cassert>

namespace fem1d::mpi {

JacobiPoisson1D::JacobiPoisson1D(const Mesh1D& mesh, const MpiVector& layout, double E)
  : dist_(layout.dist) {

  const int N  = dist_->N_global;
  const int g0 = layout.g0_owned;
  const int g1 = layout.g1_owned;
  const int n_owned = layout.n_owned;

  inv_diag_owned_.assign(n_owned, 0.0);

  for (int gi = g0; gi <= g1; ++gi) {
    double diag = 0.0;

    if (gi == 0 || gi == N - 1) {
      diag = 1.0; // Dirichlet identity row
    } else {
      const double hL = mesh.nodes[gi]     - mesh.nodes[gi - 1];
      const double hR = mesh.nodes[gi + 1] - mesh.nodes[gi];
      diag = E * (1.0 / hL + 1.0 / hR);
    }

    inv_diag_owned_[gi - g0] = 1.0 / diag;
  }
}

void JacobiPoisson1D::apply(const MpiVector& r, MpiVector& z) const {
  assert(r.dist == dist_);
  assert(z.dist == dist_);
  assert(r.n_owned == (int)inv_diag_owned_.size());
  for (int i = 0; i < r.n_owned; ++i) z.data[i] = inv_diag_owned_[i] * r.data[i];
}

} // namespace fem1d::mpi
