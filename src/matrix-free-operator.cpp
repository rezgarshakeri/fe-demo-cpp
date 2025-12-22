#include "matrix-free-operator.hpp"
#include <cassert>

namespace fem1d {

MatrixFreePoisson1D::MatrixFreePoisson1D(const Mesh1D& mesh, double E)
  : mesh_(mesh), E_(E) {}

void MatrixFreePoisson1D::apply(const std::vector<double>& x,
                                std::vector<double>& y) const {
  const int n = static_cast<int>(mesh_.nodes.size());
  assert(static_cast<int>(x.size()) == n);

  y.assign(n, 0.0);

  // Linear 1D Poisson stiffness per element:
  // Ke = (E/h) * [ 1 -1; -1  1 ]
  for (int e = 0; e < mesh_.num_elements; ++e) {
    const int i0 = mesh_.elem_nodes[2 * e + 0];
    const int i1 = mesh_.elem_nodes[2 * e + 1];

    const double h = mesh_.nodes[i1] - mesh_.nodes[i0];
    const double c = E_ / h;

    const double u0 = x[i0];
    const double u1 = x[i1];

    y[i0] += c * ( u0 - u1);
    y[i1] += c * (-u0 + u1);
  }

  // Homogeneous Dirichlet BC rows as identity (simple start)
  y[0]     = x[0];
  y[n - 1] = x[n - 1];
}

} // namespace fem1d
