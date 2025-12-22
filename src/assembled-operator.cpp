#include "assembled-operator.hpp"
#include <cassert>
#include <algorithm>

namespace fem1d {

void CSRMatrix::matvec(const std::vector<double>& x,
                       std::vector<double>&       y) const {
  assert(static_cast<int>(x.size()) == n);
  y.assign(n, 0.0);

  for (int i = 0; i < n; ++i) {
    double sum = 0.0;
    for (int k = rowptr[i]; k < rowptr[i + 1]; ++k) {
      sum += values[k] * x[colind[k]];
    }
    y[i] = sum;
  }
}

AssembledPoisson1D::AssembledPoisson1D(const Mesh1D& mesh, double E) {
  const int n = static_cast<int>(mesh.nodes.size());
  A_.n = n;

  // We’ll build a tri-diagonal CSR for 1D linear Poisson with Dirichlet at ends.
  // Row i has:
  //   i=0     -> [1]
  //   i=n-1   -> [1]
  //   else    -> [-E/h, 2E/h, -E/h]
  //
  // For uniform mesh, h is constant, but we compute it from nodes for robustness.
  const double h = mesh.nodes[1] - mesh.nodes[0];
  const double c = E / h;

  A_.rowptr.resize(n + 1);

  // Count nnz and fill rowptr
  int nnz = 0;
  for (int i = 0; i < n; ++i) {
    A_.rowptr[i] = nnz;
    if (i == 0 || i == n - 1) {
      nnz += 1;
    } else {
      nnz += 3;
    }
  }
  A_.rowptr[n] = nnz;

  A_.colind.resize(nnz);
  A_.values.resize(nnz);

  // Fill CSR
  int p = 0;
  for (int i = 0; i < n; ++i) {
    if (i == 0 || i == n - 1) {
      A_.colind[p] = i;
      A_.values[p] = 1.0;
      ++p;
    } else {
      // left
      A_.colind[p] = i - 1;
      A_.values[p] = -c;
      ++p;

      // diag
      A_.colind[p] = i;
      A_.values[p] = 2.0 * c;
      ++p;

      // right
      A_.colind[p] = i + 1;
      A_.values[p] = -c;
      ++p;
    }
  }
  assert(p == nnz);
}

void AssembledPoisson1D::apply(const std::vector<double>& x,
                               std::vector<double>&       y) const {
  A_.matvec(x, y);
}

} // namespace fem1d
