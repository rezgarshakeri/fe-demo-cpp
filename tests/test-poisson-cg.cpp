#include <catch2/catch_test_macros.hpp>

#include "mesh.hpp"
#include "assembled-operator.hpp"
#include "matrix-free-operator.hpp"
#include "linear-solver.hpp"

#include <cmath>
#include <vector>

static double l2norm(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

TEST_CASE("Assembled and matrix-free apply match", "[operator]") {
  auto mesh = fem1d::Mesh1D::build_uniform(100, 0.0, 1.0);
  const double E = 2.0;

  fem1d::AssembledPoisson1D A(mesh, E);
  fem1d::MatrixFreePoisson1D M(mesh, E);

  std::vector<double> x(mesh.nodes.size());
  for (size_t i = 0; i < x.size(); ++i) x[i] = 0.1 + 0.01 * static_cast<double>(i);

  std::vector<double> yA, yM;
  A.apply(x, yA);
  M.apply(x, yM);

  REQUIRE(yA.size() == yM.size());

  double err2 = 0.0;
  for (size_t i = 0; i < yA.size(); ++i) {
    double d = yA[i] - yM[i];
    err2 += d * d;
  }

  REQUIRE(err2 < 1e-12);
}

TEST_CASE("CG converges on 1D Poisson with Dirichlet BCs", "[cg]") {
  auto mesh = fem1d::Mesh1D::build_uniform(200, 0.0, 1.0);
  const int n = static_cast<int>(mesh.nodes.size());
  const double E = 1.0;

  fem1d::MatrixFreePoisson1D A(mesh, E);

  // RHS: constant f=1, lumped scaling b ~ h, with Dirichlet endpoints
  const double h = mesh.nodes[1] - mesh.nodes[0];
  std::vector<double> b(n, h);
  b.front() = 0.0;
  b.back()  = 0.0;

  std::vector<double> x(n, 0.0);

  fem1d::CGOptions opts;
  opts.max_iters = 2000;
  opts.rtol = 1e-10;

  auto res = fem1d::conjugate_gradient(A, b, x, opts);

  REQUIRE(res.final_res_norm < 1e-10 * l2norm(b));
  REQUIRE(res.iters < opts.max_iters);
}
