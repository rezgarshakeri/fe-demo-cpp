#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "basis.hpp"

using Catch::Approx;

// ---------------------------------------------------------------------
// gauss_legendre_quadrature
// ---------------------------------------------------------------------

TEST_CASE("Gauss-Legendre weights sum to the length of [-1, 1]", "[basis][quadrature]") {
  for (int Q = 1; Q <= 5; ++Q) {
    std::vector<double> q_ref_1d, q_weight_1d;
    fem::gauss_legendre_quadrature(Q, q_ref_1d, q_weight_1d);

    REQUIRE(q_ref_1d.size() == static_cast<size_t>(Q));
    REQUIRE(q_weight_1d.size() == static_cast<size_t>(Q));

    double sum_w = 0.0;
    for (double w : q_weight_1d) sum_w += w;
    REQUIRE(sum_w == Approx(2.0).margin(1e-12));
  }
}

TEST_CASE("Gauss-Legendre points lie inside [-1, 1] and are sorted", "[basis][quadrature]") {
  for (int Q = 1; Q <= 5; ++Q) {
    std::vector<double> q_ref_1d, q_weight_1d;
    fem::gauss_legendre_quadrature(Q, q_ref_1d, q_weight_1d);

    for (int i = 0; i < Q; ++i) {
      REQUIRE(q_ref_1d[i] >= -1.0);
      REQUIRE(q_ref_1d[i] <= 1.0);
      if (i > 0) REQUIRE(q_ref_1d[i] > q_ref_1d[i - 1]);
    }
  }
}

// A Q-point Gauss-Legendre rule is exact for polynomials up to degree 2Q-1.
// integral_{-1}^{1} x^k dx = 0 (k odd), 2/(k+1) (k even).
TEST_CASE("Gauss-Legendre quadrature is exact for polynomials up to degree 2Q-1",
          "[basis][quadrature]") {
  for (int Q = 1; Q <= 5; ++Q) {
    std::vector<double> q_ref_1d, q_weight_1d;
    fem::gauss_legendre_quadrature(Q, q_ref_1d, q_weight_1d);

    const int max_degree = 2 * Q - 1;
    for (int k = 0; k <= max_degree; ++k) {
      double integral = 0.0;
      for (int i = 0; i < Q; ++i) {
        integral += q_weight_1d[i] * std::pow(q_ref_1d[i], k);
      }
      const double exact = (k % 2 == 0) ? 2.0 / (k + 1) : 0.0;
      REQUIRE(integral == Approx(exact).margin(1e-10));
    }
  }
}

// ---------------------------------------------------------------------
// gauss_lobatto_points
// ---------------------------------------------------------------------

TEST_CASE("Gauss-Lobatto points include the endpoints and are sorted", "[basis][quadrature]") {
  for (int P = 2; P <= 5; ++P) {
    std::vector<double> points;
    fem::gauss_lobatto_points(P, points);

    REQUIRE(points.size() == static_cast<size_t>(P));
    REQUIRE(points.front() == Approx(-1.0).margin(1e-12));
    REQUIRE(points.back() == Approx(1.0).margin(1e-12));
    for (size_t i = 1; i < points.size(); ++i) {
      REQUIRE(points[i] > points[i - 1]);
    }
  }
}

TEST_CASE("Gauss-Lobatto weights sum to the length of [-1, 1]", "[basis][quadrature]") {
  for (int P = 2; P <= 5; ++P) {
    std::vector<double> points, weights;
    fem::gauss_lobatto_points(P, points, &weights);

    REQUIRE(weights.size() == static_cast<size_t>(P));
    double sum_w = 0.0;
    for (double w : weights) sum_w += w;
    REQUIRE(sum_w == Approx(2.0).margin(1e-12));
  }
}

TEST_CASE("Gauss-Lobatto points for P=2 are exactly the endpoints", "[basis][quadrature]") {
  std::vector<double> points;
  fem::gauss_lobatto_points(2, points);
  REQUIRE(points[0] == Approx(-1.0).margin(1e-12));
  REQUIRE(points[1] == Approx(1.0).margin(1e-12));
}

TEST_CASE("Gauss-Lobatto points for P=3 are {-1, 0, 1}", "[basis][quadrature]") {
  std::vector<double> points;
  fem::gauss_lobatto_points(3, points);
  REQUIRE(points[0] == Approx(-1.0).margin(1e-12));
  REQUIRE(points[1] == Approx(0.0).margin(1e-12));
  REQUIRE(points[2] == Approx(1.0).margin(1e-12));
}

// ---------------------------------------------------------------------
// lagrange_basis_matrix
// ---------------------------------------------------------------------

TEST_CASE("Lagrange basis has the Kronecker delta property at its own nodes",
          "[basis][lagrange]") {
  const std::vector<double> nodes = {-1.0, 0.0, 1.0};
  const size_t P = nodes.size();
  std::vector<double> interp, grad;

  // Evaluate at the nodes themselves: eval_points == nodes.
  fem::lagrange_basis_matrix(nodes, nodes, interp, grad);
  REQUIRE(interp.size() == P * P);

  for (size_t i = 0; i < P; ++i) {       // eval point i == nodes[i]
    for (size_t j = 0; j < P; ++j) {     // basis function j
      const double expected = (i == j) ? 1.0 : 0.0;
      REQUIRE(interp[i * P + j] == Approx(expected).margin(1e-10));
    }
  }
}

TEST_CASE("Lagrange basis functions form a partition of unity", "[basis][lagrange]") {
  const std::vector<double> nodes = {-1.0, -0.3, 0.4, 1.0};
  const std::vector<double> eval_points = {-1.0, -0.6, 0.0, 0.55, 1.0};
  const size_t P = nodes.size();
  std::vector<double> interp, grad;

  fem::lagrange_basis_matrix(nodes, eval_points, interp, grad);
  REQUIRE(interp.size() == eval_points.size() * P);

  for (size_t i = 0; i < eval_points.size(); ++i) {
    double sum_interp = 0.0, sum_grad = 0.0;
    for (size_t j = 0; j < P; ++j) {
      sum_interp += interp[i * P + j];
      sum_grad += grad[i * P + j];
    }
    REQUIRE(sum_interp == Approx(1.0).margin(1e-10));
    REQUIRE(sum_grad == Approx(0.0).margin(1e-10));  // d/dx of a constant
  }
}

TEST_CASE("Lagrange basis on two nodes matches the analytic linear shape functions",
          "[basis][lagrange]") {
  const std::vector<double> nodes = {-1.0, 1.0};
  const std::vector<double> eval_points = {-1.0, -0.5, 0.0, 0.5, 1.0};
  const size_t P = nodes.size();
  std::vector<double> interp, grad;

  fem::lagrange_basis_matrix(nodes, eval_points, interp, grad);

  for (size_t i = 0; i < eval_points.size(); ++i) {
    const double x = eval_points[i];
    REQUIRE(interp[i * P + 0] == Approx((1.0 - x) / 2.0).margin(1e-12));
    REQUIRE(interp[i * P + 1] == Approx((1.0 + x) / 2.0).margin(1e-12));
    REQUIRE(grad[i * P + 0] == Approx(-0.5).margin(1e-12));
    REQUIRE(grad[i * P + 1] == Approx(0.5).margin(1e-12));
  }
}

// ---------------------------------------------------------------------
// TensorBasis::create_tensor_H1_lagrange
//
// The real target: a dim=1, P_1d=2 (linear) tensor basis should
// reproduce the exact analytic 1D Poisson stiffness that used to be
// hand-coded in matrix-free operator
//   Ke = (E/h) * [[1, -1], [-1, 1]]
// The reference-element (h=2, E=1) stiffness is:
//   K_ref[i][j] = sum_q weight[q] * grad_1d[q][i] * grad_1d[q][j]
//              == [[0.5, -0.5], [-0.5, 0.5]]
// since physical K = (2/h) * K_ref (Jacobian scaling for a length-h
// element mapped from the length-2 reference element).
// ---------------------------------------------------------------------

TEST_CASE("TensorBasis linear 1D basis reproduces the analytic reference stiffness",
          "[basis][tensor]") {
  const int dim = 1, num_comp = 1, P_1d = 2, Q_1d = 2;
  fem::TensorBasis basis = fem::TensorBasis::create_tensor_H1_lagrange(dim, num_comp, P_1d, Q_1d);

  REQUIRE(basis.dim == dim);
  REQUIRE(basis.num_comp == num_comp);
  REQUIRE(basis.P_1d == P_1d);
  REQUIRE(basis.Q_1d == Q_1d);
  REQUIRE(basis.q_ref_1d.size() == static_cast<size_t>(Q_1d));
  REQUIRE(basis.q_weight_1d.size() == static_cast<size_t>(Q_1d));
  REQUIRE(basis.interp_1d.size() == static_cast<size_t>(Q_1d * P_1d));
  REQUIRE(basis.grad_1d.size() == static_cast<size_t>(Q_1d * P_1d));

  double K_ref[2][2] = {{0.0, 0.0}, {0.0, 0.0}};
  for (int q = 0; q < Q_1d; ++q) {
    for (int i = 0; i < P_1d; ++i) {
      for (int j = 0; j < P_1d; ++j) {
        K_ref[i][j] += basis.q_weight_1d[q] * basis.grad_1d[q * P_1d + i] *
                       basis.grad_1d[q * P_1d + j];
      }
    }
  }

  REQUIRE(K_ref[0][0] == Approx(0.5).margin(1e-10));
  REQUIRE(K_ref[0][1] == Approx(-0.5).margin(1e-10));
  REQUIRE(K_ref[1][0] == Approx(-0.5).margin(1e-10));
  REQUIRE(K_ref[1][1] == Approx(0.5).margin(1e-10));
}
