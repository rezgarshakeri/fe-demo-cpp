#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include "basis.hpp"
#include "tensor-contract.hpp"

using Catch::Approx;

// ---------------------------------------------------------------------
// Plain matrix-vector product
// ---------------------------------------------------------------------
// A small, fixed J=2 x B=3 matrix reused across tests:
//   t = [[1, 2, 3],
//        [4, 5, 6]]
static const std::vector<double> kT = {1, 2, 3, 4, 5, 6};

TEST_CASE("tensor_contract_apply: plain matrix-vector product", "[tensor-contract]") {
  const size_t A = 1, B = 3, C = 1, J = 2;
  std::vector<double> u = {2, -1, 3};
  std::vector<double> v(J, 0.0);
  fem::tensor_contract_apply(A, B, C, J, kT, fem::ContractMode::NoTranspose, false, u, v);
  REQUIRE(v.size() == J);
  REQUIRE(v[0] == Approx(9.0));  // 1*2 + 2*-1 + 3*3
  REQUIRE(v[1] == Approx(21.0)); // 4*2 + 5*-1 + 6*3
}

TEST_CASE("tensor_contract_apply: batches independently over A", "[tensor-contract]") {
  const size_t A = 2, B = 3, C = 1, J = 2;
  std::vector<double> u = {2, -1, 3, 1, 1, 1};  // two batches of size B=3
  std::vector<double> v(J * A, 0.0);
  fem::tensor_contract_apply(A, B, C, J, kT, fem::ContractMode::NoTranspose, false, u, v);
  REQUIRE(v.size() == 4);
  // Batch 0: u = {2, -1, 3}
  REQUIRE(v[0] == Approx(9.0));  // 1*2 + 2*-1 + 3*3
  REQUIRE(v[1] == Approx(21.0)); // 4*2 + 5*-1 + 6*3
  // Batch 1: u = {1, 1, 1}
  REQUIRE(v[2] == Approx(6.0));
  REQUIRE(v[3] == Approx(15.0));
}

TEST_CASE("tensor_contract_apply: batches independently over C", "[tensor-contract]") {
  const size_t A = 1, B = 3, C = 2, J = 2;
  std::vector<double> u = {2, 1, -1, 1, 3, 1};  // channel 0 = {2,-1,3}, channel 1 = {1,1,1}
  std::vector<double> v(J * A * C, 0.0);
  fem::tensor_contract_apply(A, B, C, J, kT, fem::ContractMode::NoTranspose, false, u, v);
  REQUIRE(v.size() == 4);
  // v[j*2 + c]: for each j, channel 0 then channel 1 (interleaved, not
  // concatenated like the A-batching case above).
  REQUIRE(v[0] == Approx(9.0));  // j=0, c=0: 1*2 + 2*-1 + 3*3
  REQUIRE(v[1] == Approx(6.0));  // j=0, c=1: 1*1 + 2*1 + 3*1
  REQUIRE(v[2] == Approx(21.0)); // j=1, c=0: 4*2 + 5*-1 + 6*3
  REQUIRE(v[3] == Approx(15.0)); // j=1, c=1: 4*1 + 5*1 + 6*1
}

TEST_CASE("tensor_contract_apply: add=true accumulates instead of overwriting",
          "[tensor-contract]") {
  const size_t A = 1, B = 3, C = 1, J = 2;
  std::vector<double> u = {2, -1, 3};
  std::vector<double> v = {100.0, 200.0};
  fem::tensor_contract_apply(A, B, C, J, kT, fem::ContractMode::NoTranspose, true, u, v);
  REQUIRE(v[0] == Approx(109.0)); // 100 + 9
  REQUIRE(v[1] == Approx(221.0)); // 200 + 21
}

TEST_CASE("tensor_contract_apply: Transpose reuses the same data for the adjoint",
          "[tensor-contract]") {
  // Forward: v_fwd = t * u  (A=1, B=3, C=1, J=2)
  std::vector<double> u = {2, -1, 3};
  std::vector<double> v_fwd;
  fem::tensor_contract_apply(1, 3, 1, 2, kT, fem::ContractMode::NoTranspose, false, u, v_fwd);
  REQUIRE(v_fwd[0] == Approx(9.0));
  REQUIRE(v_fwd[1] == Approx(21.0));

  // Adjoint: w = t^T * v_fwd, computed via the *same* kT array by
  // swapping B/J at the call site (B=2 now, J=3 now) and using Transpose.
  std::vector<double> w;
  fem::tensor_contract_apply(1, 2, 1, 3, kT, fem::ContractMode::Transpose, false, v_fwd, w);

  REQUIRE(w.size() == 3);
  // t^T = [[1,4],[2,5],[3,6]]; t^T * v_fwd:
  REQUIRE(w[0] == Approx(1.0 * 9.0 + 4.0 * 21.0)); // 93
  REQUIRE(w[1] == Approx(2.0 * 9.0 + 5.0 * 21.0)); // 123
  REQUIRE(w[2] == Approx(3.0 * 9.0 + 6.0 * 21.0)); // 153
}

TEST_CASE("tensor_contract_apply matches a naive computation using real TensorBasis::interp_1d",
          "[tensor-contract][basis]") {
  const int P_1d = 3, Q_1d = 4;
  fem::TensorBasis basis = fem::TensorBasis::create_tensor_H1_lagrange(1, 1, P_1d, Q_1d);

  const std::vector<double> u = {1.5, -2.0, 0.5}; // arbitrary nodal values
  std::vector<double> v;
  fem::tensor_contract_apply(1, P_1d, 1, Q_1d, basis.interp_1d,
                              fem::ContractMode::NoTranspose, false, u, v);

  REQUIRE(v.size() == static_cast<size_t>(Q_1d));
  for (int j = 0; j < Q_1d; ++j) {
    double expected = 0.0;
    for (int i = 0; i < P_1d; ++i) expected += basis.interp_1d[j * P_1d + i] * u[i];
    REQUIRE(v[j] == Approx(expected).margin(1e-12));
  }
}

// ---------------------------------------------------------------------
// Manufactured-solution checks for tensor_basis_apply_interp/_grad:
// sample a smooth analytic function at the P_1d Lobatto nodes, apply
// the basis, and compare against the function (interp) or its exact
// gradient (grad) evaluated directly at the Q_1d Gauss quadrature
// points. High-order Lagrange interpolation of a smooth (entire)
// function converges spectrally. By P_1d=16 the error has already
// hit the floating-point round-off floor (~1e-13).
// ---------------------------------------------------------------------

TEST_CASE("tensor_basis_apply_interp matches an analytic function in 1D",
          "[tensor-contract][manufactured]") {
  const int P_1d = 16, Q_1d = 16;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(1, 1, P_1d, Q_1d);

  // u(x) = sin(x) + x^2, sampled at the Lobatto nodes.
  std::vector<double> u;
  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);
  for (int i = 0; i < P_1d; ++i) {
    double x = nodes[i];
    u.push_back(std::sin(x) + x * x);
  }

  std::vector<double> v;
  fem::tensor_basis_apply_interp(basis, 1, u, v);

  REQUIRE(v.size() == static_cast<size_t>(Q_1d));
  for (int j = 0; j < Q_1d; ++j) {
    double x = basis.q_ref_1d[j];
    double expected = std::sin(x) + x * x;
    REQUIRE(v[j] == Approx(expected).margin(1e-11));
  }
}

TEST_CASE("tensor_basis_apply_grad matches an analytic gradient in 1D",
          "[tensor-contract][manufactured]") {
  const int P_1d = 16, Q_1d = 16;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(1, 1, P_1d, Q_1d);

  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);

  // u(x) = sin(x) + x^2, u'(x) = cos(x) + 2x
  std::vector<double> u;
  for (int i = 0; i < P_1d; ++i) {
    double x = nodes[i];
    u.push_back(std::sin(x) + x * x);
  }

  std::vector<double> v;
  fem::tensor_basis_apply_grad(basis, 1, u, v);

  REQUIRE(v.size() == static_cast<size_t>(Q_1d));
  for (int j = 0; j < Q_1d; ++j) {
    double x = basis.q_ref_1d[j];
    double expected = std::cos(x) + 2.0 * x;
    REQUIRE(v[j] == Approx(expected).margin(1e-11));
  }
}

TEST_CASE("tensor_basis_apply_interp matches an analytic function in 2D",
          "[tensor-contract][manufactured]") {
  const int P_1d = 16, Q_1d = 16;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(2, 1, P_1d, Q_1d);

  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);

  // u(x,y) = sin(x) + cos(x*y) + x*y
  std::vector<double> u(P_1d * P_1d);
  for (int iy = 0; iy < P_1d; ++iy) {
    for (int ix = 0; ix < P_1d; ++ix) {
      double x = nodes[ix], y = nodes[iy];
      u[iy * P_1d + ix] = std::sin(x) + std::cos(x * y) + x * y;
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_interp(basis, 1, u, v);

  REQUIRE(v.size() == static_cast<size_t>(Q_1d * Q_1d));
  for (int jy = 0; jy < Q_1d; ++jy) {
    for (int jx = 0; jx < Q_1d; ++jx) {
      double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy];
      double expected = std::sin(x) + std::cos(x * y) + x * y;
      REQUIRE(v[jy * Q_1d + jx] == Approx(expected).margin(1e-11));
    }
  }
}

TEST_CASE("tensor_basis_apply_grad matches an analytic gradient in 2D",
          "[tensor-contract][manufactured]") {
  const int P_1d = 16, Q_1d = 16;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(2, 1, P_1d, Q_1d);

  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);

  // u(x,y) = sin(x) + cos(x*y) + x*y
  // du/dx = cos(x) - y*sin(x*y) + y
  // du/dy = -x*sin(x*y) + x
  std::vector<double> u(P_1d * P_1d);
  for (int iy = 0; iy < P_1d; ++iy) {
    for (int ix = 0; ix < P_1d; ++ix) {
      double x = nodes[ix], y = nodes[iy];
      u[iy * P_1d + ix] = std::sin(x) + std::cos(x * y) + x * y;
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_grad(basis, 1, u, v);

  const int Q2 = Q_1d * Q_1d;
  REQUIRE(v.size() == static_cast<size_t>(2 * Q2));
  for (int jy = 0; jy < Q_1d; ++jy) {
    for (int jx = 0; jx < Q_1d; ++jx) {
      double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy];
      double dudx = std::cos(x) - y * std::sin(x * y) + y;
      double dudy = -x * std::sin(x * y) + x;
      int q = jy * Q_1d + jx;
      REQUIRE(v[0 * Q2 + q] == Approx(dudx).margin(1e-11));
      REQUIRE(v[1 * Q2 + q] == Approx(dudy).margin(1e-11));
    }
  }
}

TEST_CASE("tensor_basis_apply_interp matches an analytic function in 3D",
          "[tensor-contract][manufactured]") {
  const int P_1d = 16, Q_1d = 16;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(3, 1, P_1d, Q_1d);

  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);

  // u(x,y,z) = sin(x) + cos(x*y) + x*y*z
  std::vector<double> u(P_1d * P_1d * P_1d);
  for (int iz = 0; iz < P_1d; ++iz) {
    for (int iy = 0; iy < P_1d; ++iy) {
      for (int ix = 0; ix < P_1d; ++ix) {
        double x = nodes[ix], y = nodes[iy], z = nodes[iz];
        u[(iz * P_1d + iy) * P_1d + ix] = std::sin(x) + std::cos(x * y) + x * y * z;
      }
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_interp(basis, 1, u, v);

  REQUIRE(v.size() == static_cast<size_t>(Q_1d * Q_1d * Q_1d));
  for (int jz = 0; jz < Q_1d; ++jz) {
    for (int jy = 0; jy < Q_1d; ++jy) {
      for (int jx = 0; jx < Q_1d; ++jx) {
        double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy], z = basis.q_ref_1d[jz];
        double expected = std::sin(x) + std::cos(x * y) + x * y * z;
        int q = (jz * Q_1d + jy) * Q_1d + jx;
        REQUIRE(v[q] == Approx(expected).margin(1e-11));
      }
    }
  }
}

TEST_CASE("tensor_basis_apply_grad matches an analytic gradient in 3D",
          "[tensor-contract][manufactured]") {
  const int P_1d = 16, Q_1d = 16;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(3, 1, P_1d, Q_1d);

  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);

  // u(x,y,z) = sin(x) + cos(x*y) + x*y*z
  // du/dx = cos(x) - y*sin(x*y) + y*z
  // du/dy = -x*sin(x*y) + x*z
  // du/dz = x*y
  std::vector<double> u(P_1d * P_1d * P_1d);
  for (int iz = 0; iz < P_1d; ++iz) {
    for (int iy = 0; iy < P_1d; ++iy) {
      for (int ix = 0; ix < P_1d; ++ix) {
        double x = nodes[ix], y = nodes[iy], z = nodes[iz];
        u[(iz * P_1d + iy) * P_1d + ix] = std::sin(x) + std::cos(x * y) + x * y * z;
      }
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_grad(basis, 1, u, v);

  const int Q3 = Q_1d * Q_1d * Q_1d;
  REQUIRE(v.size() == static_cast<size_t>(3 * Q3));
  for (int jz = 0; jz < Q_1d; ++jz) {
    for (int jy = 0; jy < Q_1d; ++jy) {
      for (int jx = 0; jx < Q_1d; ++jx) {
        double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy], z = basis.q_ref_1d[jz];
        double dudx = std::cos(x) - y * std::sin(x * y) + y * z;
        double dudy = -x * std::sin(x * y) + x * z;
        double dudz = x * y;
        int q = (jz * Q_1d + jy) * Q_1d + jx;
        REQUIRE(v[0 * Q3 + q] == Approx(dudx).margin(1e-11));
        REQUIRE(v[1 * Q3 + q] == Approx(dudy).margin(1e-11));
        REQUIRE(v[2 * Q3 + q] == Approx(dudz).margin(1e-11));
      }
    }
  }
}

// ---------------------------------------------------------------------
// num_comp=3 (3D displacement field u = (ux, uy, uz), matching the
// eventual elasticity target) in 3D, mirroring the Julia eval2 pattern
// but extended to a third component:
//   u1(x,y,z) = sin(x) + cos(x*y) + x*y*z
//   u2(x,y,z) = cos(y) + sin(y*z) + x*y - z
//   u3(x,y,z) = sin(x*y*z) + x^2 - y*z
// Components are concatenated (not interleaved): u[c * P_1d^3 + node].
// ---------------------------------------------------------------------

TEST_CASE("tensor_basis_apply_interp matches analytic functions in 3D, num_comp=3",
          "[tensor-contract][manufactured][num_comp]") {
  const int P_1d = 16, Q_1d = 16, num_comp = 3;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(3, num_comp, P_1d, Q_1d);

  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);

  const int P3 = P_1d * P_1d * P_1d;
  std::vector<double> u(num_comp * P3);
  for (int iz = 0; iz < P_1d; ++iz) {
    for (int iy = 0; iy < P_1d; ++iy) {
      for (int ix = 0; ix < P_1d; ++ix) {
        double x = nodes[ix], y = nodes[iy], z = nodes[iz];
        int node = (iz * P_1d + iy) * P_1d + ix;
        u[0 * P3 + node] = std::sin(x) + std::cos(x * y) + x * y * z;
        u[1 * P3 + node] = std::cos(y) + std::sin(y * z) + x * y - z;
        u[2 * P3 + node] = std::sin(x * y * z) + x * x - y * z;
      }
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_interp(basis, 1, u, v);

  const int Q3 = Q_1d * Q_1d * Q_1d;
  REQUIRE(v.size() == static_cast<size_t>(num_comp * Q3));
  for (int jz = 0; jz < Q_1d; ++jz) {
    for (int jy = 0; jy < Q_1d; ++jy) {
      for (int jx = 0; jx < Q_1d; ++jx) {
        double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy], z = basis.q_ref_1d[jz];
        double u1 = std::sin(x) + std::cos(x * y) + x * y * z;
        double u2 = std::cos(y) + std::sin(y * z) + x * y - z;
        double u3 = std::sin(x * y * z) + x * x - y * z;
        int q = (jz * Q_1d + jy) * Q_1d + jx;
        REQUIRE(v[0 * Q3 + q] == Approx(u1).margin(1e-11));
        REQUIRE(v[1 * Q3 + q] == Approx(u2).margin(1e-11));
        REQUIRE(v[2 * Q3 + q] == Approx(u3).margin(1e-11));
      }
    }
  }
}

TEST_CASE("tensor_basis_apply_grad matches analytic gradients in 3D, num_comp=3",
          "[tensor-contract][manufactured][num_comp]") {
  const int P_1d = 16, Q_1d = 16, num_comp = 3;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(3, num_comp, P_1d, Q_1d);

  std::vector<double> nodes;
  fem::gauss_lobatto_points(P_1d, nodes);

  const int P3 = P_1d * P_1d * P_1d;
  std::vector<double> u(num_comp * P3);
  for (int iz = 0; iz < P_1d; ++iz) {
    for (int iy = 0; iy < P_1d; ++iy) {
      for (int ix = 0; ix < P_1d; ++ix) {
        double x = nodes[ix], y = nodes[iy], z = nodes[iz];
        int node = (iz * P_1d + iy) * P_1d + ix;
        u[0 * P3 + node] = std::sin(x) + std::cos(x * y) + x * y * z;
        u[1 * P3 + node] = std::cos(y) + std::sin(y * z) + x * y - z;
        u[2 * P3 + node] = std::sin(x * y * z) + x * x - y * z;
      }
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_grad(basis, 1, u, v);

  const int Q3 = Q_1d * Q_1d * Q_1d;
  REQUIRE(v.size() == static_cast<size_t>(3 * num_comp * Q3));
  for (int jz = 0; jz < Q_1d; ++jz) {
    for (int jy = 0; jy < Q_1d; ++jy) {
      for (int jx = 0; jx < Q_1d; ++jx) {
        double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy], z = basis.q_ref_1d[jz];
        int q = (jz * Q_1d + jy) * Q_1d + jx;

        // u1 = sin(x) + cos(x*y) + x*y*z
        double du1dx = std::cos(x) - y * std::sin(x * y) + y * z;
        double du1dy = -x * std::sin(x * y) + x * z;
        double du1dz = x * y;
        // u2 = cos(y) + sin(y*z) + x*y - z
        double du2dx = y;
        double du2dy = -std::sin(y) + z * std::cos(y * z) + x;
        double du2dz = y * std::cos(y * z) - 1.0;
        // u3 = sin(x*y*z) + x^2 - y*z
        double du3dx = y * z * std::cos(x * y * z) + 2.0 * x;
        double du3dy = x * z * std::cos(x * y * z) - z;
        double du3dz = x * y * std::cos(x * y * z) - y;

        // v[(d_axis * num_comp + component) * Q3 + q]
        REQUIRE(v[(0 * num_comp + 0) * Q3 + q] == Approx(du1dx).margin(1e-11));
        REQUIRE(v[(1 * num_comp + 0) * Q3 + q] == Approx(du1dy).margin(1e-11));
        REQUIRE(v[(2 * num_comp + 0) * Q3 + q] == Approx(du1dz).margin(1e-11));
        REQUIRE(v[(0 * num_comp + 1) * Q3 + q] == Approx(du2dx).margin(1e-11));
        REQUIRE(v[(1 * num_comp + 1) * Q3 + q] == Approx(du2dy).margin(1e-11));
        REQUIRE(v[(2 * num_comp + 1) * Q3 + q] == Approx(du2dz).margin(1e-11));
        REQUIRE(v[(0 * num_comp + 2) * Q3 + q] == Approx(du3dx).margin(1e-11));
        REQUIRE(v[(1 * num_comp + 2) * Q3 + q] == Approx(du3dy).margin(1e-11));
        REQUIRE(v[(2 * num_comp + 2) * Q3 + q] == Approx(du3dz).margin(1e-11));
      }
    }
  }
}

// ---------------------------------------------------------------------
// num_elem=3 (element batching) in 2D, num_comp=1. Each element gets
// its own bilinear function u_e(x,y) = a_e + b_e*x + c_e*y + d_e*x*y,
// exactly representable by P_1d=2. u/v layout is interleaved by element
// u[node * num_elem + elem].
// ---------------------------------------------------------------------

namespace {
constexpr int kNumElem = 3;
constexpr double kElemCoeffs[kNumElem][4] = {
    {1.0, 2.0, -3.0, 0.5},
    {-2.0, 0.7, 1.3, -1.1},
    {0.0, -1.0, 2.0, 3.0},
};
}  // namespace

TEST_CASE("tensor_basis_apply_interp batches independently over num_elem",
          "[tensor-contract][num_elem]") {
  const int P_1d = 2, Q_1d = 3;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(2, 1, P_1d, Q_1d);
  const double nodes1d[2] = {-1.0, 1.0};

  std::vector<double> u(P_1d * P_1d * kNumElem);
  for (int iy = 0; iy < P_1d; ++iy) {
    for (int ix = 0; ix < P_1d; ++ix) {
      int node = iy * P_1d + ix;
      double x = nodes1d[ix], y = nodes1d[iy];
      for (int e = 0; e < kNumElem; ++e) {
        double a = kElemCoeffs[e][0], b = kElemCoeffs[e][1];
        double c = kElemCoeffs[e][2], d = kElemCoeffs[e][3];
        u[node * kNumElem + e] = a + b * x + c * y + d * x * y;
      }
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_interp(basis, kNumElem, u, v);

  const int Qdim = Q_1d * Q_1d;
  REQUIRE(v.size() == static_cast<size_t>(Qdim * kNumElem));
  for (int jy = 0; jy < Q_1d; ++jy) {
    for (int jx = 0; jx < Q_1d; ++jx) {
      int q = jy * Q_1d + jx;
      double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy];
      for (int e = 0; e < kNumElem; ++e) {
        double a = kElemCoeffs[e][0], b = kElemCoeffs[e][1];
        double c = kElemCoeffs[e][2], d = kElemCoeffs[e][3];
        double expected = a + b * x + c * y + d * x * y;
        REQUIRE(v[q * kNumElem + e] == Approx(expected).margin(1e-12));
      }
    }
  }
}

TEST_CASE("tensor_basis_apply_grad batches independently over num_elem",
          "[tensor-contract][num_elem]") {
  const int P_1d = 2, Q_1d = 3;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(2, 1, P_1d, Q_1d);
  const double nodes1d[2] = {-1.0, 1.0};

  std::vector<double> u(P_1d * P_1d * kNumElem);
  for (int iy = 0; iy < P_1d; ++iy) {
    for (int ix = 0; ix < P_1d; ++ix) {
      int node = iy * P_1d + ix;
      double x = nodes1d[ix], y = nodes1d[iy];
      for (int e = 0; e < kNumElem; ++e) {
        double a = kElemCoeffs[e][0], b = kElemCoeffs[e][1];
        double c = kElemCoeffs[e][2], d = kElemCoeffs[e][3];
        u[node * kNumElem + e] = a + b * x + c * y + d * x * y;
      }
    }
  }

  std::vector<double> v;
  fem::tensor_basis_apply_grad(basis, kNumElem, u, v);

  const int Qdim = Q_1d * Q_1d;
  REQUIRE(v.size() == static_cast<size_t>(2 * Qdim * kNumElem));
  for (int jy = 0; jy < Q_1d; ++jy) {
    for (int jx = 0; jx < Q_1d; ++jx) {
      int q = jy * Q_1d + jx;
      double x = basis.q_ref_1d[jx], y = basis.q_ref_1d[jy];
      for (int e = 0; e < kNumElem; ++e) {
        double b = kElemCoeffs[e][1], c = kElemCoeffs[e][2], d = kElemCoeffs[e][3];
        double expected_dudx = b + d * y;
        double expected_dudy = c + d * x;
        double got_dudx = v[(0 * Qdim + q) * kNumElem + e];
        double got_dudy = v[(1 * Qdim + q) * kNumElem + e];
        REQUIRE(got_dudx == Approx(expected_dudx).margin(1e-12));
        REQUIRE(got_dudy == Approx(expected_dudy).margin(1e-12));
      }
    }
  }
}
