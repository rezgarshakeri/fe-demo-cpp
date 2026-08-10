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
  fem::TensorBasis basis = fem::TensorBasis::create_tensor_H1_lagrange(1, P_1d, Q_1d);

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