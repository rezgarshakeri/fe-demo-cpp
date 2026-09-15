#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include "elem-restriction.hpp"

using Catch::Approx;

// ---------------------------------------------------------------------
// 1D, 2 linear elements sharing one node: 3 global DOFs {0, 1, 2}.
// Element 0's local nodes map to global [0, 1]; element 1's to [1, 2].
// offsets = [0, 1, 1, 2] (row-major [num_elem=2][elem_size=2]).
// ---------------------------------------------------------------------

static fem::ElemRestriction make_shared_node_restriction() {
  fem::ElemRestriction r;
  r.num_elem = 2;
  r.elem_size = 2;
  r.num_comp = 1;
  r.comp_stride = 1;
  r.l_size = 3;
  r.offsets = {0, 1, 1, 2};
  return r;
}

TEST_CASE("elem_restriction_apply: gather duplicates the shared node's value",
          "[elem-restriction]") {
  fem::ElemRestriction r = make_shared_node_restriction();
  std::vector<double> L = {10.0, 20.0, 30.0};

  std::vector<double> E;
  fem::elem_restriction_apply(r, fem::ContractMode::NoTranspose, L, E);

  REQUIRE(E.size() == static_cast<size_t>(r.num_comp * r.elem_size * r.num_elem));
  // E[(k*elem_size+i)*num_elem+e], num_comp=1 so k=0 always -> E[i*num_elem+e]
  REQUIRE(E[0] == Approx(10.0));  // i=0, e=0 -> offsets[0]=0 -> L[0]
  REQUIRE(E[1] == Approx(20.0));  // i=0, e=1 -> offsets[2]=1 -> L[1]
  REQUIRE(E[2] == Approx(20.0));  // i=1, e=0 -> offsets[1]=1 -> L[1]
  REQUIRE(E[3] == Approx(30.0));  // i=1, e=1 -> offsets[3]=2 -> L[2]
}

TEST_CASE("elem_restriction_apply: scatter-add sums contributions at the shared node",
          "[elem-restriction]") {
  fem::ElemRestriction r = make_shared_node_restriction();
  std::vector<double> E = {10.0, 20.0, 20.0, 30.0};  // the gather result above

  std::vector<double> L(r.l_size, 0.0);  // caller zeroes first, per the documented contract
  fem::elem_restriction_apply(r, fem::ContractMode::Transpose, E, L);

  REQUIRE(L.size() == static_cast<size_t>(r.l_size));
  REQUIRE(L[0] == Approx(10.0));  // only element 0's node 0 contributes
  REQUIRE(L[1] == Approx(40.0));  // element 0's node 1 (20) + element 1's node 0 (20): shared node sums
  REQUIRE(L[2] == Approx(30.0));  // only element 1's node 1 contributes
}

TEST_CASE("elem_restriction_apply: Transpose accumulates onto existing values, doesn't overwrite",
          "[elem-restriction]") {
  fem::ElemRestriction r = make_shared_node_restriction();
  std::vector<double> E = {10.0, 20.0, 20.0, 30.0};

  std::vector<double> L = {100.0, 200.0, 300.0};  // pre-existing, non-zero
  fem::elem_restriction_apply(r, fem::ContractMode::Transpose, E, L);

  REQUIRE(L[0] == Approx(110.0));  // 100 + 10
  REQUIRE(L[1] == Approx(240.0));  // 200 + 40
  REQUIRE(L[2] == Approx(330.0));  // 300 + 30
}

// ---------------------------------------------------------------------
// num_comp=2, num_elem=2: checks the E-vector layout precisely --
// components concatenated (outer), elements interleaved (inner), i.e.
// E[(component * elem_size + i) * num_elem + e] -- matching
// tensor_basis_apply_interp/_grad's own convention (see
// elem-restriction.hpp and tensor-contract.hpp).
// ---------------------------------------------------------------------

TEST_CASE("elem_restriction_apply: E-vector layout is component-outer, element-inner",
          "[elem-restriction]") {
  fem::ElemRestriction r;
  r.num_elem = 2;
  r.elem_size = 2;
  r.num_comp = 2;
  r.comp_stride = 100;
  r.l_size = 200;
  r.offsets = {0, 1, 2, 3};  // element 0 -> global [0,1], element 1 -> global [2,3]

  std::vector<double> L(r.l_size);
  for (size_t i = 0; i < L.size(); ++i) L[i] = static_cast<double>(i);  // L[g] = g

  std::vector<double> E;
  fem::elem_restriction_apply(r, fem::ContractMode::NoTranspose, L, E);

  REQUIRE(E.size() == static_cast<size_t>(r.num_comp * r.elem_size * r.num_elem));
  const std::vector<double> expected = {0, 2, 1, 3, 100, 102, 101, 103};
  for (size_t idx = 0; idx < expected.size(); ++idx) {
    REQUIRE(E[idx] == Approx(expected[idx]));
  }
}

TEST_CASE("elem_restriction_apply: gather then scatter-add round-trips correctly with num_comp",
          "[elem-restriction]") {
  fem::ElemRestriction r;
  r.num_elem = 2;
  r.elem_size = 2;
  r.num_comp = 2;
  r.comp_stride = 100;
  r.l_size = 200;
  r.offsets = {0, 1, 2, 3};

  std::vector<double> L(r.l_size, 0.0);
  L[0] = 5; L[1] = 6; L[2] = 7; L[3] = 8;
  L[100] = 50; L[101] = 60; L[102] = 70; L[103] = 80;

  std::vector<double> E;
  fem::elem_restriction_apply(r, fem::ContractMode::NoTranspose, L, E);

  std::vector<double> L_out(r.l_size, 0.0);
  fem::elem_restriction_apply(r, fem::ContractMode::Transpose, E, L_out);

  // No node is shared between the two elements here, so scatter-add
  // should exactly reproduce the original (nonzero) entries of L.
  REQUIRE(L_out[0] == Approx(5.0));
  REQUIRE(L_out[1] == Approx(6.0));
  REQUIRE(L_out[2] == Approx(7.0));
  REQUIRE(L_out[3] == Approx(8.0));
  REQUIRE(L_out[100] == Approx(50.0));
  REQUIRE(L_out[101] == Approx(60.0));
  REQUIRE(L_out[102] == Approx(70.0));
  REQUIRE(L_out[103] == Approx(80.0));
}
