#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "elem-restriction.hpp"

using Catch::Approx;

// ---------------------------------------------------------------------
// 1D, 2 linear elements sharing one node: 3 global DOFs {0, 1, 2}.
// Element 0's local nodes map to global [0, 1]; element 1's to [1, 2].
// offsets = [0, 1, 1, 2] (row-major [num_elem=2][elem_size=2]).
// ---------------------------------------------------------------------

static fem::ElemRestriction make_shared_node_restriction() {
  fem::ElemRestriction r;
  fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2}, r);
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
  // element 0 -> global [0,1], element 1 -> global [2,3]
  fem::elem_restriction_create(2, 2, 2, 100, 200, {0, 1, 2, 3}, r);

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
  fem::elem_restriction_create(2, 2, 2, 100, 200, {0, 1, 2, 3}, r);

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

// ---------------------------------------------------------------------
// elem_restriction_create: validation. Every argument that would make
// apply read/write out of bounds (or make no sense) must throw.
// ---------------------------------------------------------------------

TEST_CASE("elem_restriction_create: fills in every field, including e_size", "[elem-restriction][create]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create(2, 2, 3, 10, 30, {0, 1, 1, 2}, r);

  REQUIRE(r.num_elem == 2);
  REQUIRE(r.elem_size == 2);
  REQUIRE(r.num_comp == 3);
  REQUIRE(r.comp_stride == 10);
  REQUIRE(r.l_size == 30);
  REQUIRE(r.e_size == 3 * 2 * 2);  // num_comp * elem_size * num_elem
  REQUIRE(r.offsets == std::vector<int>{0, 1, 1, 2});
}

TEST_CASE("elem_restriction_create: rejects invalid sizes", "[elem-restriction][create]") {
  fem::ElemRestriction r;
  const std::vector<int> ok = {0, 1, 1, 2};

  REQUIRE_THROWS_AS(fem::elem_restriction_create(0, 2, 1, 1, 3, {}, r), std::invalid_argument);   // num_elem
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 0, 1, 1, 3, {}, r), std::invalid_argument);   // elem_size
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 2, 0, 1, 3, ok, r), std::invalid_argument);   // num_comp
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 2, 1, 1, 0, ok, r), std::invalid_argument);   // l_size
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 2, 2, 0, 200, ok, r), std::invalid_argument); // comp_stride with num_comp>1
}

TEST_CASE("elem_restriction_create: comp_stride is irrelevant when num_comp == 1", "[elem-restriction][create]") {
  fem::ElemRestriction r;
  REQUIRE_NOTHROW(fem::elem_restriction_create(2, 2, 1, 0, 3, {0, 1, 1, 2}, r));
}

TEST_CASE("elem_restriction_create: rejects a wrong-sized offsets array", "[elem-restriction][create]") {
  fem::ElemRestriction r;
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1}, r), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2, 2}, r), std::invalid_argument);
}

TEST_CASE("elem_restriction_create: rejects out-of-range offsets", "[elem-restriction][create]") {
  fem::ElemRestriction r;
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 3}, r), std::invalid_argument);   // == l_size
  REQUIRE_THROWS_AS(fem::elem_restriction_create(2, 2, 1, 1, 3, {0, -1, 1, 2}, r), std::invalid_argument);  // negative
}

TEST_CASE("elem_restriction_create: rejects a comp_stride that pushes later components out of range",
          "[elem-restriction][create]") {
  fem::ElemRestriction r;
  // offsets themselves are in [0, 3), but component 1 would read offset + 100 >= l_size
  REQUIRE_THROWS_AS(fem::elem_restriction_create(1, 2, 2, 100, 3, {0, 1}, r), std::invalid_argument);
  // exactly fitting is fine: max offset 1 + (2-1)*2 = 3 < 4
  REQUIRE_NOTHROW(fem::elem_restriction_create(1, 2, 2, 2, 4, {0, 1}, r));
  // one too small: 1 + 2 = 3 >= 3
  REQUIRE_THROWS_AS(fem::elem_restriction_create(1, 2, 2, 2, 3, {0, 1}, r), std::invalid_argument);
}

// ---------------------------------------------------------------------
// elem_restriction_get_multiplicity = E^T (E 1): how many elements touch
// each L-vector entry. Ported from libCEED's t209-elemrestriction.c.
// ---------------------------------------------------------------------

TEST_CASE("elem_restriction_get_multiplicity matches libCEED t209", "[elem-restriction][multiplicity]") {
  // 3 elements x 4 nodes, consecutive elements share one node: nodes 3 and 6 are shared.
  const int num_elem = 3;
  std::vector<int> offsets;
  for (int e = 0; e < num_elem; ++e)
    for (int j = 0; j < 4; ++j) offsets.push_back(e * 3 + j);

  fem::ElemRestriction r;
  fem::elem_restriction_create(num_elem, 4, 1, 1, 3 * num_elem + 1, offsets, r);

  std::vector<double> mult;  // deliberately empty: output parameter, must be sized by the function
  fem::elem_restriction_get_multiplicity(r, mult);

  REQUIRE(mult.size() == static_cast<size_t>(3 * num_elem + 1));
  for (int i = 0; i < 3 * num_elem + 1; ++i) {
    const bool shared = i > 0 && i < 3 * num_elem && i % 3 == 0;
    REQUIRE(mult[i] == Approx(shared ? 2.0 : 1.0));
  }
}

TEST_CASE("elem_restriction_get_multiplicity overwrites stale contents of mult", "[elem-restriction][multiplicity]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2}, r);

  std::vector<double> mult = {99.0, 99.0, 99.0, 99.0, 99.0};  // wrong size and garbage values
  fem::elem_restriction_get_multiplicity(r, mult);

  REQUIRE(mult.size() == 3);
  REQUIRE(mult[0] == Approx(1.0));
  REQUIRE(mult[1] == Approx(2.0));  // the shared node
  REQUIRE(mult[2] == Approx(1.0));
}

TEST_CASE("elem_restriction_get_multiplicity counts per component and leaves untouched entries at 0",
          "[elem-restriction][multiplicity]") {
  // num_comp=2, comp_stride=5, l_size=12: L entries 0..3 (comp 0) and 5..8 (comp 1) are used,
  // entries 4 and 9..11 belong to no element.
  fem::ElemRestriction r;
  fem::elem_restriction_create(2, 2, 2, 5, 12, {0, 1, 1, 2}, r);

  std::vector<double> mult;
  fem::elem_restriction_get_multiplicity(r, mult);

  const std::vector<double> expected = {1, 2, 1, 0, 0,   // component 0: node 1 shared
                                        1, 2, 1, 0, 0,   // component 1: same pattern
                                        0, 0};           // outside every element
  REQUIRE(mult.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) REQUIRE(mult[i] == Approx(expected[i]));
}
