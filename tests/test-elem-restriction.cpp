#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "elem-restriction.hpp"
#include "tensor-contract.hpp"

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

// ---------------------------------------------------------------------
// Strided restriction (quadrature-point data). The L-vector index of
// (node i, component k, element e) is i*s0 + k*s1 + e*s2; nothing is
// shared. Backend strides = our own E/Q-vector layout, so gather is a
// plain copy.
// ---------------------------------------------------------------------

namespace {
double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}
std::vector<double> ramp(size_t n, double scale) {  // deterministic, non-symmetric test data
  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = scale * (0.37 * i - 1.3 + 0.01 * (i % 7) * (i % 5));
  return v;
}
}  // namespace

TEST_CASE("elem_restriction_create_strided: backend strides fill in every field",
          "[elem-restriction][strided]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(3, 4, 2, 24, r);  // num_elem=3, elem_size=4, num_comp=2

  REQUIRE(r.type == fem::RestrictionType::Strided);
  REQUIRE(r.has_backend_strides);
  REQUIRE(r.num_elem == 3);
  REQUIRE(r.elem_size == 4);
  REQUIRE(r.num_comp == 2);
  REQUIRE(r.l_size == 24);
  REQUIRE(r.e_size == 24);
  REQUIRE(r.offsets.empty());
  // {num_elem, elem_size * num_elem, 1}: E[(k * elem_size + i) * num_elem + e]
  REQUIRE(r.strides == std::array<int, 3>{3, 12, 1});
}

TEST_CASE("elem_restriction_create_strided: user strides are stored, not flagged as backend",
          "[elem-restriction][strided]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(2, 3, 2, 12, {1, 3, 6}, r);
  REQUIRE(r.type == fem::RestrictionType::Strided);
  REQUIRE_FALSE(r.has_backend_strides);
  REQUIRE(r.strides == std::array<int, 3>{1, 3, 6});
}

TEST_CASE("elem_restriction_create_strided: rejects invalid sizes and out-of-range strides",
          "[elem-restriction][strided]") {
  fem::ElemRestriction r;
  REQUIRE_THROWS_AS(fem::elem_restriction_create_strided(0, 4, 1, 8, r), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::elem_restriction_create_strided(2, 0, 1, 8, r), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::elem_restriction_create_strided(2, 4, 0, 8, r), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::elem_restriction_create_strided(2, 4, 1, 0, r), std::invalid_argument);

  // backend: needs l_size >= num_elem * elem_size * num_comp = 16
  REQUIRE_THROWS_AS(fem::elem_restriction_create_strided(2, 4, 2, 15, r), std::invalid_argument);
  REQUIRE_NOTHROW(fem::elem_restriction_create_strided(2, 4, 2, 16, r));

  // user strides: largest index (3-1)*1 + (2-1)*3 + (2-1)*6 = 11, so l_size 12 fits and 11 doesn't
  REQUIRE_NOTHROW(fem::elem_restriction_create_strided(2, 3, 2, 12, {1, 3, 6}, r));
  REQUIRE_THROWS_AS(fem::elem_restriction_create_strided(2, 3, 2, 11, {1, 3, 6}, r), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::elem_restriction_create_strided(2, 3, 2, 12, {-1, 3, 6}, r), std::invalid_argument);
}

TEST_CASE("elem_restriction_create: resets a struct previously used as strided",
          "[elem-restriction][strided]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(2, 2, 1, 4, r);
  fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2}, r);
  REQUIRE(r.type == fem::RestrictionType::Offset);
  REQUIRE_FALSE(r.has_backend_strides);
}

TEST_CASE("elem_restriction_apply: backend-strided gather is the identity (pins the layout)",
          "[elem-restriction][strided]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(3, 4, 2, 24, r);

  std::vector<double> L(r.l_size);
  for (size_t g = 0; g < L.size(); ++g) L[g] = static_cast<double>(g);

  std::vector<double> E;
  fem::elem_restriction_apply(r, fem::ContractMode::NoTranspose, L, E);
  REQUIRE(E.size() == static_cast<size_t>(r.e_size));
  REQUIRE(E == L);  // bitwise: backend strides are exactly the E-vector layout
}

TEST_CASE("elem_restriction_apply: user strides gather in the requested layout",
          "[elem-restriction][strided]") {
  // libCEED's CPU layout (element outermost): strides {1, elem_size, elem_size * num_comp}
  // num_elem=2, elem_size=3, num_comp=2 -> L index = i + 3k + 6e.
  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(2, 3, 2, 12, {1, 3, 6}, r);

  std::vector<double> L(r.l_size);
  for (size_t g = 0; g < L.size(); ++g) L[g] = static_cast<double>(g);

  std::vector<double> E;
  fem::elem_restriction_apply(r, fem::ContractMode::NoTranspose, L, E);

  // E[(k * 3 + i) * 2 + e] = L[i + 3k + 6e]
  const std::vector<double> expected = {0, 6, 1, 7, 2, 8, 3, 9, 4, 10, 5, 11};
  REQUIRE(E.size() == expected.size());
  for (size_t idx = 0; idx < expected.size(); ++idx) REQUIRE(E[idx] == Approx(expected[idx]));
}

TEST_CASE("elem_restriction_apply: strided Transpose is the adjoint of NoTranspose",
          "[elem-restriction][strided]") {
  fem::ElemRestriction backend, user;
  fem::elem_restriction_create_strided(3, 4, 2, 24, backend);
  fem::elem_restriction_create_strided(2, 3, 2, 14, {1, 3, 6}, user);  // l_size > used: entries 12, 13 untouched

  for (const fem::ElemRestriction* r : {&backend, &user}) {
    const auto u = ramp(r->l_size, 1.0);
    const auto w = ramp(r->e_size, -0.5);

    std::vector<double> Eu;
    fem::elem_restriction_apply(*r, fem::ContractMode::NoTranspose, u, Eu);
    std::vector<double> ETw(r->l_size, 0.0);
    fem::elem_restriction_apply(*r, fem::ContractMode::Transpose, w, ETw);

    REQUIRE(ETw.size() == static_cast<size_t>(r->l_size));
    REQUIRE(dot(Eu, w) == Approx(dot(u, ETw)).margin(1e-12));
  }
}

TEST_CASE("elem_restriction_apply: strided Transpose accumulates, doesn't overwrite",
          "[elem-restriction][strided]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(2, 2, 1, 4, r);

  const std::vector<double> E = {1, 2, 3, 4};
  std::vector<double> L = {10, 20, 30, 40};
  fem::elem_restriction_apply(r, fem::ContractMode::Transpose, E, L);
  REQUIRE(L == std::vector<double>{11, 22, 33, 44});  // backend strides: E and L share a layout
}

TEST_CASE("elem_restriction_get_multiplicity: strided is 1 where used, 0 elsewhere",
          "[elem-restriction][strided][multiplicity]") {
  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(2, 3, 2, 14, {1, 3, 6}, r);  // indices 0..11 used

  std::vector<double> mult;
  fem::elem_restriction_get_multiplicity(r, mult);
  REQUIRE(mult.size() == 14);
  for (int g = 0; g < 14; ++g) REQUIRE(mult[g] == Approx(g < 12 ? 1.0 : 0.0));
}

TEST_CASE("elem_restriction_apply: backend strides match the basis Q-vector layout",
          "[elem-restriction][strided][basis]") {
  // The point of backend strides: a Q-vector produced by tensor_basis_apply_interp can be stored
  // through a backend-strided restriction (qdata) and read back unchanged, with no reshuffle.
  const int dim = 2, num_comp = 2, num_elem = 3, P_1d = 3, Q_1d = 4;
  auto basis = fem::TensorBasis::create_tensor_H1_lagrange(dim, num_comp, P_1d, Q_1d);
  const int Qdim = Q_1d * Q_1d;

  const auto u = ramp(num_comp * P_1d * P_1d * num_elem, 1.0);
  std::vector<double> q_vec;
  fem::tensor_basis_apply_interp(basis, num_elem, fem::ContractMode::NoTranspose, u, q_vec);

  fem::ElemRestriction r;
  fem::elem_restriction_create_strided(num_elem, Qdim, num_comp, num_comp * Qdim * num_elem, r);

  std::vector<double> stored(r.l_size, 0.0);
  fem::elem_restriction_apply(r, fem::ContractMode::Transpose, q_vec, stored);  // Q-vector -> storage
  std::vector<double> read_back;
  fem::elem_restriction_apply(r, fem::ContractMode::NoTranspose, stored, read_back);  // storage -> Q-vector

  REQUIRE(stored == q_vec);
  REQUIRE(read_back == q_vec);
}
