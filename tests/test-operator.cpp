#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <stdexcept>
#include "basis.hpp"
#include "elem-restriction.hpp"
#include "operator.hpp"
#include "qfunction.hpp"
#include "qfunctions/mass.hpp"
#include "tensor-contract.hpp"

using Catch::Approx;
using fem::ContractMode;
using fem::EvalMode;

namespace {

fem::QFunction make_apply_mass_qf() {
  fem::QFunction qf;
  fem::qfunction_create(fem::qfunctions::apply_mass, qf);
  fem::qfunction_add_input(qf, "u", 1, EvalMode::Interp);
  fem::qfunction_add_input(qf, "qdata", 1, EvalMode::None);
  fem::qfunction_add_output(qf, "v", 1, EvalMode::Interp);
  return qf;
}

fem::QFunction make_build_mass_qf(int dim) {
  fem::QFunction qf;
  fem::qfunction_create(fem::qfunctions::build_mass, qf);
  fem::qfunction_add_input(qf, "dx", dim * dim, EvalMode::Grad);
  fem::qfunction_add_input(qf, "weights", 1, EvalMode::Weight);
  fem::qfunction_add_output(qf, "qdata", 1, EvalMode::None);
  fem::qfunction_set_context(qf, fem::qfunctions::BuildMassContext{dim});
  return qf;
}

double dot(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}
double sum(const std::vector<double>& a) {
  double s = 0.0;
  for (double x : a) s += x;
  return s;
}
std::vector<double> ramp(size_t n, double scale) {
  std::vector<double> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = scale * (0.37 * i - 1.3 + 0.01 * (i % 7) * (i % 5));
  return v;
}

// ---------------------------------------------------------------------
// 1D mesh: 2 linear elements, 3 nodes, local nodes -> global [0,1], [1,2].
// Linear basis (P_1d = 2), Q_1d = 3 Gauss points (exact for the degree-2
// mass integrand). qdata lives at the 3 x 2 quadrature points.
// ---------------------------------------------------------------------
struct Mesh1D {
  fem::TensorBasis basis = fem::TensorBasis::create_tensor_H1_lagrange(1, 1, 2, 3);
  fem::ElemRestriction rstr;    // nodes
  fem::ElemRestriction q_rstr;  // quadrature points (backend strides)
  Mesh1D() {
    fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2}, rstr);
    fem::elem_restriction_create_strided(2, 3, 1, 6, q_rstr);
  }
  // qdata = w * J for element sizes h0, h1 (J = h/2), in the backend-strided [q][e] layout.
  std::vector<double> qdata(double h0, double h1) const {
    std::vector<double> w;
    fem::tensor_basis_apply_weight(basis, 2, ContractMode::NoTranspose, w);
    for (int q = 0; q < 3; ++q) {
      w[q * 2 + 0] *= h0 / 2;
      w[q * 2 + 1] *= h1 / 2;
    }
    return w;
  }
  fem::Operator mass(const std::vector<double>& qdata) const {
    fem::Operator op;
    fem::operator_create(make_apply_mass_qf(), op);
    fem::operator_set_field(op, "u", rstr, basis, fem::vector_active());
    fem::operator_set_field(op, "qdata", q_rstr, std::nullopt, fem::vector_passive(qdata));
    fem::operator_set_field(op, "v", rstr, basis, fem::vector_active());
    return op;
  }
};

// ---------------------------------------------------------------------
// 2D mesh: 2 bilinear quads over a 3 x 2 grid of nodes, node(ix, iy) = 3*iy + ix.
// Bottom nodes x = {0, 1, 3} at y = 0; top nodes x = {0.5, 1, 2.5} at y = 2.
// Element 0 is a trapezoid of area 1.5, element 1 of area 3.5: total 5.
// ---------------------------------------------------------------------
struct Mesh2D {
  static constexpr double kArea = 5.0;
  fem::TensorBasis mesh_basis = fem::TensorBasis::create_tensor_H1_lagrange(2, 2, 2, 3);
  fem::TensorBasis sol_basis = fem::TensorBasis::create_tensor_H1_lagrange(2, 1, 2, 3);
  fem::ElemRestriction coord_rstr, sol_rstr, q_rstr;
  // [x0..x5, y0..y5]: component stride 6
  std::vector<double> coords = {0, 1, 3, 0.5, 1, 2.5, 0, 0, 0, 2, 2, 2};
  Mesh2D() {
    const std::vector<int> offsets = {0, 1, 3, 4, 1, 2, 4, 5};  // local lexicographic, x fastest
    fem::elem_restriction_create(2, 4, 2, 6, 12, offsets, coord_rstr);
    fem::elem_restriction_create(2, 4, 1, 1, 6, offsets, sol_rstr);
    fem::elem_restriction_create_strided(2, 9, 1, 18, q_rstr);
  }
  std::vector<double> build_qdata() const {
    fem::Operator op;
    fem::operator_create(make_build_mass_qf(2), op);
    fem::operator_set_field(op, "dx", coord_rstr, mesh_basis, fem::vector_active());
    fem::operator_set_field(op, "weights", std::nullopt, mesh_basis, fem::vector_none());
    fem::operator_set_field(op, "qdata", q_rstr, std::nullopt, fem::vector_active());
    std::vector<double> qdata;
    fem::operator_apply(op, coords, qdata);
    return qdata;
  }
  fem::Operator mass(const std::vector<double>& qdata) const {
    fem::Operator op;
    fem::operator_create(make_apply_mass_qf(), op);
    fem::operator_set_field(op, "u", sol_rstr, sol_basis, fem::vector_active());
    fem::operator_set_field(op, "qdata", q_rstr, std::nullopt, fem::vector_passive(qdata));
    fem::operator_set_field(op, "v", sol_rstr, sol_basis, fem::vector_active());
    return op;
  }
};

}  // namespace

// ---------------------------------------------------------------------
// operator_set_field bookkeeping and validation (pass without operator_apply)
// ---------------------------------------------------------------------

TEST_CASE("operator_create/set_field record fields and shared sizes", "[operator]") {
  Mesh1D m;
  fem::Operator op = m.mass(m.qdata(1, 1));

  REQUIRE(op.inputs.size() == 2);
  REQUIRE(op.outputs.size() == 1);
  REQUIRE(op.inputs[0]->name == "u");
  REQUIRE(op.inputs[1]->name == "qdata");
  REQUIRE(op.inputs[1]->vector.kind == fem::VectorKind::Passive);
  REQUIRE(op.inputs[1]->vector.data.size() == 6);
  REQUIRE(op.outputs[0]->name == "v");
  REQUIRE(op.num_elem == 2);
  REQUIRE(op.num_qpts == 3);
  REQUIRE(op.active_in_l_size == 3);
  REQUIRE(op.active_out_l_size == 3);
  REQUIRE_NOTHROW(fem::operator_check_ready(op));
}

TEST_CASE("operator_create rejects a QFunction without a user function", "[operator]") {
  fem::Operator op;
  REQUIRE_THROWS_AS(fem::operator_create(fem::QFunction{}, op), std::invalid_argument);
}

TEST_CASE("operator_check_ready rejects unset fields", "[operator]") {
  Mesh1D m;
  fem::Operator op;
  fem::operator_create(make_apply_mass_qf(), op);
  fem::operator_set_field(op, "u", m.rstr, m.basis, fem::vector_active());
  REQUIRE_THROWS_AS(fem::operator_check_ready(op), std::invalid_argument);  // qdata, v missing
}

TEST_CASE("operator_set_field rejects mismatched fields", "[operator]") {
  Mesh1D m;
  const auto qdata = m.qdata(1, 1);
  auto fresh_mass = [] {
    fem::Operator op;
    fem::operator_create(make_apply_mass_qf(), op);
    return op;
  };
  auto fresh_build = [](int dim) {
    fem::Operator op;
    fem::operator_create(make_build_mass_qf(dim), op);
    return op;
  };

  SECTION("unknown name, set twice") {
    fem::Operator op = fresh_mass();
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "w", m.rstr, m.basis, fem::vector_active()), std::invalid_argument);
    fem::operator_set_field(op, "u", m.rstr, m.basis, fem::vector_active());
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "u", m.rstr, m.basis, fem::vector_active()), std::invalid_argument);
  }
  SECTION("Weight: basis only, no restriction or vector") {
    fem::Operator op = fresh_build(1);
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "weights", m.rstr, m.basis, fem::vector_none()), std::invalid_argument);
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "weights", std::nullopt, std::nullopt, fem::vector_none()),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "weights", std::nullopt, m.basis, fem::vector_active()),
                      std::invalid_argument);
    REQUIRE_NOTHROW(fem::operator_set_field(op, "weights", std::nullopt, m.basis, fem::vector_none()));
  }
  SECTION("None: restriction only, num_comp == field size") {
    fem::Operator op = fresh_mass();
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "qdata", m.q_rstr, m.basis, fem::vector_passive(qdata)),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "qdata", std::nullopt, std::nullopt, fem::vector_passive(qdata)),
                      std::invalid_argument);
    fem::ElemRestriction two_comp;
    fem::elem_restriction_create_strided(2, 3, 2, 12, two_comp);
    REQUIRE_THROWS_AS(
        fem::operator_set_field(op, "qdata", two_comp, std::nullopt, fem::vector_passive(std::vector<double>(12))),
        std::invalid_argument);
  }
  SECTION("Interp: field size == basis num_comp, restriction elem_size == P_1d^dim") {
    fem::Operator op = fresh_mass();
    auto two_comp_basis = fem::TensorBasis::create_tensor_H1_lagrange(1, 2, 2, 3);
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "u", m.rstr, two_comp_basis, fem::vector_active()),
                      std::invalid_argument);
    fem::ElemRestriction three_nodes;  // elem_size 3, but a linear basis has 2 nodes
    fem::elem_restriction_create(2, 3, 1, 1, 5, {0, 1, 2, 2, 3, 4}, three_nodes);
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "u", three_nodes, m.basis, fem::vector_active()),
                      std::invalid_argument);
  }
  SECTION("Grad: field size == basis num_comp * dim") {
    fem::Operator op = fresh_build(2);  // "dx" has size 4
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "dx", m.rstr, m.basis, fem::vector_active()), std::invalid_argument);
  }
  SECTION("fields must agree on num_elem and number of quadrature points") {
    fem::Operator op = fresh_mass();
    fem::operator_set_field(op, "u", m.rstr, m.basis, fem::vector_active());
    fem::ElemRestriction three_elems, four_qpts;
    fem::elem_restriction_create_strided(3, 3, 1, 9, three_elems);
    fem::elem_restriction_create_strided(2, 4, 1, 8, four_qpts);
    REQUIRE_THROWS_AS(
        fem::operator_set_field(op, "qdata", three_elems, std::nullopt, fem::vector_passive(std::vector<double>(9))),
        std::invalid_argument);
    REQUIRE_THROWS_AS(
        fem::operator_set_field(op, "qdata", four_qpts, std::nullopt, fem::vector_passive(std::vector<double>(8))),
        std::invalid_argument);
  }
  SECTION("vectors: outputs active, passive sized l_size, active inputs share l_size") {
    fem::Operator op = fresh_mass();
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "v", m.rstr, m.basis, fem::vector_passive(std::vector<double>(3))),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(
        fem::operator_set_field(op, "qdata", m.q_rstr, std::nullopt, fem::vector_passive(std::vector<double>(5))),
        std::invalid_argument);
    REQUIRE_THROWS_AS(fem::operator_set_field(op, "u", m.rstr, m.basis, fem::vector_none()), std::invalid_argument);

    fem::QFunction two_in;
    fem::qfunction_create(fem::qfunctions::apply_mass, two_in);
    fem::qfunction_add_input(two_in, "a", 1, EvalMode::Interp);
    fem::qfunction_add_input(two_in, "b", 1, EvalMode::Interp);
    fem::qfunction_add_output(two_in, "c", 1, EvalMode::Interp);
    fem::Operator op2;
    fem::operator_create(two_in, op2);
    fem::ElemRestriction disjoint;  // same shape, but l_size 4
    fem::elem_restriction_create(2, 2, 1, 1, 4, {0, 1, 2, 3}, disjoint);
    fem::operator_set_field(op2, "a", m.rstr, m.basis, fem::vector_active());
    REQUIRE_THROWS_AS(fem::operator_set_field(op2, "b", disjoint, m.basis, fem::vector_active()),
                      std::invalid_argument);
  }
}

// ---------------------------------------------------------------------
// operator_apply (fail until operator_apply is implemented)
// ---------------------------------------------------------------------

TEST_CASE("operator_apply: 1D mass matrix matches (h/6)[2 1; 1 2] per element", "[operator][apply]") {
  Mesh1D m;
  fem::Operator op = m.mass(m.qdata(1, 1));  // nodes 0, 1, 2

  std::vector<double> v;
  fem::operator_apply(op, {1, 1, 1}, v);
  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == Approx(0.5));
  REQUIRE(v[1] == Approx(1.0));
  REQUIRE(v[2] == Approx(0.5));

  fem::operator_apply(op, {0, 1, 2}, v);  // M * x: 1/6 [0*2+1, (0+2*1) + (2*1+2), 1+2*2]
  REQUIRE(v[0] == Approx(1.0 / 6));
  REQUIRE(v[1] == Approx(1.0));
  REQUIRE(v[2] == Approx(5.0 / 6));
}

TEST_CASE("operator_apply: 1D build operator gives qdata = w * J", "[operator][apply]") {
  Mesh1D m;
  fem::Operator op;
  fem::operator_create(make_build_mass_qf(1), op);
  fem::operator_set_field(op, "dx", m.rstr, m.basis, fem::vector_active());
  fem::operator_set_field(op, "weights", std::nullopt, m.basis, fem::vector_none());
  fem::operator_set_field(op, "qdata", m.q_rstr, std::nullopt, fem::vector_active());

  std::vector<double> qdata;
  fem::operator_apply(op, {0, 1, 3}, qdata);  // element sizes 1 and 2
  const std::vector<double> expected = m.qdata(1, 2);
  REQUIRE(qdata.size() == expected.size());
  for (size_t i = 0; i < expected.size(); ++i) REQUIRE(qdata[i] == Approx(expected[i]));
}

TEST_CASE("operator_apply: 2D build + mass gives the exact area of a distorted mesh", "[operator][apply]") {
  Mesh2D m;
  const auto qdata = m.build_qdata();
  REQUIRE(sum(qdata) == Approx(Mesh2D::kArea));  // integral of 1 = sum of det(J) w

  fem::Operator op = m.mass(qdata);
  std::vector<double> v;
  fem::operator_apply(op, std::vector<double>(6, 1.0), v);
  REQUIRE(sum(v) == Approx(Mesh2D::kArea));  // 1^T M 1
}

TEST_CASE("operator_apply equals the hand-composed restrict/basis/QFunction chain", "[operator][apply]") {
  Mesh1D m;
  const auto qdata = m.qdata(1, 2);
  fem::Operator op = m.mass(qdata);
  const std::vector<double> u = {0.3, -1.2, 2.5};

  std::vector<double> e_u, q_u, e_v, expected(3, 0.0);
  fem::elem_restriction_apply(m.rstr, ContractMode::NoTranspose, u, e_u);
  fem::tensor_basis_apply_interp(m.basis, 2, ContractMode::NoTranspose, e_u, q_u);
  fem::QFunction qf = make_apply_mass_qf();
  std::vector<std::vector<double>> q_out;
  fem::qfunction_apply(qf, 6, {q_u, qdata}, q_out);
  fem::tensor_basis_apply_interp(m.basis, 2, ContractMode::Transpose, q_out[0], e_v);
  fem::elem_restriction_apply(m.rstr, ContractMode::Transpose, e_v, expected);

  std::vector<double> v;
  fem::operator_apply(op, u, v);
  REQUIRE(v.size() == expected.size());
  for (size_t i = 0; i < v.size(); ++i) REQUIRE(v[i] == Approx(expected[i]).margin(1e-14));
}

TEST_CASE("operator_apply: the mass operator is symmetric", "[operator][apply]") {
  Mesh2D m;
  fem::Operator op = m.mass(m.build_qdata());
  const auto u = ramp(6, 1.0), w = ramp(6, -0.7);

  std::vector<double> Mu, Mw;
  fem::operator_apply(op, u, Mu);
  fem::operator_apply(op, w, Mw);
  REQUIRE(dot(w, Mu) == Approx(dot(u, Mw)).margin(1e-12));
}

TEST_CASE("operator_apply overwrites out (resize + zero), it doesn't accumulate", "[operator][apply]") {
  Mesh1D m;
  fem::Operator op = m.mass(m.qdata(1, 1));
  std::vector<double> v(10, 99.0);  // wrong size, garbage
  fem::operator_apply(op, {1, 1, 1}, v);
  REQUIRE(v.size() == 3);
  REQUIRE(v[0] == Approx(0.5));
  REQUIRE(v[1] == Approx(1.0));
  REQUIRE(v[2] == Approx(0.5));
}

TEST_CASE("operator_apply rejects unset fields and a wrong-sized input", "[operator][apply]") {
  Mesh1D m;
  fem::Operator incomplete;
  fem::operator_create(make_apply_mass_qf(), incomplete);
  fem::operator_set_field(incomplete, "u", m.rstr, m.basis, fem::vector_active());
  std::vector<double> v;
  REQUIRE_THROWS_AS(fem::operator_apply(incomplete, {1, 1, 1}, v), std::invalid_argument);

  fem::Operator op = m.mass(m.qdata(1, 1));
  REQUIRE_THROWS_AS(fem::operator_apply(op, {1, 1}, v), std::invalid_argument);
}
