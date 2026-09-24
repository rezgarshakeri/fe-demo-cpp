
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <stdexcept>
#include "qfunction.hpp"
#include "qfunctions/mass.hpp"

using Catch::Approx;

// ---------------------------------------------------------------------
// Bookkeeping: create, add fields, context
// ---------------------------------------------------------------------

namespace {
int noop(void*, int, const double* const*, double* const*) { return 0; }
}  // namespace

TEST_CASE("qfunction_create/add_input/add_output record fields in order", "[qfunction]") {
  fem::QFunction qf;
  fem::qfunction_create(fem::qfunctions::apply_mass, qf);
  fem::qfunction_add_input(qf, "u", 1, fem::EvalMode::Interp);
  fem::qfunction_add_input(qf, "qdata", 1, fem::EvalMode::None);
  fem::qfunction_add_output(qf, "v", 1, fem::EvalMode::Interp);

  REQUIRE(qf.user == &fem::qfunctions::apply_mass);
  REQUIRE(qf.inputs.size() == 2);
  REQUIRE(qf.inputs[0].name == "u");
  REQUIRE(qf.inputs[0].eval_mode == fem::EvalMode::Interp);
  REQUIRE(qf.inputs[1].name == "qdata");
  REQUIRE(qf.inputs[1].eval_mode == fem::EvalMode::None);
  REQUIRE(qf.outputs.size() == 1);
  REQUIRE(qf.outputs[0].name == "v");
  REQUIRE(qf.ctx.empty());
}

TEST_CASE("qfunction_create/add reject invalid arguments", "[qfunction]") {
  fem::QFunction qf;
  REQUIRE_THROWS_AS(fem::qfunction_create(nullptr, qf), std::invalid_argument);

  fem::qfunction_create(noop, qf);
  REQUIRE_THROWS_AS(fem::qfunction_add_input(qf, "u", 0, fem::EvalMode::Interp), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::qfunction_add_input(qf, "w", 2, fem::EvalMode::Weight), std::invalid_argument);
  REQUIRE_THROWS_AS(fem::qfunction_add_output(qf, "w", 1, fem::EvalMode::Weight), std::invalid_argument);

  fem::qfunction_add_input(qf, "u", 1, fem::EvalMode::Interp);
  REQUIRE_THROWS_AS(fem::qfunction_add_input(qf, "u", 1, fem::EvalMode::Grad), std::invalid_argument);
  // names are unique across inputs and outputs too (an Operator sets fields by name)
  REQUIRE_THROWS_AS(fem::qfunction_add_output(qf, "u", 1, fem::EvalMode::Interp), std::invalid_argument);
  REQUIRE_NOTHROW(fem::qfunction_add_output(qf, "v", 1, fem::EvalMode::Interp));
  REQUIRE_THROWS_AS(fem::qfunction_add_input(qf, "v", 1, fem::EvalMode::Interp), std::invalid_argument);
}

TEST_CASE("qfunction_create resets a previously used QFunction", "[qfunction]") {
  fem::QFunction qf;
  fem::qfunction_create(noop, qf);
  fem::qfunction_add_input(qf, "u", 1, fem::EvalMode::Interp);
  fem::qfunction_set_context(qf, fem::qfunctions::BuildMassContext{3});

  fem::qfunction_create(fem::qfunctions::apply_mass, qf);
  REQUIRE(qf.inputs.empty());
  REQUIRE(qf.outputs.empty());
  REQUIRE(qf.ctx.empty());
}

TEST_CASE("qfunction_set_context stores a copy, not a pointer", "[qfunction]") {
  fem::QFunction qf;
  fem::qfunction_create(noop, qf);

  fem::qfunctions::BuildMassContext ctx{2};
  fem::qfunction_set_context(qf, ctx);
  ctx.dim = 99;  // changing the original afterwards must not affect the QFunction

  REQUIRE(qf.ctx.size() == sizeof(fem::qfunctions::BuildMassContext));
  fem::qfunctions::BuildMassContext stored;
  std::memcpy(&stored, qf.ctx.data(), sizeof(stored));
  REQUIRE(stored.dim == 2);
}

// ---------------------------------------------------------------------
// qfunction_apply: data layout [size][Q], sizes, errors
// ---------------------------------------------------------------------

TEST_CASE("qfunction_apply: apply_mass computes v = qdata * u pointwise", "[qfunction][apply]") {
  fem::QFunction qf;
  fem::qfunction_create(fem::qfunctions::apply_mass, qf);
  fem::qfunction_add_input(qf, "u", 1, fem::EvalMode::Interp);
  fem::qfunction_add_input(qf, "qdata", 1, fem::EvalMode::None);
  fem::qfunction_add_output(qf, "v", 1, fem::EvalMode::Interp);

  const int Q = 5;
  const std::vector<std::vector<double>> in = {{1, 2, 3, 4, 5}, {0.5, -1, 2, 0, 10}};
  std::vector<std::vector<double>> out;
  fem::qfunction_apply(qf, Q, in, out);

  REQUIRE(out.size() == 1);
  REQUIRE(out[0] == std::vector<double>{0.5, -2, 6, 0, 50});
}

TEST_CASE("qfunction_apply: build_mass computes det(J) * w in 1D, 2D, 3D", "[qfunction][apply]") {
  // Two points per call. J is laid out [d][c][p]: J[(d * dim + c) * Q + p] = d x_c / d X_d.
  const int Q = 2;
  const std::vector<double> w = {0.5, 2.0};

  struct Case {
    int dim;
    std::vector<double> J;  // [d][c][p]
    double det0, det1;      // det(J) at point 0 and point 1
  };
  const std::vector<Case> cases = {
      {1, {3.0, -2.0}, 3.0, -2.0},
      // p0: [[2,1],[0,3]] det 6;  p1: [[1,2],[3,4]] det -2
      {2, {2, 1,   1, 2,     // d=0: (c=0: p0,p1), (c=1: p0,p1)
           0, 3,   3, 4},    // d=1
       6.0, -2.0},
      // p0: diag(1,2,3) det 6;  p1: [[2,0,1],[1,3,0],[0,1,4]] det 2*(12-0) - 0 + 1*(1-0) = 25
      {3, {1, 2,  0, 0,  0, 1,     // d=0: c=0,1,2
           0, 1,  2, 3,  0, 0,     // d=1
           0, 0,  0, 1,  3, 4},    // d=2
       6.0, 25.0},
  };

  for (const Case& c : cases) {
    fem::QFunction qf;
    fem::qfunction_create(fem::qfunctions::build_mass, qf);
    fem::qfunction_add_input(qf, "dx", c.dim * c.dim, fem::EvalMode::Grad);
    fem::qfunction_add_input(qf, "weights", 1, fem::EvalMode::Weight);
    fem::qfunction_add_output(qf, "qdata", 1, fem::EvalMode::None);
    fem::qfunction_set_context(qf, fem::qfunctions::BuildMassContext{c.dim});

    std::vector<std::vector<double>> out;
    fem::qfunction_apply(qf, Q, {c.J, w}, out);

    INFO("dim = " << c.dim);
    REQUIRE(out[0].size() == static_cast<size_t>(Q));
    REQUIRE(out[0][0] == Approx(c.det0 * w[0]));
    REQUIRE(out[0][1] == Approx(c.det1 * w[1]));
  }
}

namespace {
// Two fields of size 2, to pin down the [size][Q] layout: out = (u0 + u1, u0 - u1) per point.
int sum_diff(void*, int Q, const double* const* in, double* const* out) {
  for (int p = 0; p < Q; p++) {
    const double a = in[0][0 * Q + p], b = in[0][1 * Q + p];
    out[0][0 * Q + p] = a + b;
    out[0][1 * Q + p] = a - b;
  }
  return 0;
}
int check_null_ctx(void* ctx, int Q, const double* const*, double* const* out) {
  for (int p = 0; p < Q; p++) out[0][p] = (ctx == nullptr) ? 1.0 : 0.0;
  return 0;
}
int fail(void*, int, const double* const*, double* const*) { return 3; }
}  // namespace

TEST_CASE("qfunction_apply: multi-component fields are [size][Q], component-major", "[qfunction][apply]") {
  fem::QFunction qf;
  fem::qfunction_create(sum_diff, qf);
  fem::qfunction_add_input(qf, "u", 2, fem::EvalMode::Interp);
  fem::qfunction_add_output(qf, "v", 2, fem::EvalMode::Interp);

  const int Q = 3;
  // component 0 = {1, 2, 3}, component 1 = {10, 20, 30}
  std::vector<std::vector<double>> out;
  fem::qfunction_apply(qf, Q, {{1, 2, 3, 10, 20, 30}}, out);
  REQUIRE(out[0] == std::vector<double>{11, 22, 33, -9, -18, -27});
}

TEST_CASE("qfunction_apply: outputs are resized and overwritten", "[qfunction][apply]") {
  fem::QFunction qf;
  fem::qfunction_create(fem::qfunctions::apply_mass, qf);
  fem::qfunction_add_input(qf, "u", 1, fem::EvalMode::Interp);
  fem::qfunction_add_input(qf, "qdata", 1, fem::EvalMode::None);
  fem::qfunction_add_output(qf, "v", 1, fem::EvalMode::Interp);

  std::vector<std::vector<double>> out = {{99, 99, 99, 99, 99, 99, 99}, {99}};  // wrong count and size
  fem::qfunction_apply(qf, 2, {{1, 2}, {3, 4}}, out);
  REQUIRE(out.size() == 1);
  REQUIRE(out[0] == std::vector<double>{3, 8});
}

TEST_CASE("qfunction_apply: passes a null ctx when no context is set", "[qfunction][apply]") {
  fem::QFunction qf;
  fem::qfunction_create(check_null_ctx, qf);
  fem::qfunction_add_output(qf, "flag", 1, fem::EvalMode::None);

  std::vector<std::vector<double>> out;
  fem::qfunction_apply(qf, 2, {}, out);
  REQUIRE(out[0] == std::vector<double>{1, 1});
}

TEST_CASE("qfunction_apply: rejects wrong field counts and sizes", "[qfunction][apply]") {
  fem::QFunction qf;
  fem::qfunction_create(fem::qfunctions::apply_mass, qf);
  fem::qfunction_add_input(qf, "u", 1, fem::EvalMode::Interp);
  fem::qfunction_add_input(qf, "qdata", 1, fem::EvalMode::None);
  fem::qfunction_add_output(qf, "v", 1, fem::EvalMode::Interp);

  std::vector<std::vector<double>> out;
  REQUIRE_THROWS_AS(fem::qfunction_apply(qf, 2, {{1, 2}}, out), std::invalid_argument);                  // 1 of 2 inputs
  REQUIRE_THROWS_AS(fem::qfunction_apply(qf, 2, {{1, 2}, {3, 4}, {5, 6}}, out), std::invalid_argument);  // 3 inputs
  REQUIRE_THROWS_AS(fem::qfunction_apply(qf, 2, {{1, 2}, {3}}, out), std::invalid_argument);             // short input
  REQUIRE_THROWS_AS(fem::qfunction_apply(qf, 2, {{1, 2, 3}, {3, 4}}, out), std::invalid_argument);       // long input
  REQUIRE_THROWS_AS(fem::qfunction_apply(qf, 0, {{}, {}}, out), std::invalid_argument);                  // Q = 0
}

TEST_CASE("qfunction_apply: a non-zero return from the user function throws", "[qfunction][apply]") {
  fem::QFunction qf;
  fem::qfunction_create(fail, qf);
  std::vector<std::vector<double>> out;
  REQUIRE_THROWS_AS(fem::qfunction_apply(qf, 1, {}, out), std::runtime_error);
}