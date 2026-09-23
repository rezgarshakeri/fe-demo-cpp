#pragma once
// Mass-operator QFunctions, ported from libCEED's examples/ceed/ex1-volume.h.
// These are "user" code: plain functions matching fem::QFunctionUser, with
// all state passed through ctx.

namespace fem::qfunctions {

struct BuildMassContext {
  int dim;  // reference dimension == physical dimension here
};

// qdata = det(J) * w at each point.
//   in[0]: "dx", size dim*dim, EvalMode::Grad of the mesh coordinates;
//          in[0][(d * dim + c) * Q + p] = d x_c / d X_d  (row d = reference direction)
//   in[1]: "weights", size 1, EvalMode::Weight
//   out[0]: "qdata", size 1, EvalMode::None
// det(J) = det(J^T), so the row/column convention doesn't matter here (it will for Laplace).
inline int build_mass(void* ctx, int Q, const double* const* in, double* const* out) {
  const int dim = static_cast<const BuildMassContext*>(ctx)->dim;
  const double* J = in[0];
  const double* w = in[1];
  double* qdata = out[0];
  auto j = [&](int d, int c, int p) { return J[(d * dim + c) * Q + p]; };

  for (int p = 0; p < Q; p++) {
    double det = 0.0;
    switch (dim) {
      case 1: det = j(0, 0, p); break;
      case 2: det = j(0, 0, p) * j(1, 1, p) - j(0, 1, p) * j(1, 0, p); break;
      case 3:
        det = j(0, 0, p) * (j(1, 1, p) * j(2, 2, p) - j(1, 2, p) * j(2, 1, p)) -
              j(0, 1, p) * (j(1, 0, p) * j(2, 2, p) - j(1, 2, p) * j(2, 0, p)) +
              j(0, 2, p) * (j(1, 0, p) * j(2, 1, p) - j(1, 1, p) * j(2, 0, p));
        break;
      default: return 1;
    }
    qdata[p] = det * w[p];
  }
  return 0;
}

// v = qdata * u at each point.
//   in[0]: "u", size 1, EvalMode::Interp
//   in[1]: "qdata", size 1, EvalMode::None
//   out[0]: "v", size 1, EvalMode::Interp
inline int apply_mass(void* /*ctx*/, int Q, const double* const* in, double* const* out) {
  const double* u = in[0];
  const double* qdata = in[1];
  double* v = out[0];
  for (int p = 0; p < Q; p++) v[p] = qdata[p] * u[p];
  return 0;
}

} // namespace fem::qfunctions