#pragma once

namespace fem {

// What a basis is asked to compute, and what a QFunction field expects at
// quadrature points. Mirrors libCEED's CeedEvalMode (ceed/types.h).
//   None:   no basis action; the field is passed through as-is (e.g. qdata)
//   Interp: values at quadrature points
//   Grad:   reference-space gradients at quadrature points
//   Weight: quadrature weights (no input vector)
enum class EvalMode { None, Interp, Grad, Weight };

} // namespace fem
