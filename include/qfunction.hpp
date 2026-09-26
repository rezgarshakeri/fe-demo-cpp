#pragma once
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>
#include "eval-mode.hpp"

namespace fem {

// The pointwise physics kernel, called on a batch of Q quadrature points.
// in[f] / out[f] point to field f's data, laid out [size][Q]: component c of
// point p is in[f][c * Q + p]. When called from an Operator, Q is
// Q_1d^dim * num_elem and p = q * num_elem + e -- exactly the Q-vector layout
// tensor_basis_apply_* and backend-strided restrictions produce, so no copy.
// Return 0 on success.
//
// A plain function pointer (not std::function / a capturing lambda) on
// purpose: libCEED compiles the same QFunction source for CPU and GPU, and a
// function with all its state in ctx is what can be ported to a CUDA kernel.
using QFunctionUser = int (*)(void* ctx, int Q, const double* const* in, double* const* out);

struct QFunctionField {
  std::string name;
  int size;            // components per point: num_comp (Interp/None), num_comp*dim (Grad), 1 (Weight)
  EvalMode eval_mode;
};

// Mirrors libCEED's CeedQFunction (interface/ceed-qfunction.c).
struct QFunction {
  QFunctionUser user = nullptr;
  std::vector<QFunctionField> inputs;
  std::vector<QFunctionField> outputs;
  std::vector<unsigned char> ctx;  // user context data (libCEED's CeedQFunctionContext), copied to device if needed
};

void qfunction_create(QFunctionUser user, QFunction& qf);
void qfunction_add_input(QFunction& qf, const std::string& name, int size, EvalMode eval_mode);
void qfunction_add_output(QFunction& qf, const std::string& name, int size, EvalMode eval_mode);

/**
  @brief Store a byte copy of the user context; the user function receives it as `void* ctx`

  @param[in,out] qf   The `QFunction`
  @param[in]     data Context struct; must be trivially copyable (plain data, no pointers to host memory)
                      so it can be byte-copied, e.g. to a GPU

  @ref CeedQFunctionSetContext (interface/ceed-qfunction.c)
**/
template <class T>
void qfunction_set_context(QFunction& qf, const T& data) {
  static_assert(std::is_trivially_copyable_v<T>, "QFunction context must be trivially copyable");
  qf.ctx.resize(sizeof(T));
  std::memcpy(qf.ctx.data(), &data, sizeof(T));
}

// in[f] must have exactly inputs[f].size * Q entries; out[f] is resized to
// outputs[f].size * Q. The user function is expected to write every output
// entry. Throws on a wrong number of fields or wrong input sizes, or if the
// user function returns non-zero.
void qfunction_apply(QFunction& qf, int Q, const std::vector<std::vector<double>>& in,
                     std::vector<std::vector<double>>& out);


}  // namespace fem