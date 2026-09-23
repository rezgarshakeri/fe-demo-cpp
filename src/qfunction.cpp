#include "qfunction.hpp"
#include <stdexcept>

namespace fem {

/**
  @brief Create a `QFunction` around a user function, with no fields and no context yet

  @param[in]  user The pointwise kernel (see `QFunctionUser` in qfunction.hpp)
  @param[out] qf   The `QFunction` to fill in

  @ref CeedQFunctionCreateInterior (interface/ceed-qfunction.c)
**/
void qfunction_create(QFunctionUser user, QFunction& qf) {
    if (!user) {
        throw std::invalid_argument("QFunction user function pointer is null");
    }
    // Initialize the QFunction with the user function and empty fields/context
    qf = QFunction{};
    qf.user = user;
}

static void check_field(const std::vector<QFunctionField>& fields, const std::string& name, int size,
                        EvalMode eval_mode, const char* who) {
  if (size <= 0) throw std::invalid_argument(std::string(who) + ": size must be positive");
  if (eval_mode == EvalMode::Weight && size != 1) {
    throw std::invalid_argument(std::string(who) + ": an EvalMode::Weight field must have size 1");
  }
  for (const QFunctionField& f : fields) {
    if (f.name == name) throw std::invalid_argument(std::string(who) + ": duplicate field name '" + name + "'");
  }
}

/**
  @brief Add an input field; field index = order of addition (in[0], in[1], ...)

  @ref CeedQFunctionAddInput (interface/ceed-qfunction.c)
**/
void qfunction_add_input(QFunction& qf, const std::string& name, int size, EvalMode eval_mode) {
  check_field(qf.inputs, name, size, eval_mode, "qfunction_add_input");
  qf.inputs.push_back({name, size, eval_mode});
}

/**
  @brief Add an output field; field index = order of addition (out[0], out[1], ...)

  @ref CeedQFunctionAddOutput (interface/ceed-qfunction.c)
**/
void qfunction_add_output(QFunction& qf, const std::string& name, int size, EvalMode eval_mode) {
  if (eval_mode == EvalMode::Weight) {
    throw std::invalid_argument("qfunction_add_output: EvalMode::Weight is input-only");
  }
  check_field(qf.outputs, name, size, eval_mode, "qfunction_add_output");
  qf.outputs.push_back({name, size, eval_mode});
}

/**
  @brief Run the user function on Q quadrature points

  @param[in]  qf  The `QFunction`
  @param[in]  Q   Number of points in this call (Q_1d^dim * num_elem from an Operator)
  @param[in]  in  One array per input field, in[f] of size inputs[f].size * Q, layout [size][Q]
  @param[out] out One array per output field, resized to outputs[f].size * Q, layout [size][Q]

  @ref CeedQFunctionApply (interface/ceed-qfunction.c)
**/
void qfunction_apply(QFunction& qf, int Q, const std::vector<std::vector<double>>& in,
                     std::vector<std::vector<double>>& out) {
  if (Q <= 0) {
    throw std::invalid_argument("qfunction_apply: Q must be positive");
  }
  if (in.size() != qf.inputs.size()) {
    throw std::invalid_argument("qfunction_apply: number of input fields does not match QFunction");
  }
  for (size_t f = 0; f < in.size(); ++f) {
    if (in[f].size() != static_cast<size_t>(qf.inputs[f].size * Q)) {
      throw std::invalid_argument("qfunction_apply: input field " + qf.inputs[f].name + " has incorrect size");
    }
  }
  for (size_t f = 0; f < out.size(); ++f) {
    if (out[f].size() != static_cast<size_t>(qf.outputs[f].size * Q)) {
      throw std::invalid_argument("qfunction_apply: output field " + qf.outputs[f].name + " has incorrect size");
    }
  }
  std::vector<const double*> in_ptrs(in.size());
  std::vector<double*> out_ptrs(out.size());
  for (size_t f = 0; f < in.size(); ++f) {
    in_ptrs[f] = in[f].data();
  }
  for (size_t f = 0; f < out.size(); ++f) {
    out_ptrs[f] = out[f].data();
  }
  void* ctx_ptr = qf.ctx.empty() ? nullptr : qf.ctx.data();
  int ierr = qf.user(ctx_ptr, Q, in_ptrs.data(), out_ptrs.data());
  if (ierr != 0) {
    throw std::runtime_error("qfunction_apply: user function returned non-zero error code");
  }
}

} // namespace fem