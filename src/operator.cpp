#include "operator.hpp"
#include <stdexcept>
#include "tensor-contract.hpp"

namespace fem {

static int int_pow(int base, int exp) {
  int result = 1;
  for (int i = 0; i < exp; ++i) result *= base;
  return result;
}

/**
  @brief Create an `Operator` from a `QFunction` whose fields are already added

  @param[in]  qf The pointwise kernel; copied into the operator
  @param[out] op The `Operator` to fill in, with every field still unset

  @ref CeedOperatorCreate (interface/ceed-operator.c)
**/
void operator_create(const QFunction& qf, Operator& op) {
  if (!qf.user) throw std::invalid_argument("operator_create: QFunction has no user function");
  op = Operator{};
  op.qf = qf;
  op.inputs.resize(qf.inputs.size());
  op.outputs.resize(qf.outputs.size());
}

static std::invalid_argument field_error(const std::string& name, const std::string& msg) {
  return std::invalid_argument("operator_set_field '" + name + "': " + msg);
}

/**
  @brief Attach a restriction, basis and vector to a `QFunction` field

  @param[in,out] op          The `Operator`
  @param[in]     name        Name of an input or output field of op.qf
  @param[in]     restriction Maps the field's L-vector to/from an E-vector; std::nullopt for Weight
  @param[in]     basis       Maps E-vector to/from quadrature points; std::nullopt for None
  @param[in]     vector      vector_active(), vector_passive(data) or vector_none()

  @ref CeedOperatorSetField, CeedOperatorCheckField (interface/ceed-operator.c)
**/
void operator_set_field(Operator& op, const std::string& name, std::optional<ElemRestriction> restriction,
                        std::optional<TensorBasis> basis, FieldVector vector) {
  // Find the QFunction field (names are unique across inputs and outputs).
  const QFunctionField* qf_field = nullptr;
  std::optional<OperatorField>* slot = nullptr;
  bool is_input = true;
  for (size_t f = 0; f < op.qf.inputs.size() && !qf_field; ++f) {
    if (op.qf.inputs[f].name == name) qf_field = &op.qf.inputs[f], slot = &op.inputs[f];
  }
  for (size_t f = 0; f < op.qf.outputs.size() && !qf_field; ++f) {
    if (op.qf.outputs[f].name == name) qf_field = &op.qf.outputs[f], slot = &op.outputs[f], is_input = false;
  }
  if (!qf_field) throw field_error(name, "the QFunction has no field with this name");
  if (slot->has_value()) throw field_error(name, "already set");

  const EvalMode mode = qf_field->eval_mode;
  const int size = qf_field->size;

  // Which of restriction / basis / vector each EvalMode needs.
  switch (mode) {
    case EvalMode::Weight:
      if (restriction) throw field_error(name, "EvalMode::Weight takes no restriction");
      if (!basis) throw field_error(name, "EvalMode::Weight needs a basis (for its quadrature weights)");
      if (vector.kind != VectorKind::None) throw field_error(name, "EvalMode::Weight takes vector_none()");
      break;
    case EvalMode::None:
      if (!restriction) throw field_error(name, "EvalMode::None needs a restriction");
      if (basis) throw field_error(name, "EvalMode::None takes no basis");
      if (restriction->num_comp != size) throw field_error(name, "restriction num_comp must equal the field size");
      break;
    case EvalMode::Interp:
    case EvalMode::Grad: {
      if (!restriction) throw field_error(name, "EvalMode::Interp/Grad needs a restriction");
      if (!basis) throw field_error(name, "EvalMode::Interp/Grad needs a basis");
      const int expected = (mode == EvalMode::Interp) ? basis->num_comp : basis->num_comp * basis->dim;
      if (size != expected) {
        throw field_error(name, mode == EvalMode::Interp ? "field size must equal basis num_comp"
                                                         : "field size must equal basis num_comp * dim");
      }
      if (restriction->num_comp != basis->num_comp) throw field_error(name, "restriction and basis num_comp differ");
      if (restriction->elem_size != int_pow(basis->P_1d, basis->dim)) {
        throw field_error(name, "restriction elem_size must equal basis P_1d^dim");
      }
      break;
    }
  }
  if (mode != EvalMode::Weight && vector.kind == VectorKind::None) {
    throw field_error(name, "needs vector_active() or vector_passive(...)");
  }
  if (!is_input && vector.kind != VectorKind::Active) throw field_error(name, "output fields must be active");
  if (vector.kind == VectorKind::Passive && static_cast<int>(vector.data.size()) != restriction->l_size) {
    throw field_error(name, "passive vector size must equal restriction l_size");
  }

  // Consistency with fields already set.
  const int num_qpts = basis ? int_pow(basis->Q_1d, basis->dim) : restriction->elem_size;
  if (op.num_qpts != 0 && num_qpts != op.num_qpts) {
    throw field_error(name, "number of quadrature points differs from previously set fields");
  }
  if (restriction && op.num_elem != 0 && restriction->num_elem != op.num_elem) {
    throw field_error(name, "num_elem differs from previously set fields");
  }
  if (vector.kind == VectorKind::Active) {
    int& active_l_size = is_input ? op.active_in_l_size : op.active_out_l_size;
    if (active_l_size != 0 && restriction->l_size != active_l_size) {
      throw field_error(name, "active fields share one vector, so their restrictions need the same l_size");
    }
    active_l_size = restriction->l_size;
  }

  op.num_qpts = num_qpts;
  if (restriction) op.num_elem = restriction->num_elem;
  *slot = OperatorField{name, mode, size, std::move(restriction), std::move(basis), std::move(vector)};
}

/**
  @brief Check that an `Operator` is ready to apply: every field set, num_elem known

  @ref CeedOperatorCheckReady (interface/ceed-operator.c)
**/
void operator_check_ready(const Operator& op) {
  for (const auto* fields : {&op.inputs, &op.outputs}) {
    for (size_t f = 0; f < fields->size(); ++f) {
      if (!(*fields)[f].has_value()) {
        const auto& qf_fields = (fields == &op.inputs) ? op.qf.inputs : op.qf.outputs;
        throw std::invalid_argument("operator_check_ready: field '" + qf_fields[f].name + "' is not set");
      }
    }
  }
  if (op.num_elem == 0) throw std::invalid_argument("operator_check_ready: no field has a restriction");
}

/**
  @brief Apply the operator: out = A(in)

  @param[in,out] op  The `Operator` (non-const because the QFunction context is passed as void*)
  @param[in]     in  Active input L-vector, active_in_l_size entries (ignored if no input is active)
  @param[out]    out Active output L-vector: resized to active_out_l_size, zeroed, then summed into

  @ref CeedOperatorApply (interface/ceed-operator.c), CeedOperatorApplyAdd_Ref
       (backends/ref/ceed-ref-operator.c) -- but all elements batched in one pass, not one at a time
**/
void operator_apply(Operator& op, const std::vector<double>& in, std::vector<double>& out) {

  operator_check_ready(op);
  if (op.active_in_l_size > 0 && static_cast<int>(in.size()) != op.active_in_l_size) {
    throw std::invalid_argument("operator_apply: input vector size does not match active_in_l_size");
  }
  int Q = op.num_qpts * op.num_elem;
  
  std::vector<std::vector<double>> q_in(op.qf.inputs.size()), q_out;
  std::vector<double> e;  // E-vector scratch, reused by every field

  for (size_t f = 0; f < op.qf.inputs.size(); ++f) {
    const OperatorField& fld = *op.inputs[f];
    const std::vector<double>& vec = (fld.vector.kind == VectorKind::Active) ? in : fld.vector.data;
    if (fld.eval_mode == EvalMode::Weight) {
      tensor_basis_apply_weight(*fld.basis, op.num_elem, fem::ContractMode::NoTranspose, q_in[f]);
    } else if (fld.eval_mode == EvalMode::None) {
      elem_restriction_apply(*fld.restriction, fem::ContractMode::NoTranspose, vec, q_in[f]);
    } else {
      elem_restriction_apply(*fld.restriction, fem::ContractMode::NoTranspose, vec, e);
      tensor_basis_apply(*fld.basis, op.num_elem, fem::ContractMode::NoTranspose, fld.eval_mode, e, q_in[f]);
    }
  }
  qfunction_apply(op.qf, Q, q_in, q_out);
  out.assign(op.active_out_l_size, 0.0);
  for (size_t f = 0; f < op.qf.outputs.size(); ++f) {
    const OperatorField& fld = *op.outputs[f];
    if (fld.eval_mode == EvalMode::None) {
      elem_restriction_apply(*fld.restriction, fem::ContractMode::Transpose, q_out[f], out);
    } else {
      tensor_basis_apply(*fld.basis, op.num_elem, fem::ContractMode::Transpose, fld.eval_mode, q_out[f], e);
      elem_restriction_apply(*fld.restriction, fem::ContractMode::Transpose, e, out);  // scatter-add
    }
  }
}

} // namespace fem
