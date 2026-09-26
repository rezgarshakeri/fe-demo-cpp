#pragma once
#include <optional>
#include <string>
#include <vector>
#include "basis.hpp"
#include "elem-restriction.hpp"
#include "qfunction.hpp"

namespace fem {

// Where a field's L-vector comes from:
//   Active:  the in / out vector passed to operator_apply
//   Passive: a fixed vector stored (copied) in the operator, e.g. qdata
//   None:    no vector (EvalMode::Weight inputs)
enum class VectorKind { Active, Passive, None };

struct FieldVector {
  VectorKind kind = VectorKind::None;
  std::vector<double> data;  // Passive only
};
inline FieldVector vector_active() { return {VectorKind::Active, {}}; }
inline FieldVector vector_none() { return {VectorKind::None, {}}; }
inline FieldVector vector_passive(std::vector<double> data) { return {VectorKind::Passive, std::move(data)}; }

struct OperatorField {
  std::string name;
  EvalMode eval_mode;  // copied from the QFunction field
  int size;            // copied from the QFunction field
  std::optional<ElemRestriction> restriction;  // empty for Weight
  std::optional<TensorBasis> basis;            // empty for None
  FieldVector vector;
};

// Mirrors libCEED's CeedOperator (interface/ceed-operator.c). Everything is
// stored by value (the QFunction, restrictions, bases, passive vectors): no
// dangling pointers, and it's what a GPU backend needs (its own copies).
// A passive vector is copied at operator_set_field, so compute it first.
struct Operator {
  QFunction qf;
  std::vector<std::optional<OperatorField>> inputs;   // same order as qf.inputs; empty until set
  std::vector<std::optional<OperatorField>> outputs;  // same order as qf.outputs; empty until set

  int num_elem = 0;           // from the fields' restrictions (0 = not yet known)
  int num_qpts = 0;           // quadrature points per element (0 = not yet known)
  int active_in_l_size = 0;   // l_size of the active input fields' restrictions (0 = none)
  int active_out_l_size = 0;  // l_size of the active output fields' restrictions (0 = none)
};

void operator_create(const QFunction& qf, Operator& op);

// Attach a restriction, basis and vector to the QFunction field called `name`.
// Pass std::nullopt for "no restriction" / "no basis". Throws if the name is
// unknown or already set, or if restriction/basis/vector don't fit the field's
// EvalMode and size, or disagree with fields already set (num_elem, number of
// quadrature points, active l_size).
void operator_set_field(Operator& op, const std::string& name, std::optional<ElemRestriction> restriction,
                        std::optional<TensorBasis> basis, FieldVector vector);

// Throws unless every field is set and num_elem is known.
void operator_check_ready(const Operator& op);

// out = A(in). `in` must have active_in_l_size entries (it is ignored if no
// input is active); out is resized to active_out_l_size, zeroed, then summed into.
void operator_apply(Operator& op, const std::vector<double>& in, std::vector<double>& out);

} // namespace fem
