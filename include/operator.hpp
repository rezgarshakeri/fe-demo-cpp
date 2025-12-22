#pragma once
#include <vector>

namespace fem1d {

class Operator {
public:
    virtual ~Operator() = default;

    // y = A x
    virtual void apply(const std::vector<double>& x,
                       std::vector<double>&       y) const = 0;

    // optional: assemble global sparse matrix (if available)
    virtual bool can_assemble() const { return false; }
};

} // namespace fem1d
