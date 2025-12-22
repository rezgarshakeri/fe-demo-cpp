#pragma once
#include <vector>

namespace fem1d {

struct Mesh1D {
    int    num_elements;  // Ne
    double x0, x1;        // domain [x0, x1]
    std::vector<double> nodes;       // size = num_nodes
    std::vector<int>    elem_nodes;  // size = 2 * num_elements (linear elems)

    static Mesh1D build_uniform(int num_elements, double x0, double x1);
};

} // namespace fem1d
