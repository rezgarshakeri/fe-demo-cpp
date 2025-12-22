#include "mesh.hpp"

namespace fem1d {

Mesh1D Mesh1D::build_uniform(int Ne, double x0_in, double x1_in) {
    Mesh1D mesh;
    mesh.num_elements = Ne;
    mesh.x0 = x0_in;
    mesh.x1 = x1_in;
    int num_nodes = Ne + 1;

    mesh.nodes.resize(num_nodes);
    double h = (x1_in - x0_in) / Ne;
    for (int i = 0; i < num_nodes; ++i) {
        mesh.nodes[i] = x0_in + i * h;
    }

    mesh.elem_nodes.resize(2 * Ne);
    for (int e = 0; e < Ne; ++e) {
        mesh.elem_nodes[2*e + 0] = e;
        mesh.elem_nodes[2*e + 1] = e + 1;
    }

    return mesh;
}

} // namespace fem1d
