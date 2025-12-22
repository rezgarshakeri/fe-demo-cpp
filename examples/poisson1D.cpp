#include "mesh.hpp"
#include "assembled-operator.hpp"
#include "matrix-free-operator.hpp"
#include "linear-solver.hpp"
#include "utils.hpp"
#include <iostream>

using namespace fem1d;

int main() {
    int    Ne = 1000;
    double x0 = 0.0, x1 = 1.0;
    double E  = 1.0;

    Mesh1D mesh = Mesh1D::build_uniform(Ne, x0, x1);
    int num_nodes = mesh.nodes.size();

    std::vector<double> b(num_nodes, 0.0);
    // build RHS f(x) ≈ 1, simple lumped load:
    for (int i = 0; i < num_nodes; ++i) {
        b[i] = 1.0; // TODO: scale properly by h, etc.
    }

    b[0] = 0.0;
    b.back() = 0.0; // b[n - 1] = 0.0;

    // Assembled version
    AssembledPoisson1D A_assembled(mesh, E);
    std::vector<double> x_asm(num_nodes, 0.0);
    auto t0 = now();           // from utils.hpp
    auto res_asm = conjugate_gradient(A_assembled, b, x_asm);
    auto t1 = now();
    x_asm[0] = 0.0;   x_asm.back() = 0.0;

    std::cout << "Assembled CG: iters=" << res_asm.iters
              << ", residual=" << res_asm.final_res_norm
              << ", time=" << elapsed_seconds(t0, t1) << "\n";

    // Matrix-free version
    MatrixFreePoisson1D A_mf(mesh, E);
    std::vector<double> x_mf(num_nodes, 0.0);
    auto t2 = now();
    auto res_mf = conjugate_gradient(A_mf, b, x_mf);
    auto t3 = now();
    x_mf[0]  = 0.0;   x_mf.back()  = 0.0;

    std::cout << "Matrix-free CG: iters=" << res_mf.iters
              << ", residual=" << res_mf.final_res_norm
              << ", time=" << elapsed_seconds(t2, t3) << "\n";

    // compare solutions
    double err = 0.0;
    for (int i = 0; i < num_nodes; ++i) {
        double diff = x_asm[i] - x_mf[i];
        err += diff * diff;
    }
    std::cout << "||x_asm - x_mf||_2^2 = " << err << "\n";

    return 0;
}
