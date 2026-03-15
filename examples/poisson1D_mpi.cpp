#include "mesh.hpp"
#include "matrix-free-operator.hpp"
#include "linear-solver.hpp"

#include "mpi/dist1d.hpp"
#include "mpi/mpi-vector.hpp"
#include "mpi/mpi-matrix-free.hpp"
#include "mpi/mpi-linear-solver.hpp"
#include "mpi/gather.hpp"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

#include "mpi/timer.hpp"
#include "mpi/jacobi-poisson1d.hpp"

namespace fem1d::mpi { extern TimerDB g_timers; }

static double l2_norm_sq(const std::vector<double>& a) {
  double s = 0.0;
  for (double v : a) s += v*v;
  return s;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int    Ne = 1000;
  double x0 = 0.0, x1 = 1.0;
  double E  = 1.0;

  auto mesh = fem1d::Mesh1D::build_uniform(Ne, x0, x1);
  auto dist = fem1d::mpi::Dist1D::create(MPI_COMM_WORLD, mesh.num_elements, /*ghost_width=*/1);

  fem1d::mpi::MpiVector b(dist, 0.0, 0.0);
  fem1d::mpi::MpiVector x(dist, 0.0, 0.0);

  // RHS: simple test, e.g., b=1 interior, 0 on Dirichlet endpoints
  const double h = (x1 - x0) / Ne;
  for (int i = 0; i < b.n_owned; ++i) {
    const int g = b.global_owned(i);
    b.owned(i) = h;
    if (g == 0 || g == dist.N_global - 1) b.owned(i) = 0.0;
  }

  fem1d::mpi::MpiMatrixFreePoisson1D A(mesh, dist, E);

  fem1d::CGOptions opts;
  opts.max_iters = 5000; // CG iteration is \approx O(Ne)
  opts.rtol = 1e-6;

  // auto res = fem1d::mpi::conjugate_gradient(A, b, x, opts);
  fem1d::mpi::JacobiPoisson1D M(mesh, x, E);
  auto res = fem1d::mpi::pcg(A, M, b, x, opts);

  if (dist.rank == 0) {
    std::cout << "MPI PCG: iters=" << res.iters
              << ", residual=" << res.final_res_norm << "\n";
  }

  // Optional: gather solution to root for inspection
  std::vector<double> x_global;
  fem1d::mpi::gather_to_root_global(x, x_global, 0);

  if (dist.rank == 0) {
    fem1d::MatrixFreePoisson1D A_ser(mesh, E);
    std::vector<double> b_ser(mesh.nodes.size(), 1.0), x_ser(mesh.nodes.size(), 0.0);
    b_ser[0] = 0.0; b_ser.back() = 0.0;
  
    auto res_ser = fem1d::conjugate_gradient(A_ser, b_ser, x_ser, opts);
    x_ser[0] = 0.0; x_ser.back() = 0.0;
  
    double err = 0.0;
    for (size_t i = 0; i < x_ser.size(); ++i) {
      double d = x_global[i] - x_ser[i];
      err += d*d;
    }
    std::cout << "||x_mpi - x_serial||_2^2 = " << err << "\n";
  }

  fem1d::mpi::g_timers.print_rank0(dist.comm, dist.rank);

  MPI_Finalize();
  return 0;
}
