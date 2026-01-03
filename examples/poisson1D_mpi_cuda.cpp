#include <mpi.h>
#include <iostream>

#include "mesh.hpp"
#include "mpi/dist1d.hpp"
#include "mpi/mpi-vector.hpp"
#include "mpi/mpi-linear-solver.hpp"
#include "cuda/cuda-matrix-free.hpp"

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  int rank=0, size=1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int    Ne = 1000;
  double x0 = 0.0, x1 = 1.0;
  double E  = 1.0;

  auto mesh = fem1d::Mesh1D::build_uniform(Ne, x0, x1);
  auto dist = fem1d::mpi::Dist1D::create(MPI_COMM_WORLD, mesh.num_elements, /*ghost_width=*/1);

  fem1d::mpi::MpiVector b(dist, 0.0, 0.0);
  fem1d::mpi::MpiVector x(dist, 0.0, 0.0);

  // RHS: f=1, integrated -> scale by h for P1 lump-ish
  const double h = (x1 - x0) / Ne;
  for (int i = 0; i < b.n_owned; ++i) {
    const int g = b.global_owned(i);
    b.owned(i) = (g==0 || g==dist.N_global-1) ? 0.0 : h;
  }

  fem1d::cuda::MpiMatrixFreePoisson1D_CUDA A(mesh, dist, E);

  fem1d::CGOptions opts;
  opts.max_iters = 5000;
  opts.rtol = 1e-6;

  auto res = fem1d::mpi::conjugate_gradient(A, b, x, opts);

  if (rank == 0) {
    std::cout << "MPI CG (CUDA apply): iters=" << res.iters
              << ", residual=" << res.final_res_norm << "\n";
  }

  MPI_Finalize();
  return 0;
}
