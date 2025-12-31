#include "mpi/dist1d.hpp"
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  auto d = fem1d::mpi::Dist1D::create(MPI_COMM_WORLD, /*Ne_global=*/16, /*ghost_width=*/1);

  MPI_Barrier(MPI_COMM_WORLD);
  for (int r = 0; r < d.size; ++r) {
    if (r == d.rank) {
      std::cout
        << "Rank " << d.rank << "/" << d.size << "\n"
        << "  elements: [" << d.e0 << ", " << d.e1 << ")  (Ne_owned=" << d.Ne_owned << ")\n"
        << "  nodes:    [" << d.i0 << ", " << d.i1 << "]  (n_owned=" << d.n_owned << ")\n"
        << "  ghosts L/R: " << d.n_ghost_left << " / " << d.n_ghost_right
        << "  (n_local=" << d.n_local << ")\n";

      if (d.n_ghost_left > 0) {
        std::cout << "  left ghost globals: ";
        for (int k = 0; k < d.n_ghost_left; ++k) std::cout << d.left_ghost_global(k) << " ";
        std::cout << "\n";
      }
      if (d.n_ghost_right > 0) {
        std::cout << "  right ghost globals: ";
        for (int k = 0; k < d.n_ghost_right; ++k) std::cout << d.right_ghost_global(k) << " ";
        std::cout << "\n";
      }
      std::cout << std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}
