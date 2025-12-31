#include "mpi/dist1d.hpp"
#include "mpi/mpi-vector.hpp"
#include "mpi/halo.hpp"
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  auto d = fem1d::mpi::Dist1D::create(MPI_COMM_WORLD, /*Ne=*/16, /*ghost_width=*/1);

  fem1d::mpi::MpiVector x(d);

  // Fill owned values with their global index (so ghosts are easy to verify)
  for (int i = 0; i < x.n_owned; ++i) {
    x.owned(i) = static_cast<double>(x.global_owned(i));
  }

  // Set ghosts to -1 initially so we can see they update
  for (int i = x.n_owned; i < x.n_local; ++i) x.data[i] = -1.0;

  fem1d::mpi::halo_update(x);

  MPI_Barrier(d.comm);
  for (int r = 0; r < d.size; ++r) {
    if (r == d.rank) {
      std::cout << "Rank " << d.rank << "\n";
      std::cout << "  owned globals: [" << x.g0_owned << ", " << x.g1_owned << "]\n";
      std::cout << "  owned values: ";
      for (int i = 0; i < x.n_owned; ++i) std::cout << x.owned(i) << " ";
      std::cout << "\n";

      if (x.n_ghost_left > 0) {
        std::cout << "  left ghosts:  ";
        for (int k = 0; k < x.n_ghost_left; ++k) std::cout << x.left_ghost_ptr()[k] << " ";
        std::cout << "\n";
      }
      if (x.n_ghost_right > 0) {
        std::cout << "  right ghosts: ";
        for (int k = 0; k < x.n_ghost_right; ++k) std::cout << x.right_ghost_ptr()[k] << " ";
        std::cout << "\n";
      }
      std::cout << std::endl;
    }
    MPI_Barrier(d.comm);
  }

  MPI_Finalize();
  return 0;
}
