#include "mesh.hpp"
#include "matrix-free-operator.hpp"

#include "mpi/dist1d.hpp"
#include "mpi/mpi-vector.hpp"
#include "mpi/mpi-matrix-free.hpp"
#include "mpi/gather.hpp"

#include <mpi.h>
#include <iostream>
#include <vector>
#include <cmath>

static double l2_norm_sq_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    const double d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);

  // Build full mesh on every rank (simple for now)
  auto mesh = fem1d::Mesh1D::build_uniform(16, 0.0, 1.0);

  auto dist = fem1d::mpi::Dist1D::create(MPI_COMM_WORLD, mesh.num_elements, /*ghost_width=*/1);

  fem1d::mpi::MpiVector x(dist), y(dist);

  // Fill x with global index values
  for (int i = 0; i < x.n_owned; ++i) x.owned(i) = static_cast<double>(x.global_owned(i));
  for (int i = x.n_owned; i < x.n_local; ++i) x.data[i] = -1.0;

  fem1d::mpi::MpiMatrixFreePoisson1D A_mpi(mesh, dist, /*E=*/1.0);
  A_mpi.apply(x, y);

  // Gather y to root
  std::vector<double> y_global;
  fem1d::mpi::gather_to_root_global(y, y_global, /*root=*/0);

  if (dist.rank == 0) {
    // Serial reference
    fem1d::MatrixFreePoisson1D A_serial(mesh, /*E=*/1.0);

    std::vector<double> x_ser(mesh.nodes.size()), y_ser;
    for (int i = 0; i < (int)x_ser.size(); ++i) x_ser[i] = static_cast<double>(i);

    A_serial.apply(x_ser, y_ser);

    const double err = l2_norm_sq_diff(y_global, y_ser);
    std::cout << "||y_mpi - y_serial||_2^2 = " << err << "\n";

    // Optional: print a few entries if mismatch
    if (err > 1e-12) {
      std::cout << "Mismatch detected. First few entries:\n";
      for (int i = 0; i < std::min<int>(10, (int)y_ser.size()); ++i) {
        std::cout << "i=" << i << " mpi=" << y_global[i] << " ser=" << y_ser[i] << "\n";
      }
      // Hard fail
      MPI_Abort(MPI_COMM_WORLD, 2);
    } else {
      std::cout << "MPI apply matches serial.\n";
    }
  }

  MPI_Finalize();
  return 0;
}
