#include "mpi/gather.hpp"
#include <cassert>

namespace fem1d::mpi {

void gather_to_root_global(const MpiVector& v, std::vector<double>& global_out, int root) {
  const auto& d = *v.dist;

  // Gather each rank's [g0_owned, g1_owned] and n_owned
  struct Range { int g0, g1, n; };

  Range local{v.g0_owned, v.g1_owned, v.n_owned};
  std::vector<Range> all;
  if (d.rank == root) all.resize(d.size);

  MPI_Gather(&local, sizeof(Range)/sizeof(int), MPI_INT,
             d.rank == root ? all.data() : nullptr, sizeof(Range)/sizeof(int), MPI_INT,
             root, d.comm);

  // Gather the owned values as variable-length blocks
  std::vector<int> counts, displs;
  std::vector<double> recvbuf;
  if (d.rank == root) {
    counts.resize(d.size);
    displs.resize(d.size);
    int total = 0;
    for (int r = 0; r < d.size; ++r) {
      counts[r] = all[r].n;
      displs[r] = total;
      total += counts[r];
    }
    recvbuf.resize(total);
    global_out.assign(d.N_global, 0.0);
  }

  MPI_Gatherv(v.data.data(), v.n_owned, MPI_DOUBLE,
              d.rank == root ? recvbuf.data() : nullptr,
              d.rank == root ? counts.data() : nullptr,
              d.rank == root ? displs.data() : nullptr,
              MPI_DOUBLE, root, d.comm);

  if (d.rank == root) {
    // Scatter the received concatenated blocks into global_out using [g0,g1]
    for (int r = 0; r < d.size; ++r) {
      const int n = all[r].n;
      if (n == 0) continue;
      const int g0 = all[r].g0;
      const int g1 = all[r].g1;
      assert(g1 - g0 + 1 == n);

      const int off = displs[r];
      for (int i = 0; i < n; ++i) {
        global_out[g0 + i] = recvbuf[off + i];
      }
    }
  }
}

} // namespace fem1d::mpi
