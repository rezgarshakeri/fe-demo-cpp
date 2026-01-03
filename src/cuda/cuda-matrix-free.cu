#include "cuda/cuda-matrix-free.hpp"
#include "mpi/halo.hpp"          // still CPU halo update for now
#include "mpi/timer.hpp"

#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>

namespace fem1d::cuda {

static inline void cuda_check(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    std::abort();
  }
}

__global__ void poisson1d_apply_owned_kernel(
    const double* __restrict__ nodes,  // global nodes array (size N_global)
    const double* __restrict__ x_local, // local x (owned+ghost)
    double* __restrict__ y_owned,        // owned y (size n_owned)
    int n_owned,
    int g0_owned, int g1_owned,
    int n_ghost_left,
    int left_ghost_begin, int right_ghost_begin,
    int N_global,
    double E)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_owned) return;

  const int g = g0_owned + i; // global row index

  // Dirichlet endpoints as identity row
  if (g == 0 || g == N_global - 1) {
    // local index for g is owned, so it's just i (since g in [g0_owned,g1_owned])
    y_owned[i] = x_local[i];
    return;
  }

  // Map global indices g-1, g, g+1 into local indices in x_local.
  // Local layout:
  // owned: [0..n_owned-1] corresponds to globals [g0_owned..g1_owned]
  // left ghosts: globals [g0_owned-n_ghost_left .. g0_owned-1] at [left_ghost_begin ..]
  // right ghosts: globals [g1_owned+1 .. g1_owned+n_ghost_right] at [right_ghost_begin ..]

  auto local_index = [&](int gg) -> int {
    if (gg >= g0_owned && gg <= g1_owned) return gg - g0_owned;
    // left ghosts
    int gl0 = g0_owned - n_ghost_left;
    int gl1 = g0_owned - 1;
    if (n_ghost_left > 0 && gg >= gl0 && gg <= gl1) {
      return left_ghost_begin + (gg - gl0);
    }
    // right ghosts
    int gr0 = g1_owned + 1;
    int gr1 = g1_owned + ( (right_ghost_begin - (n_owned + n_ghost_left)) + 0 ); // not used
    (void)gr1;
    // We can compute n_ghost_right from (N_global-1 - g1_owned) but safer:
    // We will never index beyond one ghost in 1D P1 if ghost_width=1.
    // We'll map right ghost with right_ghost_begin + (gg - (g1_owned+1)).
    if (gg == g1_owned + 1) {
      return right_ghost_begin;
    }
    if (gg == g0_owned - 1 && n_ghost_left == 1) {
      return left_ghost_begin;
    }
    return -1;
  };

  const int il = local_index(g - 1);
  const int ic = local_index(g);
  const int ir = local_index(g + 1);

  // With ghost_width=1 this must exist
  if (il < 0 || ic < 0 || ir < 0) return;

  const double xm1 = x_local[il];
  const double x0  = x_local[ic];
  const double xp1 = x_local[ir];

  const double hL = nodes[g]   - nodes[g - 1];
  const double hR = nodes[g+1] - nodes[g];

  // row-based SPD stencil consistent with linear FEM:
  // y = E*( (x0-xm1)/hL + (x0-xp1)/hR )
  y_owned[i] = E * ((x0 - xm1) / hL + (x0 - xp1) / hR);
}

MpiMatrixFreePoisson1D_CUDA::MpiMatrixFreePoisson1D_CUDA(const fem1d::Mesh1D& mesh,
                                                         const fem1d::mpi::Dist1D& dist,
                                                         double E)
  : mesh_(mesh), dist_(dist), E_(E) {}

MpiMatrixFreePoisson1D_CUDA::~MpiMatrixFreePoisson1D_CUDA() {
  if (d_x_)     cudaFree(d_x_);
  if (d_y_)     cudaFree(d_y_);
  if (d_nodes_) cudaFree(d_nodes_);
}

void MpiMatrixFreePoisson1D_CUDA::upload_nodes_if_needed() const {
  const int N = static_cast<int>(mesh_.nodes.size());
  if (cap_nodes_ == N && d_nodes_) return;

  if (d_nodes_) cuda_check(cudaFree(d_nodes_), "free d_nodes");
  cuda_check(cudaMalloc((void**)&d_nodes_, sizeof(double) * N), "malloc d_nodes");
  cuda_check(cudaMemcpy(d_nodes_, mesh_.nodes.data(), sizeof(double) * N, cudaMemcpyHostToDevice),
             "copy nodes H2D");
  cap_nodes_ = N;
}

void MpiMatrixFreePoisson1D_CUDA::ensure_device_buffers(const fem1d::mpi::MpiVector& x,
                                                        const fem1d::mpi::MpiVector& y) const {
  if (cap_x_ != x.n_local) {
    if (d_x_) cuda_check(cudaFree(d_x_), "free d_x");
    cuda_check(cudaMalloc((void**)&d_x_, sizeof(double) * x.n_local), "malloc d_x");
    cap_x_ = x.n_local;
  }
  if (cap_y_ != y.n_owned) {
    if (d_y_) cuda_check(cudaFree(d_y_), "free d_y");
    cuda_check(cudaMalloc((void**)&d_y_, sizeof(double) * y.n_owned), "malloc d_y");
    cap_y_ = y.n_owned;
  }
}

void MpiMatrixFreePoisson1D_CUDA::apply(fem1d::mpi::MpiVector& x,
                                       fem1d::mpi::MpiVector& y) const {
  assert(x.dist == &dist_);
  assert(y.dist == &dist_);

  // CPU halo update for now
  fem1d::mpi::halo_update(x);

  // y owned only (GPU computes owned rows)
  // keep y.data sized for consistency with your current code
  y.data.assign(y.n_local, 0.0);

  upload_nodes_if_needed();
  ensure_device_buffers(x, y);

  // Upload x local (owned+ghost)
  cuda_check(cudaMemcpy(d_x_, x.data.data(), sizeof(double) * x.n_local, cudaMemcpyHostToDevice),
             "copy x H2D");

  // Launch kernel
  const int threads = 256;
  const int blocks  = (y.n_owned + threads - 1) / threads;

  poisson1d_apply_owned_kernel<<<blocks, threads>>>(
      d_nodes_, d_x_, d_y_,
      y.n_owned,
      y.g0_owned, y.g1_owned,
      y.n_ghost_left,
      y.left_ghost_begin, y.right_ghost_begin,
      dist_.N_global,
      E_);

  cuda_check(cudaGetLastError(), "kernel launch");
  cuda_check(cudaDeviceSynchronize(), "kernel sync");

  // Download y owned into y.data[0:n_owned]
  cuda_check(cudaMemcpy(y.data.data(), d_y_, sizeof(double) * y.n_owned, cudaMemcpyDeviceToHost),
             "copy y D2H");

  // Dirichlet endpoints handled in kernel already (identity).
}

} // namespace fem1d::cuda
