#pragma once
#include "mpi/mpi-operator.hpp"
#include "mpi/mpi-vector.hpp"
#include "mesh.hpp"

namespace fem1d::cuda {

// GPU-backed matrix-free Poisson apply for the same MPI distribution.
// For now: halo update stays on CPU (host), we upload x, run kernel, download y_owned.
class MpiMatrixFreePoisson1D_CUDA : public fem1d::mpi::MpiOperator {
public:
  MpiMatrixFreePoisson1D_CUDA(const fem1d::Mesh1D& mesh,
                              const fem1d::mpi::Dist1D& dist,
                              double E);

  ~MpiMatrixFreePoisson1D_CUDA();

  void apply(fem1d::mpi::MpiVector& x, fem1d::mpi::MpiVector& y) const override;

private:
  const fem1d::Mesh1D& mesh_;
  const fem1d::mpi::Dist1D& dist_;
  double E_;

  // Device storage (mutable because apply updates buffers)
  mutable double* d_x_ = nullptr;   // size = x.n_local
  mutable double* d_y_ = nullptr;   // size = y.n_owned
  mutable double* d_nodes_ = nullptr; // mesh nodes, size = N_global

  mutable int cap_x_ = 0;
  mutable int cap_y_ = 0;
  mutable int cap_nodes_ = 0;

  void ensure_device_buffers(const fem1d::mpi::MpiVector& x,
                             const fem1d::mpi::MpiVector& y) const;

  void upload_nodes_if_needed() const;
};

} // namespace fem1d::cuda
