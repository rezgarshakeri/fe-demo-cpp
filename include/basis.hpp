#pragma once
#include <vector>

namespace fem {

// 1D Gauss-Legendre quadrature on the reference interval [-1, 1].
void gauss_legendre_quadrature(int Q, std::vector<double>& q_ref_1d, std::vector<double>& q_weight_1d);
// 1D Gauss-Lobatto points on [-1, 1]: the two endpoints -1, 1 plus the
// interior roots of P'_{Q-1}. q_weight_1d is optional
void gauss_lobatto_points(int Q, std::vector<double>& q_ref_1d,
                           std::vector<double>* q_weight_1d = nullptr);

// Evaluate all P Lagrange basis functions built on `nodes` (and their derivatives) at every point in `eval_points`, using Fornberg's (1998)
void lagrange_basis_matrix(const std::vector<double>& nodes, const std::vector<double>& eval_points, std::vector<double>& interp_1d, std::vector<double>& grad_1d);

// Which 1D rule to use for the Q_1d quadrature points in TensorBasis.
// Node placement (P_1d) is always Gauss-Lobatto, independent of this.
//
//   Gauss          -- Gauss-Legendre, exact for degree <= 2*Q_1d - 1.
//   GaussLobatto   -- Gauss-Lobatto, exact for degree <= 2*Q_1d - 3. When
//                     P_1d == Q_1d, quadrature points coincide with the
//                     nodes, so interp_1d becomes the identity matrix
enum class QuadMode { Gauss, GaussLobatto };

// A tensor-product H1 Lagrange basis on the dim-dimensional reference cube [-1, 1]^dim,
// These 1D matrices are all that's needed to evaluate the basis in 1D
// directly, and (via tensor contraction along each dimension) in 2D/3D
struct TensorBasis {
  int dim;    // spatial dimension (1, 2, or 3)
  int P_1d;   // nodes per dimension
  int Q_1d;   // quadrature points per dimension

  std::vector<double> q_ref_1d;     // size Q_1d
  std::vector<double> q_weight_1d;  // size Q_1d
  std::vector<double> interp_1d;    // size Q_1d * P_1d
  std::vector<double> grad_1d;      // size Q_1d * P_1d

  static TensorBasis create_tensor_H1_lagrange(int dim, int P_1d, int Q_1d, QuadMode quad_mode = QuadMode::Gauss);
};

} // namespace fem
