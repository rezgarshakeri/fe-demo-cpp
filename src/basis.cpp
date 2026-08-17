#include "basis.hpp"
#include <stdexcept>
#include <cmath>

namespace fem {

/**
  @brief Construct a Gauss-Legendre quadrature

  @param[in]  Q           Number of quadrature points (integrates polynomials of degree `2*Q-1` exactly)
  @param[out] q_ref_1d    Resized to length `Q`, holds the abscissa on `[-1, 1]`
  @param[out] q_weight_1d Resized to length `Q`, holds the weights

  @throw std::invalid_argument if `Q <= 0`
**/
void gauss_legendre_quadrature(int Q,
                                std::vector<double>& q_ref_1d,
                                std::vector<double>& q_weight_1d) {
  if (Q <= 0) {
    throw std::invalid_argument("gauss_legendre_quadrature: Q must be positive");
  }

  double P0, P1, P2, dP2, xi, wi;
  q_ref_1d.resize(Q);
  q_weight_1d.resize(Q);

  const double PI = 4.0 * atan(1.0);
  const double EPSILON = 10 * std::numeric_limits<double>::epsilon();

  // Newton iteration to find roots of P_Q(x) (Legendre polynomial of degree Q)
  // and compute weights from the roots.
  for (int i = 0; i <= Q / 2; i++) {
    // Guess initial root location (Chebyshev nodes)
    xi = std::cos(PI * (2.0 * i + 1.0) / (2.0 * Q));
    // Evaluate P_Q(xi) and its derivative using recurrence relation
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (int j = 2; j <= Q; j++) {
      P2 = ((2.0 * j - 1.0) * xi * P1 - (j - 1.0) * P0) / j;
      P0 = P1;
      P1 = P2;
    }
    // First Newton Step
    dP2 = (xi * P2 - P0) * Q / (xi * xi - 1.0);
    xi  = xi - P2 / dP2;
    // Newton's method to refine root
    for (int k = 0; k < 100 && std::fabs(P2) > EPSILON; k++) {
      // Re-evaluate P_Q(xi)
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j <= Q; j++) {
        P2 = ((2.0 * j - 1.0) * xi * P1 - (j - 1.0) * P0) / j;
        P0 = P1;
        P1 = P2;
      }
      // Compute derivative P'_Q(xi)
      dP2 = (xi * P2 - P0) * Q / (xi * xi - 1.0);
      xi -= P2 / dP2;
    }
    // Store the root and its weight
    wi = 2.0 / ((1.0 - xi * xi) * dP2 * dP2);
    q_weight_1d[i] = wi;
    q_weight_1d[Q - 1 - i] = wi;
    q_ref_1d[i] = -xi;
    q_ref_1d[Q - 1 - i] = xi;
  }
}


/**
  @brief Construct a Gauss-Legendre-Lobatto quadrature

  @param[in]  Q           Number of points (endpoints -1, 1 plus interior roots of P'_{Q-1})
  @param[out] q_ref_1d    Resized to length `Q`, holds the abscissa on `[-1, 1]`, sorted ascending
  @param[out] q_weight_1d Optional; if non-null, resized to length `Q` and filled with the weights.
                          Pass nullptr (the default) when only node placement is needed.

  @throw std::invalid_argument if `Q <= 1`
**/
void gauss_lobatto_points(int Q, std::vector<double>& q_ref_1d,
                           std::vector<double>* q_weight_1d) {
  if (Q <= 1) {
    throw std::invalid_argument("Cannot create Lobatto quadrature with Q <= 1");
  }

  double P0, P1, P2, dP2, d2P2, xi, wi;
  q_ref_1d.resize(Q);
  if (q_weight_1d) q_weight_1d->resize(Q);

  const double PI = 4.0 * atan(1.0);
  const double EPSILON = 10 * std::numeric_limits<double>::epsilon();

  // Set endpoints
  q_ref_1d[0] = -1.0;
  q_ref_1d[Q - 1] = 1.0;
  if (q_weight_1d) {
    wi = 2.0 / (Q * (Q - 1));
    (*q_weight_1d)[0] = wi;
    (*q_weight_1d)[Q - 1] = wi;
  }

  // Interior points
  for (int i = 1; i <= (Q - 1) / 2; i++) {
    // Initial guess for the root
    xi = std::cos(PI * i / (Q - 1));
    // Evaluate P_{Q-1}(xi) and its derivative using recurrence relation
    P0 = 1.0;
    P1 = xi;
    P2 = 0.0;
    for (int j = 2; j < Q; j++) {
      P2 = ((2.0 * j - 1.0) * xi * P1 - (j - 1.0) * P0) / j;
      P0 = P1;
      P1 = P2;
    }
    // First Newton Step
    dP2 = (xi * P2 - P0) * Q / (xi * xi - 1.0);
    d2P2 = (2 * xi * dP2 - (Q * (Q - 1)) * P2) / (1.0 - xi * xi);
    xi -= dP2 / d2P2;
    // Newton's method to refine root
    for (int k = 0; k < 100 && std::fabs(dP2) > EPSILON; k++) {
      // Re-evaluate P_{Q-1}(xi)
      P0 = 1.0;
      P1 = xi;
      for (int j = 2; j < Q; j++) {
        P2 = ((2.0 * j - 1.0) * xi * P1 - (j - 1.0) * P0) / j;
        P0 = P1;
        P1 = P2;
      }
      // Compute derivative
      dP2 = (xi * P2 - P0) * Q / (xi * xi - 1.0);
      d2P2 = (2 * xi * dP2 - (Q * (Q - 1)) * P2) / (1.0 - xi * xi);
      xi -= dP2 / d2P2;
    }
    // Store the root and its weight
    q_ref_1d[i] = -xi;
    q_ref_1d[Q - 1 - i] = xi;
    if (q_weight_1d) {
      wi = 2.0 / ((Q * (Q - 1)) * P2 * P2);
      (*q_weight_1d)[i] = wi;
      (*q_weight_1d)[Q - 1 - i] = wi;
    }
  }
}

/**
  @brief Build Lagrange basis interpolation and gradient matrices, via Fornberg's (1998) algorithm

  @param[in]  nodes       Node locations defining the Lagrange basis, length `P`
  @param[in]  eval_points Points to evaluate the basis (and its derivative) at, length `M`
  @param[out] interp_1d   Resized to `M * P`, row-major: interp_1d[i*P+j] = L_j(eval_points[i])
  @param[out] grad_1d     Resized to `M * P`, row-major: grad_1d[i*P+j] = L_j'(eval_points[i])

  @ref libCEED's CeedBasisCreateTensorH1Lagrange in ceed-basis.c
**/
void lagrange_basis_matrix(const std::vector<double>& nodes,
                            const std::vector<double>& eval_points,
                            std::vector<double>& interp_1d,
                            std::vector<double>& grad_1d) {
  size_t P = nodes.size();
  size_t M = eval_points.size(); // Q in libCEED
  interp_1d.resize(M * P);
  grad_1d.resize(M * P);
  double c1, c2, c3, c4, dx;
  for (int i = 0; i < M; i++) {
    c1                   = 1.0;
    c3                   = nodes[0] - eval_points[i];
    interp_1d[i * P + 0] = 1.0;
    for (int j = 1; j < P; j++) {
      c2 = 1.0;
      c4 = c3;
      c3 = nodes[j] - eval_points[i];
      for (int k = 0; k < j; k++) {
        dx = nodes[j] - nodes[k];
        c2 *= dx;
        if (k == j - 1) {
          grad_1d[i * P + j]   = c1 * (interp_1d[i * P + k] - c4 * grad_1d[i * P + k]) / c2;
          interp_1d[i * P + j] = -c1 * c4 * interp_1d[i * P + k] / c2;
        }
        grad_1d[i * P + k]   = (c3 * grad_1d[i * P + k] - interp_1d[i * P + k]) / dx;
        interp_1d[i * P + k] = c3 * interp_1d[i * P + k] / dx;
      }
      c1 = c2;
    }
  }
}

/**
  @brief Construct a tensor-product H1 Lagrange TensorBasis

  @param[in] dim       Spatial dimension (1, 2, or 3)
  @param[in] num_comp  Number of components (1, 2, or 3)
  @param[in] P_1d      Nodes per dimension (Gauss-Lobatto node placement), must be >= 2
  @param[in] Q_1d      Quadrature points per dimension, must be >= 1
  @param[in] quad_mode Quadrature rule for q_ref_1d/q_weight_1d: QuadMode::Gauss (default)
                       or QuadMode::GaussLobatto

  @return The constructed TensorBasis (interp_1d/grad_1d evaluated at the Q_1d quadrature points)

  @throw std::invalid_argument if dim, P_1d, or Q_1d are out of range
**/
TensorBasis TensorBasis::create_tensor_H1_lagrange(int dim, int num_comp, int P_1d, int Q_1d,
                                                     QuadMode quad_mode) {
  if (dim < 1 || dim > 3) {
    throw std::invalid_argument("create_tensor_H1_lagrange: dim must be 1, 2, or 3");
  }
  if (num_comp < 1 || num_comp > 3) {
    throw std::invalid_argument("create_tensor_H1_lagrange: num_comp must be 1, 2, or 3");
  }
  if (P_1d < 2) {
    throw std::invalid_argument("create_tensor_H1_lagrange: P_1d must be >= 2");
  }
  if (Q_1d < 1) {
    throw std::invalid_argument("create_tensor_H1_lagrange: Q_1d must be >= 1");
  }
  std::vector<double> nodes;
  gauss_lobatto_points(P_1d, nodes);  // weights not needed for node placement
  std::vector<double> q_ref_1d, q_weight_1d, interp_1d, grad_1d;
  if (quad_mode == QuadMode::Gauss) {
    gauss_legendre_quadrature(Q_1d, q_ref_1d, q_weight_1d);
  } else if (quad_mode == QuadMode::GaussLobatto) {
    gauss_lobatto_points(Q_1d, q_ref_1d, &q_weight_1d);
  } else {
    throw std::invalid_argument("create_tensor_H1_lagrange: unknown quad_mode");
  }
  lagrange_basis_matrix(nodes, q_ref_1d, interp_1d, grad_1d);
  return TensorBasis{dim, num_comp, P_1d, Q_1d, q_ref_1d, q_weight_1d, interp_1d, grad_1d};
}

} // namespace fem
