#include "mpi/mpi-linear-solver.hpp"
#include "mpi/timer.hpp"
#include <cmath>
#include <cassert>

namespace fem1d::mpi {

extern TimerDB g_timers;

static double dot_owned(const MpiVector& a, const MpiVector& b) {
  assert(a.n_owned == b.n_owned);
  double local = 0.0;
  for (int i = 0; i < a.n_owned; ++i) local += a.data[i] * b.data[i];

  double global = 0.0;
  {
    ScopedTimer t(g_timers, "cg_allreduce_dot");
    MPI_Allreduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, a.dist->comm);
  }
  return global;
}

static void axpy_owned(double alpha, const MpiVector& x, MpiVector& y) {
  assert(x.n_owned == y.n_owned);
  for (int i = 0; i < y.n_owned; ++i) y.data[i] += alpha * x.data[i];
}

static void xpay_owned(const MpiVector& x, double beta, MpiVector& y) {
  // y = x + beta*y
  assert(x.n_owned == y.n_owned);
  for (int i = 0; i < y.n_owned; ++i) y.data[i] = x.data[i] + beta * y.data[i];
}

static void copy_owned(const MpiVector& src, MpiVector& dst) {
  assert(src.n_owned == dst.n_owned);
  for (int i = 0; i < dst.n_owned; ++i) dst.data[i] = src.data[i];
}

CGResult conjugate_gradient(const MpiOperator& A,
                            const MpiVector&  b,
                            MpiVector&        x,
                            const CGOptions&  opts) {
  ScopedTimer t_total(g_timers, "cg_total");
  // Basic checks: same distribution
  assert(x.dist == b.dist);

  MpiVector r(*x.dist), p(*x.dist), Ap(*x.dist);

  // r = b - A x
  {
    ScopedTimer t(g_timers, "cg_apply");
    A.apply(x, Ap);
  }
  for (int i = 0; i < x.n_owned; ++i) r.data[i] = b.data[i] - Ap.data[i];

  // p = r
  copy_owned(r, p);

  const double r0_sq = dot_owned(r, r);
  const double r0 = std::sqrt(r0_sq);

  CGResult result;
  result.iters = 0;
  result.final_res_norm = r0;

  if (r0 == 0.0) return result;

  double rsold = r0_sq;

  for (int k = 0; k < opts.max_iters; ++k) {
    // Ap = A p
    {
      ScopedTimer t(g_timers, "cg_apply");
      A.apply(p, Ap);
    }

    const double pAp = dot_owned(p, Ap);
    // If pAp is zero/negative (breakdown), stop.
    if (pAp <= 0.0) break;

    const double alpha = rsold / pAp;

    // x = x + alpha p
    axpy_owned(alpha, p, x);

    // r = r - alpha Ap
    axpy_owned(-alpha, Ap, r);

    const double rsnew = dot_owned(r, r);
    const double rnorm = std::sqrt(rsnew);

    result.iters = k + 1;
    result.final_res_norm = rnorm;

    if (rnorm <= opts.rtol * r0) break;

    const double beta = rsnew / rsold;

    // p = r + beta p
    xpay_owned(r, beta, p);

    rsold = rsnew;
  }

  return result;
}


CGResult pcg(const MpiOperator& A,
             const Preconditioner& M,
             const MpiVector& b,
             MpiVector& x,
             const CGOptions& opts) {
  ScopedTimer t_total(g_timers, "pcg_total");
  assert(x.dist == b.dist);

  MpiVector r(*x.dist), z(*x.dist), p(*x.dist), Ap(*x.dist);

  // r = b - A x
  {
    ScopedTimer t(g_timers, "pcg_apply");
    A.apply(x, Ap);
  }
  for (int i = 0; i < x.n_owned; ++i) r.data[i] = b.data[i] - Ap.data[i];

  // z = M^{-1} r
  M.apply(r, z);

  // p = z
  copy_owned(z, p);

  const double r0_sq = dot_owned(r, r);
  const double r0    = std::sqrt(r0_sq);

  CGResult result{0, r0};
  if (r0 == 0.0) return result;

  double rzold = dot_owned(r, z);

  // To confirm M is used in Jacobi Jacobi, rz0/r0_sq should be about the scale of inv_diag (\approx O(h)). It won’t be 1.0.
  // For 1D Poisson with Dirichlet, diagonal \approx O(E/h
  if (x.dist->rank == 0) {
  std::cout << "r0=" << r0
            << "  rz0=" << rzold
            << "  (rz0/r0_sq)=" << (rzold / r0_sq) << "\n";
  }

  for (int k = 0; k < opts.max_iters; ++k) {
    {
      ScopedTimer t(g_timers, "pcg_apply");
      A.apply(p, Ap);
    }

    const double pAp = dot_owned(p, Ap);
    if (pAp <= 0.0) break;

    const double alpha = rzold / pAp;

    axpy_owned(alpha, p, x);      // x = x + alpha p
    axpy_owned(-alpha, Ap, r);    // r = r - alpha Ap

    const double rsnew = dot_owned(r, r);
    const double rnorm = std::sqrt(rsnew);

    result.iters = k + 1;
    result.final_res_norm = rnorm;
    if (rnorm <= opts.rtol * r0) break;

    // z = M^{-1} r
    M.apply(r, z);

    const double rznew = dot_owned(r, z);
    const double beta  = rznew / rzold;

    xpay_owned(z, beta, p);       // p = z + beta p
    rzold = rznew;
  }

  return result;
}

} // namespace fem1d::mpi
