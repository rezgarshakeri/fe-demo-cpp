#include "linear-solver.hpp"
#include <cmath>
#include <cassert>

namespace fem1d {

static double dot(const std::vector<double>& a, const std::vector<double>& b) {
  assert(a.size() == b.size());
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
  return s;
}

static void axpy(double alpha, const std::vector<double>& x, std::vector<double>& y) {
  assert(x.size() == y.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] += alpha * x[i];
}

static void scal(double alpha, std::vector<double>& x) {
  for (double &xi : x) xi *= alpha;
}

CGResult conjugate_gradient(const Operator& A,
                            const std::vector<double>& b,
                            std::vector<double>&       x,
                            const CGOptions&           opts) {
  const size_t n = b.size();
  assert(x.size() == n);

  std::vector<double> r(n), p(n), Ap(n);

  // r = b - A x
  A.apply(x, Ap);
  for (size_t i = 0; i < n; ++i) r[i] = b[i] - Ap[i];

  double r0_norm = std::sqrt(dot(r, r));
  if (r0_norm == 0.0) return {0, 0.0};

  p = r;
  double rr = dot(r, r);

  double r_norm = std::sqrt(rr);
  int k = 0;

  for (; k < opts.max_iters; ++k) {
    A.apply(p, Ap);
    double pAp = dot(p, Ap);

    if (pAp == 0.0) break;

    double alpha = rr / pAp;

    // x = x + alpha p
    axpy(alpha, p, x);

    // r = r - alpha Ap
    axpy(-alpha, Ap, r);

    double rr_new = dot(r, r);
    r_norm = std::sqrt(rr_new);

    if (r_norm <= opts.rtol * r0_norm) {
      rr = rr_new;
      break;
    }

    double beta = rr_new / rr;

    // p = r + beta p
    scal(beta, p);
    axpy(1.0, r, p);

    rr = rr_new;
  }

  return {k + 1, r_norm};
}

} // namespace fem1d
