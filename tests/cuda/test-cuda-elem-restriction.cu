#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include "cuda/device-array.hpp"
#include "cuda/cuda-elem-restriction.hpp"
#include "elem-restriction.hpp"

using Catch::Approx;
using fem::ContractMode;
using fem::cuda::DeviceArray;
using fem::cuda::ScatterMode;

namespace {

// Structured mesh of nx*ny*nz hexahedra (dim = 3; ny = nz = 1 gives a 1D line
// of elements), P nodes per direction per element, lexicographic local nodes.
// Offsets are node * node_stride; components at + k * comp_stride.
//   blocked layout:     node_stride = 1,        comp_stride = num_nodes
//   interleaved layout: node_stride = num_comp, comp_stride = 1
fem::ElemRestriction box_restriction(int nx, int ny, int nz, int P, int num_comp, bool interleaved) {
  const int Nx = nx * (P - 1) + 1, Ny = ny * (P - 1) + 1, Nz = nz * (P - 1) + 1;
  const int num_nodes = Nx * Ny * Nz;
  const int node_stride = interleaved ? num_comp : 1;
  const int comp_stride = interleaved ? 1 : num_nodes;
  std::vector<int> offsets;
  for (int ez = 0; ez < nz; ++ez)
    for (int ey = 0; ey < ny; ++ey)
      for (int ex = 0; ex < nx; ++ex)
        for (int iz = 0; iz < P; ++iz)
          for (int iy = 0; iy < P; ++iy)
            for (int ix = 0; ix < P; ++ix) {
              const int gx = ex * (P - 1) + ix, gy = ey * (P - 1) + iy, gz = ez * (P - 1) + iz;
              offsets.push_back((gx + Nx * (gy + Ny * gz)) * node_stride);
            }
  fem::ElemRestriction r;
  fem::elem_restriction_create(nx * ny * nz, P * P * P, num_comp, comp_stride, num_nodes * num_comp, offsets, r);
  return r;
}

// Values whose sums round differently depending on the order of the additions.
std::vector<double> wiggly(int n, double phase) {
  std::vector<double> v(n);
  for (int i = 0; i < n; ++i) v[i] = std::sin(0.37 * i + phase) + 1e-3 * std::cos(11.0 * i);
  return v;
}

void require_close(const std::vector<double>& gpu, const std::vector<double>& cpu, double margin = 1e-13) {
  REQUIRE(gpu.size() == cpu.size());
  for (size_t i = 0; i < cpu.size(); ++i) {
    INFO("entry " << i);
    REQUIRE(gpu[i] == Approx(cpu[i]).margin(margin));
  }
}

// GPU gather/scatter vs the CPU version, for one restriction and scatter mode.
void check_against_cpu(const fem::ElemRestriction& host, ScatterMode mode) {
  fem::cuda::ElemRestriction dev;
  fem::cuda::elem_restriction_create(host, mode, dev);

  // Gather: E = P L
  const std::vector<double> l_in = wiggly(host.l_size, 0.1);
  std::vector<double> e_cpu;
  fem::elem_restriction_apply(host, ContractMode::NoTranspose, l_in, e_cpu);
  DeviceArray<double> d_l(l_in), d_e;
  fem::cuda::elem_restriction_apply(dev, ContractMode::NoTranspose, d_l, d_e);
  require_close(d_e.to_host(), e_cpu, 0.0);  // a gather only copies: exact

  // Scatter-add: L = 1 + P^T E  (out starts at 1 to check it adds, not overwrites)
  const std::vector<double> e_in = wiggly(host.e_size, 0.7);
  std::vector<double> l_cpu(host.l_size, 1.0);
  fem::elem_restriction_apply(host, ContractMode::Transpose, e_in, l_cpu);
  DeviceArray<double> d_e_in(e_in), d_l_out(std::vector<double>(host.l_size, 1.0));
  fem::cuda::elem_restriction_apply(dev, ContractMode::Transpose, d_e_in, d_l_out);
  require_close(d_l_out.to_host(), l_cpu);
}

}  // namespace

// ---------------------------------------------------------------------
// build_transpose_map (host only)
// ---------------------------------------------------------------------

TEST_CASE("build_transpose_map inverts the offsets", "[cuda][elem-restriction][transpose-map]") {
  SECTION("two 1D elements sharing node 1") {
    fem::ElemRestriction r;
    fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2}, r);
    fem::cuda::TransposeMap map;
    fem::cuda::build_transpose_map(r, map);
    REQUIRE(map.l_vec_indices == std::vector<int>{0, 1, 2});
    REQUIRE(map.t_offsets == std::vector<int>{0, 1, 3, 4});
    REQUIRE(map.t_indices == std::vector<int>{0, 1, 2, 3});
  }
  SECTION("unused L-nodes are skipped, rows are sorted") {
    // l_size 6, node 0 and 3 never used; element 1 lists its nodes in reverse
    fem::ElemRestriction r;
    fem::elem_restriction_create(2, 2, 1, 1, 6, {2, 4, 5, 2}, r);
    fem::cuda::TransposeMap map;
    fem::cuda::build_transpose_map(r, map);
    REQUIRE(map.l_vec_indices == std::vector<int>{2, 4, 5});
    REQUIRE(map.t_offsets == std::vector<int>{0, 2, 3, 4});
    REQUIRE(map.t_indices == std::vector<int>{0, 3, 1, 2});  // node 2 <- E-nodes 0 and 3
  }
  SECTION("every E-node appears exactly once, under its own L-node") {
    const auto r = box_restriction(3, 2, 2, 3, 1, false);
    fem::cuda::TransposeMap map;
    fem::cuda::build_transpose_map(r, map);
    REQUIRE(map.t_offsets.front() == 0);
    REQUIRE(map.t_offsets.back() == r.num_elem * r.elem_size);
    std::vector<int> seen(r.num_elem * r.elem_size, 0);
    for (size_t n = 0; n < map.l_vec_indices.size(); ++n) {
      for (int t = map.t_offsets[n]; t < map.t_offsets[n + 1]; ++t) {
        REQUIRE(r.offsets[map.t_indices[t]] == map.l_vec_indices[n]);
        ++seen[map.t_indices[t]];
      }
    }
    for (int count : seen) REQUIRE(count == 1);
  }
  SECTION("strided restrictions have no map") {
    fem::ElemRestriction r;
    fem::elem_restriction_create_strided(2, 3, 1, 6, r);
    fem::cuda::TransposeMap map;
    REQUIRE_THROWS_AS(fem::cuda::build_transpose_map(r, map), std::invalid_argument);
  }
}

// ---------------------------------------------------------------------
// create and apply argument checks
// ---------------------------------------------------------------------

TEST_CASE("cuda::elem_restriction_create copies sizes and index arrays", "[cuda][elem-restriction]") {
  const auto host = box_restriction(2, 2, 1, 2, 3, true);
  fem::cuda::ElemRestriction dev;
  fem::cuda::elem_restriction_create(host, ScatterMode::Deterministic, dev);
  REQUIRE(dev.num_elem == host.num_elem);
  REQUIRE(dev.elem_size == host.elem_size);
  REQUIRE(dev.num_comp == 3);
  REQUIRE(dev.comp_stride == 1);
  REQUIRE(dev.l_size == host.l_size);
  REQUIRE(dev.e_size == host.e_size);
  REQUIRE(dev.offsets.to_host() == host.offsets);

  fem::cuda::TransposeMap map;
  fem::cuda::build_transpose_map(host, map);
  REQUIRE(dev.num_nodes == static_cast<int>(map.l_vec_indices.size()));
  REQUIRE(dev.l_vec_indices.to_host() == map.l_vec_indices);
  REQUIRE(dev.t_offsets.to_host() == map.t_offsets);
  REQUIRE(dev.t_indices.to_host() == map.t_indices);

  fem::cuda::ElemRestriction atomic;  // no transpose map for Atomic
  fem::cuda::elem_restriction_create(host, ScatterMode::Atomic, atomic);
  REQUIRE(atomic.t_indices.empty());
}

TEST_CASE("cuda::elem_restriction_apply checks sizes", "[cuda][elem-restriction]") {
  fem::ElemRestriction host;
  fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2}, host);
  fem::cuda::ElemRestriction dev;
  fem::cuda::elem_restriction_create(host, ScatterMode::Atomic, dev);

  DeviceArray<double> out;
  REQUIRE_THROWS_AS(fem::cuda::elem_restriction_apply(dev, ContractMode::NoTranspose, DeviceArray<double>(4), out),
                    std::invalid_argument);  // in must be l_size = 3
  REQUIRE_THROWS_AS(fem::cuda::elem_restriction_apply(dev, ContractMode::Transpose, DeviceArray<double>(3), out),
                    std::invalid_argument);  // in must be e_size = 4
  REQUIRE_THROWS_AS(fem::cuda::elem_restriction_apply(dev, ContractMode::Transpose, DeviceArray<double>(4), out),
                    std::invalid_argument);  // out must already be l_size = 3
}

// ---------------------------------------------------------------------
// Kernels vs the CPU elem_restriction_apply
// ---------------------------------------------------------------------

TEST_CASE("cuda offset restriction matches the CPU: small cases", "[cuda][elem-restriction][kernels]") {
  const ScatterMode mode = GENERATE(ScatterMode::Atomic, ScatterMode::Deterministic);
  INFO("scatter mode " << (mode == ScatterMode::Atomic ? "Atomic" : "Deterministic"));

  SECTION("1D, two elements sharing a node") {
    fem::ElemRestriction r;
    fem::elem_restriction_create(2, 2, 1, 1, 3, {0, 1, 1, 2}, r);
    check_against_cpu(r, mode);
  }
  SECTION("unused L-nodes keep their value") {
    fem::ElemRestriction r;
    fem::elem_restriction_create(2, 2, 1, 1, 6, {2, 4, 5, 2}, r);
    check_against_cpu(r, mode);
  }
  SECTION("2D quads, 2 components, blocked layout") {
    fem::ElemRestriction r;
    fem::elem_restriction_create(2, 4, 2, 6, 12, {0, 1, 3, 4, 1, 2, 4, 5}, r);
    check_against_cpu(r, mode);
  }
}

TEST_CASE("cuda offset restriction matches the CPU: 3D meshes", "[cuda][elem-restriction][kernels]") {
  const ScatterMode mode = GENERATE(ScatterMode::Atomic, ScatterMode::Deterministic);
  const bool interleaved = GENERATE(false, true);
  INFO("scatter mode " << (mode == ScatterMode::Atomic ? "Atomic" : "Deterministic") << ", interleaved "
                       << interleaved);

  SECTION("linear hexes, 1 component") { check_against_cpu(box_restriction(4, 3, 2, 2, 1, interleaved), mode); }
  SECTION("quadratic hexes, 3 components") { check_against_cpu(box_restriction(3, 3, 3, 3, 3, interleaved), mode); }
  SECTION("larger: 20^3 quadratic hexes, 216k E-nodes, many blocks") {
    check_against_cpu(box_restriction(20, 20, 20, 3, 1, interleaved), mode);
  }
}

TEST_CASE("cuda deterministic scatter is bitwise reproducible", "[cuda][elem-restriction][kernels]") {
  const auto host = box_restriction(16, 16, 16, 3, 3, true);
  fem::cuda::ElemRestriction dev;
  fem::cuda::elem_restriction_create(host, ScatterMode::Deterministic, dev);
  DeviceArray<double> e(wiggly(host.e_size, 0.3));

  std::vector<double> first;
  for (int run = 0; run < 5; ++run) {
    DeviceArray<double> l(host.l_size);
    l.zero();
    fem::cuda::elem_restriction_apply(dev, ContractMode::Transpose, e, l);
    const std::vector<double> result = l.to_host();
    if (run == 0) {
      first = result;
    } else {
      REQUIRE(result == first);  // exact equality, not Approx
    }
  }
  // Sanity: it's the right answer, too.
  std::vector<double> cpu(host.l_size, 0.0);
  fem::elem_restriction_apply(host, ContractMode::Transpose, e.to_host(), cpu);
  require_close(first, cpu);
}

TEST_CASE("cuda strided restriction matches the CPU", "[cuda][elem-restriction][strided][kernels]") {
  SECTION("backend strides (qdata layout): gather is a copy") {
    fem::ElemRestriction r;
    fem::elem_restriction_create_strided(5, 27, 7, 5 * 27 * 7, r);
    check_against_cpu(r, ScatterMode::Atomic);
  }
  SECTION("user strides [e][i][k]") {
    const int num_elem = 4, elem_size = 8, num_comp = 3;
    fem::ElemRestriction r;
    fem::elem_restriction_create_strided(num_elem, elem_size, num_comp, num_elem * elem_size * num_comp,
                                         {num_comp, 1, elem_size * num_comp}, r);
    check_against_cpu(r, ScatterMode::Deterministic);  // mode is ignored for strided
  }
}
