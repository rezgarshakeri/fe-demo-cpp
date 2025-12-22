#include <catch2/catch_test_macros.hpp>
#include "mesh.hpp"

TEST_CASE("Uniform Mesh1D has expected sizes and connectivity", "[mesh]") {
  const int Ne = 10;
  auto mesh = fem1d::Mesh1D::build_uniform(Ne, 0.0, 1.0);

  REQUIRE(mesh.num_elements == Ne);
  REQUIRE(mesh.nodes.size() == static_cast<size_t>(Ne + 1));
  REQUIRE(mesh.elem_nodes.size() == static_cast<size_t>(2 * Ne));

  // Check element connectivity
  for (int e = 0; e < Ne; ++e) {
    REQUIRE(mesh.elem_nodes[2 * e + 0] == e);
    REQUIRE(mesh.elem_nodes[2 * e + 1] == e + 1);
  }

  // Monotone nodes
  for (size_t i = 1; i < mesh.nodes.size(); ++i) {
    REQUIRE(mesh.nodes[i] > mesh.nodes[i - 1]);
  }
}
