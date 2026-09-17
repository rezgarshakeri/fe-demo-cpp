# fe-demo-cpp

A from-scratch C++ matrix-free finite-element library, built mirroring [libCEED](https://github.com/CEED/libCEED)'s architecture
(and, through it, [Ratel](https://gitlab.com/micromorph/ratel)). Long-term
goal: a CPU + MPI + CUDA matrix-free solid mechanics solver in 3D.

Everything is hand-written here rather than pulled in as a libCEED
dependency. The library currently has no dependencies at all: no MPI, no
PETSc, no CUDA.

## Roadmap

1. **Basis and tensor contraction** (done)
2. **Element restriction** (done)
3. **QFunction and Operator**, then the CEED bakeoff problems (BP1-BP6) in a
   single-file, PETSc-free form, like libCEED's `examples/ceed/ex1-volume.c`:
   structured Cartesian mesh, restriction, basis, QFunction, operator and a
   CG solve, all built directly (in progress)
4. PETSc / DMPlex (unstructured meshes, solvers), MPI, then CUDA

## What's implemented

**Basis** (`include/basis.hpp`, `src/basis.cpp`)
- 1D Gauss-Legendre and Gauss-Lobatto quadrature
- Lagrange interpolation/derivative matrices via Fornberg's algorithm
- `TensorBasis`: tensor-product H1 Lagrange basis for `dim` = 1, 2, 3, arbitrary
  node/quadrature order, `num_comp` fields, `Gauss` or `GaussLobatto` quadrature
  (libCEED's `CeedBasisCreateTensorH1Lagrange`)

**Tensor contraction** (`include/tensor-contract.hpp`, `src/tensor-contract.cpp`)
- `tensor_contract_apply`: the single batched-contraction primitive
  (libCEED's `CeedTensorContractApply`)
- `tensor_basis_apply_interp` / `tensor_basis_apply_grad`: sum-factorized
  evaluation at quadrature points, batched over `num_comp` fields and
  `num_elem` elements, with `ContractMode::Transpose` (the adjoint)
- `tensor_basis_apply`: dispatches by `EvalMode` (libCEED's `CeedBasisApply`)

**Element restriction** (`include/elem-restriction.hpp`, `src/elem-restriction.cpp`)
- `ElemRestriction` + `elem_restriction_create` (validates sizes and offsets)
- `elem_restriction_apply`: gather (L-vector to E-vector) and its adjoint,
  scatter-add. The E-vector layout matches the basis apply functions, so the
  output feeds straight into them
- `elem_restriction_get_multiplicity`

`include/transpose-mode.hpp` holds `ContractMode` (`NoTranspose` /
`Transpose`), shared by all three modules (libCEED's `CeedTransposeMode`).

**Not yet implemented**: QFunction, Operator, `EvalMode::Weight`, strided
(quadrature-point) restriction, mesh generation, solvers, MPI, CUDA, PETSc.

## Requirements

- C++17 compiler
- CMake >= 3.20

Catch2 is fetched automatically via CMake's `FetchContent`.

## Build and test

```bash
cmake -S . -B build
cmake --build build -j
./build/fe_demo_test
```

Run a subset of the tests by tag or name:

```bash
./build/fe_demo_test "[elem-restriction]"
./build/fe_demo_test "[manufactured]"
./build/fe_demo_test "[transpose]"
./build/fe_demo_test --list-tests
```
