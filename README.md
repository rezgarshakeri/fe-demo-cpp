# fe-demo-cpp

A from-scratch C++ matrix-free finite-element library, built mirroring [libCEED](https://github.com/CEED/libCEED)'s architecture
(and, through it, [Ratel](https://gitlab.com/micromorph/ratel)). Long-term
goal: a CPU + MPI + CUDA matrix-free solid mechanics solver in 3D.

Everything is hand-written here rather than pulled in as a libCEED
dependency. The library currently has no dependencies at all: no MPI, no
PETSc, no CUDA.

## Roadmap

Done: **basis + tensor contraction**, **element restriction**.

**Current branch (`rezgar/setup-qfunc-opt`): QFunction + Operator.**
Goal: a PETSc-free, single-file pipeline like libCEED's
`examples/ceed/ex1-volume.c`. It builds a structured mesh, restriction, bases,
QFunctions and operators directly, then computes the domain volume as
`1^T M 1` and checks it against the exact value. (Note `examples/petsc/bpsraw.c`
is "raw" only in skipping DMPlex, it still uses PETSc for `Vec`/`KSP`/MPI.
`ex1-volume.c` has no PETSc at all.)

Milestone A: ex1-volume
- [x] `EvalMode::Weight`: tensor-product quadrature weights
      (`w[q] = prod_d q_weight_1d[q_d]`); test that they sum to `2^dim`
- [x] Decide how quadrature-point data (qdata) is stored. libCEED uses a
      *strided* restriction; qdata is never shared between elements, so we
      could instead treat `EVAL_NONE` fields as already in Q-vector layout
- [ ] `QFunction`: callback over `Q` points (times `num_elem`), named input and
      output fields with size and `EvalMode`, optional context. Decide the
      in/out layout (libCEED: `[size][Q]` per field)
- [ ] `Operator`: named fields tying restriction + basis + vector (active or
      passive). Apply = restrict -> basis eval -> QFunction -> basis transpose
      -> restrict transpose. Test against a hand-composed version and check
      symmetry/adjointness of the mass operator
- [ ] Cartesian mesh builder for dim 1-3, degree p: offsets + node coordinates
      (libCEED: `BuildCartesianRestriction`, `SetCartesianMeshCoords`)
- [ ] Geometry via the mesh basis: `Grad` of the coordinates gives the Jacobian
      `J`, then QFunction `build_mass` computes `det(J) * w`; QFunction
      `apply_mass` computes `v = qdata * u`
- [ ] `examples/ex1-volume`: volume vs exact, for several `dim`, mesh and
      solution degrees

Milestone B: CEED bakeoff problems (`libCEED/examples/bps.md`)
- [ ] Own CG solver (no PETSc)
- [ ] BP1/BP2: mass, right-hand side operator, manufactured solution, error
- [ ] BP3/BP4: Laplace (`qdata` from `J^-1 J^-T det(J) w`, `Grad` in and out)
- [ ] BP5/BP6: same with collocated Gauss-Lobatto quadrature
- [ ] Compare errors against libCEED's own bps output for the same sizes

Later
- PETSc / DMPlex for unstructured meshes and solvers. Lead: PETSc's
  `DMPlexGetLocalOffsets` (`petsc/src/dm/impls/plex/plexceed.c`) returns exactly
  our `offsets`/`num_elem`/`elem_size`/`num_comp`/`l_size`; it needs a `PetscFE`
  on the DM and `DMPlexSetClosurePermutationTensor` for tensor node ordering.
  PETSc lives in `~/RATEL/petsc` (`arch-mpich-cuda` has MPI and CUDA)
- MPI, then CUDA (start from a libCEED `ref`-style kernel, then `shared`)
- Elasticity: Ratel-style QFunctions

Small open items
- `basis.cpp`: two signed/unsigned comparison warnings under `-Wall -Wextra`
- `tensor_basis_apply_grad`: the `u_slice` copy in Transpose mode could be
  zero-copy (see the `TODO(perf)` comment there)

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
- `tensor_basis_apply_weight`: tensor-product quadrature weights
- `tensor_basis_apply`: dispatches by `EvalMode` (`Interp`, `Grad`, `Weight`;
  libCEED's `CeedBasisApply`)

**Element restriction** (`include/elem-restriction.hpp`, `src/elem-restriction.cpp`)
- `ElemRestriction` + `elem_restriction_create` (validates sizes and offsets)
- `elem_restriction_apply`: gather (L-vector to E-vector) and its adjoint,
  scatter-add. The E-vector layout matches the basis apply functions, so the
  output feeds straight into them
- `elem_restriction_get_multiplicity`

`include/transpose-mode.hpp` holds `ContractMode` (`NoTranspose` /
`Transpose`), shared by all three modules (libCEED's `CeedTransposeMode`).

**Not yet implemented**: QFunction, Operator, strided
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
