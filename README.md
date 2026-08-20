# fe-demo-cpp

A from-scratch C++ matrix-free finite-element basis library, built as a
learning project mirroring [libCEED](https://github.com/CEED/libCEED)'s
architecture (and, through it, [Ratel](https://gitlab.com/micromorph/ratel)).
Long-term goal: a full CPU+MPI+CUDA matrix-free solid mechanics solver in
3D. Right now this covers the tensor-product basis layer — quadrature,
Lagrange basis construction, and tensor-contraction-based basis evaluation
— everything below the mesh/solver level.

PETSc is intended for the solver/preconditioner layer only, once that's
reached; basis functions, element restriction, and (later) CUDA kernels
are hand-written here rather than pulled in as a libCEED dependency.

## What's implemented

- ** 1D quadrature rules: Fornberg's algorithm for Lagrange interpolation/derivative matrices
  - Tensor-product H1 Lagrange basis for `dim` = 1, 2, or 3, arbitrary node/quadrature order,
    `num_comp` fields, `Gauss` or `GaussLobatto` quadrature
- ** The single batched-contraction primitive (libCEED's `CeedTensorContractApply`) 
  - `tensor_basis_apply` — dispatches the two above by `EvalMode`, mirroring `CeedBasisApply`


**Not yet implemented**: element restriction (local-to-global DOF
mapping), mesh/geometry, MPI, CUDA, PETSc integration, any physics.

## Requirements

- C++17 compiler
- CMake ≥ 3.20

(Catch2 is fetched automatically via CMake's `FetchContent` — no manual install needed.)

## Build and test

```bash
cmake -S . -B build
cmake --build build -j
./build/fe_demo_test
```

Run a subset of tests by tag or name:

```bash
./build/fe_demo_test "[manufactured]"
./build/fe_demo_test "[transpose]"
./build/fe_demo_test --list-tests
```
