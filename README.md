# fe-demo-cpp

A from-scratch C++ matrix-free finite-element library, built mirroring [libCEED](https://github.com/CEED/libCEED)'s architecture
(and, through it, [Ratel](https://gitlab.com/micromorph/ratel)).

Everything is hand-written here rather than pulled in as a libCEED
dependency. The core library has no dependencies: no MPI, no PETSc, no CUDA.
The CUDA part is optional and off by default.

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
- `elem_restriction_create` (offsets, validated) and
  `elem_restriction_create_strided` (for quadrature-point data such as qdata;
  backend strides or user strides)
- `elem_restriction_apply`: gather (L-vector to E-vector) and its adjoint,
  scatter-add. The E-vector layout matches the basis apply functions, so the
  output feeds straight into them
- `elem_restriction_get_multiplicity`

**QFunction** (`include/qfunction.hpp`, `src/qfunction.cpp`)
- The pointwise kernel: a plain function pointer called on a batch of `Q`
  points, with named input/output fields (size and `EvalMode`), field layout
  `[size][Q]`, and an optional trivially copyable context
  (libCEED's `CeedQFunction`)
- `include/qfunctions/mass.hpp`: `build_mass` (`qdata = det(J) * w`, dim 1-3)
  and `apply_mass` (`v = qdata * u`)

**Operator** (`include/operator.hpp`, `src/operator.cpp`)
- Ties each QFunction field to a restriction, a basis and a vector (active,
  passive or none), with the consistency checks of libCEED's
  `CeedOperatorSetField`
- `operator_apply`: restrict, basis evaluation, QFunction, basis transpose,
  restrict transpose (scatter-add), batched over all elements
  (libCEED's `CeedOperatorApply`)

**CUDA infrastructure** (`include/cuda/`, `src/cuda/`; optional, `-DFE_DEMO_CUDA=ON`)
- `CUDA_CHECK` / `CUDA_CHECK_LAUNCH`: turn CUDA errors into exceptions with file and line
- `DeviceArray<T>`: move-only owner of a GPU allocation, with explicit host/device copies
- `device_info`: device limits (shared memory, threads per block, SMs) and free memory

`include/eval-mode.hpp` holds `EvalMode` (`None`, `Interp`, `Grad`, `Weight`) and
`include/transpose-mode.hpp` holds `ContractMode` (`NoTranspose` / `Transpose`;
libCEED's `CeedTransposeMode`).

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

CUDA (optional; needs the CUDA toolkit and a GPU, built for the GPU in the
machine unless `-DCMAKE_CUDA_ARCHITECTURES=...` is given):

```bash
cmake -S . -B build-cuda -DFE_DEMO_CUDA=ON
cmake --build build-cuda -j
./build-cuda/fe_demo_cuda_test
```

Run a subset of the tests by tag or name:

```bash
./build/fe_demo_test "[operator]"
./build/fe_demo_test "[qfunction]"
./build/fe_demo_test "[elem-restriction]"
./build/fe_demo_test "[manufactured]"
./build/fe_demo_test --list-tests
```
