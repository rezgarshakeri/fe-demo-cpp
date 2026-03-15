# fe-demo-cpp

A minimal **matrix-free finite element (FEM) demo** in C++ for solving the **1D Poisson equation** using:

- MPI domain decomposition
- Conjugate Gradient (CG) solver
- Matrix-free operator
- Optional CUDA acceleration

## Requirements

- C++17 compiler
- CMake ≥ 3.20
- MPI (OpenMPI / MPICH)

Optional:
- NVIDIA CUDA toolkit

---

## Build and Run (MPI CPU)

```bash
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Debug -DMINIFEM_ENABLE_MPI=ON
cmake --build build-mpi -j
mpiexec -n 4 ./build-mpi/poisson1D_mpi
```

## Build and Run (MPI CUDA)

```bash
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DMINIFEM_ENABLE_MPI=ON -DMINIFEM_ENABLE_CUDA=ON
cmake --build build-gpu -j
CUDA_LAUNCH_BLOCKING=1 mpiexec -n 4 ./build-gpu/poisson1D_mpi_cuda
```