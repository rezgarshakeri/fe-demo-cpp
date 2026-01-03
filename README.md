# fe-demo-cpp
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Debug -DMINIFEM_ENABLE_MPI=ON
cmake --build build-mpi -j
mpiexec -n 4 ./build-mpi/poisson1D_mpi


cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DMINIFEM_ENABLE_MPI=ON -DMINIFEM_ENABLE_CUDA=ON
cmake --build build-gpu -j
CUDA_LAUNCH_BLOCKING=1 mpiexec -n 4 ./build-gpu/poisson1D_mpi_cuda