# fe-demo-cpp
cmake -S . -B build-mpi -DCMAKE_BUILD_TYPE=Debug -DMINIFEM_ENABLE_MPI=ON
cmake --build build-mpi -j
mpiexec -n 4 ./build-mpi/poisson1D_mpi