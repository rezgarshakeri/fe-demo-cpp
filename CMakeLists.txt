cmake_minimum_required(VERSION 3.20)
project(mini_fem LANGUAGES CXX)

# ---------- Options ----------
option(MINIFEM_ENABLE_MPI  "Enable MPI support"  ON)
option(MINIFEM_ENABLE_CUDA "Enable CUDA support" ON)
option(MINIFEM_BUILD_TESTS "Build unit tests"    ON)

# ---------- C++ standard ----------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------- Library ----------
file(GLOB MINIFEM_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(MINIFEM_ENABLE_MPI)
  file(GLOB MINIFEM_MPI_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/mpi/*.cpp"
  )
  list(APPEND MINIFEM_SOURCES ${MINIFEM_MPI_SOURCES})
endif()

add_library(mini_fem STATIC ${MINIFEM_SOURCES})
target_include_directories(mini_fem PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Warnings
target_compile_options(mini_fem PRIVATE
  $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall -Wextra -Wpedantic>
  $<$<CXX_COMPILER_ID:MSVC>:/W4>
)

# Useful for backtraces/profiling on Linux
if(UNIX AND NOT APPLE)
  target_link_options(mini_fem PUBLIC -rdynamic)
endif()

# ---------- MPI ----------
if(MINIFEM_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(mini_fem PUBLIC MPI::MPI_CXX)
  target_compile_definitions(mini_fem PUBLIC MINIFEM_USE_MPI=1)
  add_executable(halo_print examples/halo_print.cpp)
  target_link_libraries(halo_print PRIVATE mini_fem MPI::MPI_CXX)
  add_executable(apply_mpi_check examples/apply_mpi_check.cpp)
  target_link_libraries(apply_mpi_check PRIVATE mini_fem MPI::MPI_CXX)
  add_executable(poisson1D_mpi examples/poisson1D_mpi.cpp)
  target_link_libraries(poisson1D_mpi PRIVATE mini_fem MPI::MPI_CXX)

  add_executable(mpi_hello examples/mpi_hello.cpp)
  target_link_libraries(mpi_hello PRIVATE MPI::MPI_CXX)

  add_executable(dist1d_print examples/dist1d_print.cpp)
  target_link_libraries(dist1d_print PRIVATE mini_fem MPI::MPI_CXX)
endif()

# ---------- CUDA ----------
if(MINIFEM_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  # For my laptop: Quadro P520 = Pascal, compute capability 6.1 (sm_61)
  set(CMAKE_CUDA_ARCHITECTURES 61)

  add_library(mini_fem_cuda STATIC
    src/cuda/cuda-device-vector.cu
    src/cuda/cuda-matrix-free.cu
  )

  target_include_directories(mini_fem_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_link_libraries(mini_fem_cuda PRIVATE mini_fem)

  # Good defaults
  set_target_properties(mini_fem_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
  )

  target_compile_options(mini_fem_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
  )

  # MPI+CUDA example (only if MPI is enabled too)
  if(MINIFEM_ENABLE_MPI)
    add_executable(poisson1D_mpi_cuda examples/poisson1D_mpi_cuda.cpp)
    target_link_libraries(poisson1D_mpi_cuda PRIVATE mini_fem mini_fem_cuda MPI::MPI_CXX)
  endif()
endif()

# ---------- Example executable ----------
add_executable(poisson1D examples/poisson1D.cpp)
target_link_libraries(poisson1D PRIVATE mini_fem)

# ---------- Unit tests ----------
if(MINIFEM_BUILD_TESTS)
  include(CTest)
  enable_testing()

  include(FetchContent)
  FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG        v3.5.4
  )
  FetchContent_MakeAvailable(Catch2)

  file(GLOB MINIFEM_TESTS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
  )

  # Only add test target if tests exist
  if(MINIFEM_TESTS)
    add_executable(mini_fem_tests ${MINIFEM_TESTS})
    target_link_libraries(mini_fem_tests PRIVATE mini_fem Catch2::Catch2WithMain)

    add_test(NAME mini_fem_tests COMMAND mini_fem_tests)
  endif()
endif()
