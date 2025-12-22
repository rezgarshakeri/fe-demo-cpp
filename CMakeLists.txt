cmake_minimum_required(VERSION 3.20)
project(mini_fem LANGUAGES CXX)

# ---------- Options ----------
option(MINIFEM_ENABLE_MPI  "Enable MPI support"  OFF)
option(MINIFEM_ENABLE_CUDA "Enable CUDA support" OFF)
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

# ---------- MPI (TODO) ----------
if(MINIFEM_ENABLE_MPI)
  find_package(MPI REQUIRED)
  target_link_libraries(mini_fem PUBLIC MPI::MPI_CXX)
  target_compile_definitions(mini_fem PUBLIC MINIFEM_USE_MPI=1)
endif()

# ---------- CUDA (TODO) ----------
if(MINIFEM_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  target_compile_definitions(mini_fem PUBLIC MINIFEM_USE_CUDA=1)

  # Later we need .cu files like:
  # target_sources(mini_fem PRIVATE src/cuda/mf_apply.cu)
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
