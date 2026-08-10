cmake_minimum_required(VERSION 3.20)
project(fe_demo_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(fe_demo STATIC
  src/basis.cpp
  src/tensor-contract.cpp
)

target_include_directories(fe_demo PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG v3.5.4
)
FetchContent_MakeAvailable(Catch2)

add_executable(fe_demo_test
  tests/test-basis.cpp
  tests/test-tensor-contract.cpp
)
target_link_libraries(fe_demo_test PRIVATE fe_demo Catch2::Catch2WithMain)

include(CTest)
enable_testing()
add_test(NAME fe_demo_test COMMAND fe_demo_test)