cmake_minimum_required(VERSION 3.16)
set(CMAKE_C_COMPILER clang) # Specify gcc/msvc/clang
set(CMAKE_CXX_COMPILER clang++) # Specify g++/msvc/clang++

project(deepity CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

find_path(BLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas x86_64-linux-gnu
    REQUIRED
)

include(FetchContent)

# --------------------------------------------------------------------
# SLEEF
# --------------------------------------------------------------------

include(FetchContent)

set(SLEEF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_STATIC_TEST_BINS OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

set(SLEEF_BUILD_QUAD OFF CACHE BOOL "" FORCE)
set(SLEEF_BUILD_DFT OFF CACHE BOOL "" FORCE)

set(SLEEF_ENABLE_TLFLOAT OFF CACHE BOOL "" FORCE)
set(ENABLE_TLFLOAT OFF CACHE BOOL "" FORCE)

set(SLEEF_ENABLE_FLOAT128 OFF CACHE BOOL "" FORCE)
set(SLEEF_ENFORCE_FLOAT128 OFF CACHE BOOL "" FORCE)

set(SLEEF_ENABLE_MPFR OFF CACHE BOOL "" FORCE)
set(SLEEF_ENABLE_SSL OFF CACHE BOOL "" FORCE)

set(SLEEF_ENABLE_AVX2 ON CACHE BOOL "" FORCE)
set(SLEEF_ENABLE_AVX512F ON CACHE BOOL "" FORCE)

set(CMAKE_INSTALL_PREFIX
    "${CMAKE_BINARY_DIR}/sleef_install"
    CACHE PATH "" FORCE
)

FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG 3.9.0
)

FetchContent_MakeAvailable(sleef)

# --------------------------------------------------------------------
# Google Benchmark
# --------------------------------------------------------------------

# Suppress the C2y extension warning/error on Clang for the benchmark build
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wno-c2y-extensions)
endif()

set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_PEDANTIC OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)

FetchContent_MakeAvailable(benchmark)

# --------------------------------------------------------------------
# Deepity library
# --------------------------------------------------------------------

add_library(Deepity STATIC
    src/DiscriminativePCLayer.cpp
    src/RBLayer.cpp
    src/DiscriminativePCNetwork.cpp
)

target_include_directories(Deepity PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${BLAS_INCLUDE_DIR}
    ${sleef_BINARY_DIR}/include
    ${sleef_SOURCE_DIR}/include
)

target_link_libraries(Deepity PUBLIC
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    sleef
)


# --------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------

if(MSVC)

    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
        $<$<CONFIG:Debug>:/Od /Zi /W4>
    )

else()

    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math -fvectorize -fslp-vectorize>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g -march=native -ffast-math -fvectorize -fslp-vectorize>
        $<$<CONFIG:Debug>:-O0 -g3 -Wall -Wextra -Wpedantic>
    )

    target_compile_options(Deepity PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )

    target_link_options(Deepity PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )

endif()


target_compile_definitions(Deepity PUBLIC
    $<$<CONFIG:Debug>:_DEBUG>
)


set_target_properties(Deepity PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# --------------------------------------------------------------------
# Python bindings
# --------------------------------------------------------------------

pybind11_add_module(deepity
    bindings/pybinding.cpp
)

target_link_libraries(deepity PRIVATE
    Deepity
)

set_target_properties(deepity PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------

add_executable(DeepityTests
    tests/tBenchmark.cpp
)

target_link_libraries(DeepityTests PRIVATE
    Deepity
    benchmark::benchmark
)

set_target_properties(DeepityTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)