cmake_minimum_required(VERSION 3.16)
set(CMAKE_C_COMPILER gcc) # Specify gcc/msvc/clang
set(CMAKE_CXX_COMPILER g++) # Specify g++/msvc/clang++

project(deepity CXX)
option(DEEPITY_ENABLE_CUDA "Build with CUDA support" ON)
option(DEEPITY_BUILD_TESTS "Build the DeepityTests target" OFF)
option(DEEPITY_BUILD_PYTHON_BINDINGS "Build the Python bindings target" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------

function(deploy_runtime_deps target)
    if(WIN32)
        add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND "C:/vcpkg/vcpkg.exe" z-applocal
                --target-binary=$<TARGET_FILE:${target}>
                --installed-bin-dir=C:/vcpkg/installed/x64-windows/bin
        )
        add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "C:/Program Files/LLVM/bin/libomp.dll"
                $<TARGET_FILE_DIR:${target}>
        )
        if(CUDAToolkit_FOUND)
    file(GLOB CUDA_RUNTIME_DLLS
        "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
        "${CUDAToolkit_BIN_DIR}/cudart64_*.dll"
    )

    foreach(dll ${CUDA_RUNTIME_DLLS})
        add_custom_command(
            TARGET ${target} POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${dll}"
                $<TARGET_FILE_DIR:${target}>
        )
        endforeach()
    endif()
    endif()
endfunction()

# --------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
if(DEEPITY_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)

    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA support enabled.")
    else()
        message(STATUS "CUDA toolkit not found. Building CPU-only.")
    endif()
endif()

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
    src/ConvPCLayer.cpp
    src/SimpleConvPCLayer.cpp
    src/SimplePCLayer.cpp
    src/DiscriminativePCNetwork.cpp
    src/ConvPCNetwork.cpp    
    src/SimplePCNetwork.cpp
    src/ModelIO.cpp
    src/StreamAlignedBatcher.cpp
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

if(CUDAToolkit_FOUND)
    target_link_libraries(Deepity PUBLIC
        CUDA::cublas
        CUDA::cudart
    )

    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_CUDA)
endif()

target_compile_definitions(Deepity PUBLIC
    SLEEF_STATIC_LIBS
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

elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

    # -fvectorize / -fslp-vectorize are Clang-only spellings; GCC has no
    # flags by these names at all and errors outright on them.
    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math -fvectorize -fslp-vectorize>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g -march=native -ffast-math -fvectorize -fslp-vectorize>
        $<$<CONFIG:Debug>:-O0 -g3 -Wall -Wextra -Wpedantic -march=native>
    )

    target_compile_options(Deepity PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )

    target_link_options(Deepity PUBLIC
        $<$<CONFIG:Debug>:-fsanitize=address,undefined>
    )

else() # GNU and other non-MSVC, non-Clang compilers

    # GCC's equivalent auto-vectorization flags -- different spelling than
    # Clang's. -O3 already implies both; stated explicitly to mirror intent.
    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native -ffast-math -ftree-vectorize -ftree-slp-vectorize>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g -march=native -ffast-math -ftree-vectorize -ftree-slp-vectorize>
        $<$<CONFIG:Debug>:-O0 -g3 -Wall -Wextra -Wpedantic -march=native>
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

if(WIN32)
    target_compile_definitions(Deepity PUBLIC NOMINMAX)
endif()

set_target_properties(Deepity PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)


# --------------------------------------------------------------------
# Python bindings
# --------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")

    pybind11_add_module(deepity
        bindings/pybinding.cpp
    )

target_link_libraries(deepity PRIVATE
    Deepity
)

if(WIN32)
    target_compile_definitions(deepity PRIVATE NOMINMAX)
endif()

    set_target_properties(deepity PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/pydeepity
    )

    add_custom_command(
        TARGET deepity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${CMAKE_CURRENT_SOURCE_DIR}/bindings/__init__.py
            $<TARGET_FILE_DIR:deepity>/__init__.py
    )

endif()
# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------
add_executable(DeepityTests
    tests/tResearch.cpp
)

# target_compile_definitions(Deepity PRIVATE PCN_PROFILE)
# target_compile_definitions(DeepityTests PRIVATE PCN_PROFILE)


target_link_libraries(DeepityTests PRIVATE
    Deepity
    benchmark::benchmark
)

set_target_properties(DeepityTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
