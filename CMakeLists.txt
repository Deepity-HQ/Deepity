cmake_minimum_required(VERSION 3.21)

# Do NOT force GCC here. Let CMake detect the native compiler (MSVC, Clang, GCC)
project(deepity CXX)

option(DEEPITY_ENABLE_CUDA "Build with CUDA support" ON)
option(DEEPITY_BUILD_TESTS "Build the DeepityTests target" OFF)
option(DEEPITY_BUILD_PYTHON_BINDINGS "Build the Python bindings target" ON)
option(DEEPITY_NATIVE_OPT "Enable -march=native for max performance (disable for portable builds)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

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

# Find cblas.h without hardcoding the Linux arch path
find_path(BLAS_INCLUDE_DIR
    NAMES cblas.h
    PATH_SUFFIXES openblas include/openblas
    REQUIRED
)

if(DEEPITY_BUILD_PYTHON_BINDINGS)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 REQUIRED)
endif()

include(FetchContent)

# --------------------------------------------------------------------
# SLEEF
# --------------------------------------------------------------------

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

set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/sleef_install" CACHE PATH "" FORCE)

FetchContent_Declare(
    sleef
    GIT_REPOSITORY https://github.com/shibatch/sleef.git
    GIT_TAG 3.9.0
)

set(CMAKE_REQUIRED_QUIET ON)
FetchContent_MakeAvailable(sleef)
set(CMAKE_REQUIRED_QUIET OFF)

# --------------------------------------------------------------------
# Google Benchmark
# --------------------------------------------------------------------

if(DEEPITY_BUILD_TESTS)
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

    set(CMAKE_REQUIRED_QUIET ON)
    FetchContent_MakeAvailable(benchmark)
    set(CMAKE_REQUIRED_QUIET OFF)
endif()

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
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${BLAS_INCLUDE_DIR}
)

target_include_directories(Deepity SYSTEM PUBLIC
    ${sleef_SOURCE_DIR}/include
    ${sleef_BINARY_DIR}/include
)

target_link_libraries(Deepity PUBLIC
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
    sleef
)

if(CUDAToolkit_FOUND)
    target_link_libraries(Deepity PUBLIC CUDA::cublas CUDA::cudart)
    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_CUDA)
endif()

target_compile_definitions(Deepity PUBLIC SLEEF_STATIC_LIBS)

# --------------------------------------------------------------------
# Compiler flags
# --------------------------------------------------------------------

# Handle architecture optimization dynamically
set(ARCH_FLAGS "")
if(DEEPITY_NATIVE_OPT AND NOT MSVC)
    set(ARCH_FLAGS "-march=native")
endif()

if(MSVC)
    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:/O2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi>
        $<$<CONFIG:Debug>:/Od /Zi /W4>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:-O3 ${ARCH_FLAGS} -ffast-math -fvectorize -fslp-vectorize>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g ${ARCH_FLAGS} -ffast-math -fvectorize -fslp-vectorize>
        $<$<CONFIG:Debug>:-O0 -g3 -Wall -Wextra -Wpedantic ${ARCH_FLAGS}>
    )
    target_compile_options(Deepity PUBLIC $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(Deepity PUBLIC $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
else() # GCC
    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:-O3 ${ARCH_FLAGS} -ffast-math -ftree-vectorize -ftree-slp-vectorize>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g ${ARCH_FLAGS} -ffast-math -ftree-vectorize -ftree-slp-vectorize>
        $<$<CONFIG:Debug>:-O0 -g3 -Wall -Wextra -Wpedantic ${ARCH_FLAGS}>
    )
    target_compile_options(Deepity PUBLIC $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
    target_link_options(Deepity PUBLIC $<$<CONFIG:Debug>:-fsanitize=address,undefined>)
endif()

target_compile_definitions(Deepity PUBLIC $<$<CONFIG:Debug>:_DEBUG>)

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

if(DEEPITY_BUILD_PYTHON_BINDINGS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    pybind11_add_module(deepity bindings/pybinding.cpp)

    target_link_libraries(deepity PRIVATE Deepity)

    if(WIN32)
        target_compile_definitions(deepity PRIVATE NOMINMAX)
    endif()

    set_target_properties(deepity PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pydeepity
        PREFIX ""
    )

    # Modern cross-platform way to copy required DLLs (like OpenMP/BLAS) to the Python package on Windows
    if(WIN32)
        add_custom_command(TARGET deepity POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:deepity>
            $<TARGET_FILE_DIR:deepity>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

# --------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------

if(DEEPITY_BUILD_TESTS)
    add_executable(DeepityTests tests/tResearch.cpp)

    target_link_libraries(DeepityTests PRIVATE
        Deepity
        benchmark::benchmark
    )

    set_target_properties(DeepityTests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # Copy DLLs for the test executable on Windows
    if(WIN32)
        add_custom_command(TARGET DeepityTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_RUNTIME_DLLS:DeepityTests>
            $<TARGET_FILE_DIR:DeepityTests>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()
