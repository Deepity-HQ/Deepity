cmake_minimum_required(VERSION 3.21)

project(deepity C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DEEPITY_ENABLE_CUDA "Build with CUDA support" ON)
option(DEEPITY_BUILD_TESTS "Build the DeepityTests target" OFF)
option(DEEPITY_BUILD_PYTHON_BINDINGS "Build the Python bindings target" ON)

# --- Global Architecture & SIMD Flags -------------------------------
# NOTE: build.py always sets these explicitly based on --native/--fast/
# --distributed. These CACHE defaults only matter if someone invokes cmake
# directly. Default to the portable "fast" profile (AVX2+FMA, x86-64-v3)
# rather than -march=native, since native binaries crash on other CPUs.
set(DEEPITY_ARCH_FLAGS "-march=x86-64-v3 -mtune=generic" CACHE STRING "Architecture/SIMD flags (e.g., -march=native, -march=x86-64-v2)")
set(DEEPITY_MSVC_ARCH_FLAGS "/arch:AVX2" CACHE STRING "Architecture flags for MSVC (empty = compiler default / most portable)")

if(MSVC)
    if(DEEPITY_MSVC_ARCH_FLAGS)
        add_compile_options(${DEEPITY_MSVC_ARCH_FLAGS})
    endif()

    if(DEEPITY_MSVC_ARCH_FLAGS MATCHES "AVX512")
        add_compile_definitions(__SSE2__ __AVX__ __AVX2__ __AVX512F__)
    elseif(DEEPITY_MSVC_ARCH_FLAGS MATCHES "AVX2")
        add_compile_definitions(__SSE2__ __AVX__ __AVX2__)
    elseif(DEEPITY_MSVC_ARCH_FLAGS MATCHES "AVX")
        add_compile_definitions(__SSE2__ __AVX__)
    endif()
else()
    separate_arguments(ARCH_FLAGS_LIST UNIX_COMMAND ${DEEPITY_ARCH_FLAGS})
    add_compile_options(${ARCH_FLAGS_LIST})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")

# --- DEPENDENCIES ---------------------------------------------------

find_package(OpenMP REQUIRED)

if(DEEPITY_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        message(STATUS "CUDA support enabled.")
    else()
        message(STATUS "CUDA toolkit not found. Building CPU-only.")
    endif()
endif()

if(DEEPITY_BUILD_PYTHON_BINDINGS)
    set(PYBIND11_FINDPYTHON ON)
    find_package(pybind11 REQUIRED)
endif()

include(FetchContent)

# --- OpenBLAS -------------------------------------------------------

if(WIN32)
    set(OPENBLAS_WIN_URL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.34/OpenBLAS-0.3.34-x64.zip"
        CACHE STRING "URL of the official OpenBLAS prebuilt Windows release zip")

    FetchContent_Declare(
        openblas_prebuilt
        URL ${OPENBLAS_WIN_URL}
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(openblas_prebuilt)

    set(BLAS_INCLUDE_DIR "${openblas_prebuilt_SOURCE_DIR}/include")

    find_library(OPENBLAS_IMPLIB
        NAMES openblas libopenblas
        PATHS "${openblas_prebuilt_SOURCE_DIR}/lib"
        REQUIRED
    )
    find_file(OPENBLAS_DLL
        NAMES libopenblas.dll openblas.dll
        PATHS "${openblas_prebuilt_SOURCE_DIR}/bin"
        REQUIRED
    )
    message(STATUS "OPENBLAS_DLL resolved to: ${OPENBLAS_DLL}")

    add_library(openblas_dll SHARED IMPORTED)
    set_target_properties(openblas_dll PROPERTIES
        IMPORTED_LOCATION "${OPENBLAS_DLL}"
        IMPORTED_IMPLIB "${OPENBLAS_IMPLIB}"
    )
    set(BLAS_LIBRARIES openblas_dll)
else()
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS REQUIRED)
    find_path(BLAS_INCLUDE_DIR
        NAMES cblas.h
        PATH_SUFFIXES openblas include/openblas
        REQUIRED
    )
endif()

# --- SLEEF ----------------------------------------------------------
# These are necessary, but don't ask me why
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
set(SLEEF_ENABLE_SSE2 ON CACHE BOOL "" FORCE)
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

# --- Google Benchmark -----------------------------------------------

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

# --- Deepity library ------------------------------------------------

add_library(Deepity
    src/DiscriminativePCLayer.cpp
    src/RBLayer.cpp
    src/ConvPCLayer.cpp
    src/SimpleConvPCLayer.cpp
    src/SimplePCLayer.cpp
    src/DiscriminativePCNetwork.cpp
    src/ConvPCNetwork.cpp
    src/SimplePCNetwork.cpp
    src/SimpleConvPCNetwork.cpp
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

# --- Compiler flags -------------------------------------------------

if(MSVC)
    target_compile_options(Deepity PUBLIC
        $<$<CONFIG:Release>:/O2 /fp:fast /openmp:llvm>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi /fp:fast /openmp:llvm>
        $<$<CONFIG:Debug>:/Od /Zi /W4 /openmp:llvm>
    )
endif()

target_compile_definitions(Deepity PUBLIC $<$<CONFIG:Debug>:_DEBUG>)

if(WIN32)
    target_compile_definitions(Deepity PUBLIC NOMINMAX)
endif()

set_target_properties(Deepity PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# --- pydeepity ------------------------------------------------------

if(DEEPITY_BUILD_PYTHON_BINDINGS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    pybind11_add_module(pydeepity bindings/pybinding.cpp)

    target_link_libraries(pydeepity PRIVATE Deepity)

    if(WIN32)
        target_compile_definitions(pydeepity PRIVATE NOMINMAX)
    endif()

    set_target_properties(pydeepity PROPERTIES
        PREFIX ""
    )

    # Modern cross-platform way to copy the compiled extension and required runtime libs
    add_custom_command(TARGET pydeepity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        $<TARGET_FILE:pydeepity>
        ${CMAKE_SOURCE_DIR}/pydeepity/
    )
    add_custom_command(TARGET pydeepity POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:pydeepity>>,copy_if_different,true>
        $<TARGET_RUNTIME_DLLS:pydeepity>
        ${CMAKE_SOURCE_DIR}/pydeepity/
        COMMAND_EXPAND_LISTS
    )

    install(TARGETS pydeepity
        LIBRARY DESTINATION pydeepity
        RUNTIME DESTINATION pydeepity
    )
endif()

# --- Tests ----------------------------------------------------------

if(DEEPITY_BUILD_TESTS)
    add_executable(DeepityTests tests/tSimpleConvVerify.cpp)

    target_link_libraries(DeepityTests PRIVATE
        Deepity
        benchmark::benchmark
    )

    set_target_properties(DeepityTests PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    if(WIN32)
        add_custom_command(TARGET DeepityTests POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:DeepityTests>>,copy_if_different,true>
            $<TARGET_RUNTIME_DLLS:DeepityTests>
            $<TARGET_FILE_DIR:DeepityTests>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

# --- Profiling ------------------------------------------------------

if(DEEPITY_BUILD_TESTS)
    add_library(DeepityProfiled STATIC
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

    target_include_directories(DeepityProfiled PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${BLAS_INCLUDE_DIR}
    )
    target_include_directories(DeepityProfiled SYSTEM PUBLIC
        ${sleef_SOURCE_DIR}/include
        ${sleef_BINARY_DIR}/include
    )
    target_link_libraries(DeepityProfiled PUBLIC
        ${BLAS_LIBRARIES}
        OpenMP::OpenMP_CXX
        sleef
    )

    target_compile_definitions(DeepityProfiled PUBLIC SLEEF_STATIC_LIBS PCN_PROFILE)

    if(WIN32)
        target_compile_definitions(DeepityProfiled PUBLIC NOMINMAX)
    endif()

    if(MSVC)
        target_compile_options(DeepityProfiled PRIVATE /O2 /fp:fast /openmp:llvm)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(DeepityProfiled PRIVATE -O3 -ffast-math -fvectorize -fslp-vectorize)
    else()
        target_compile_options(DeepityProfiled PRIVATE -O3 -ffast-math -ftree-vectorize -ftree-slp-vectorize)
    endif()

    add_executable(DeepityProfile tests/tProfile.cpp)
    target_link_libraries(DeepityProfile PRIVATE DeepityProfiled)
    target_compile_definitions(DeepityProfile PRIVATE PCN_PROFILE)

    set_target_properties(DeepityProfile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    if(WIN32)
        add_custom_command(TARGET DeepityProfile POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:DeepityProfile>>,copy_if_different,true>
            $<TARGET_RUNTIME_DLLS:DeepityProfile>
            $<TARGET_FILE_DIR:DeepityProfile>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()
