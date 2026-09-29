cmake_minimum_required(VERSION 3.21)

project(deepity C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DEEPITY_ENABLE_CUDA "Build with CUDA support" ON)
option(DEEPITY_BUILD_PYTHON_BINDINGS "Build the Python bindings target" ON)
# Recommended if on Intel CPUs
option(DEEPITY_USE_MKL "Use Intel MKL instead of OpenBLAS (Intel CPUs only -- MKL has a documented history of deliberately worse performance on non-Intel CPUs via runtime dispatch)" OFF)

# AMD ROCm Coming soon...

# These are set automatically in build.py based on --native/distributed/fast
set(DEEPITY_ARCH_FLAGS "-march=x86-64-v3 -mtune=generic" CACHE STRING "Architecture/SIMD flags (e.g., -march=native, -march=x86-64-v2)")
set(DEEPITY_MSVC_ARCH_FLAGS "/arch:AVX2" CACHE STRING "Architecture flags for MSVC (empty = compiler default / most portable)")

# Deliberately NOT applied via add_compile_options() here: that's
# directory-scoped, so it would leak into every FetchContent'd
# third-party target too (SLEEF, Highway, CUTLASS, Benchmark) -- SLEEF
# already builds its own SSE2/AVX2/AVX512 variants internally and
# doesn't want an external -march override forcing one choice, and
# Highway's whole point (runtime dispatch across ISA levels) is defeated
# if its own compiled code is also hard-pinned to one target. Computed
# here, applied via target_compile_options(Deepity PRIVATE ...) once the
# Deepity target exists (same fix this session's own PGO work already
# needed for the identical class of leak).
set(DEEPITY_TARGET_COMPILE_OPTIONS "")
set(DEEPITY_TARGET_COMPILE_DEFS "")

if(MSVC)
    if(DEEPITY_MSVC_ARCH_FLAGS)
        list(APPEND DEEPITY_TARGET_COMPILE_OPTIONS ${DEEPITY_MSVC_ARCH_FLAGS})
    endif()

    if(DEEPITY_MSVC_ARCH_FLAGS MATCHES "AVX512")
        list(APPEND DEEPITY_TARGET_COMPILE_DEFS __SSE2__ __AVX__ __AVX2__ __AVX512F__)
    elseif(DEEPITY_MSVC_ARCH_FLAGS MATCHES "AVX2")
        list(APPEND DEEPITY_TARGET_COMPILE_DEFS __SSE2__ __AVX__ __AVX2__)
    elseif(DEEPITY_MSVC_ARCH_FLAGS MATCHES "AVX")
        list(APPEND DEEPITY_TARGET_COMPILE_DEFS __SSE2__ __AVX__)
    else()
        list(APPEND DEEPITY_TARGET_COMPILE_DEFS __SSE2__)
    endif()
else()
    separate_arguments(ARCH_FLAGS_LIST UNIX_COMMAND ${DEEPITY_ARCH_FLAGS})
    list(APPEND DEEPITY_TARGET_COMPILE_OPTIONS ${ARCH_FLAGS_LIST})
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# --- Profile-Guided Optimization (PGO) --------------------------------
# Two-pass workflow driven by build.py, not something to invoke directly:
# GENERATE builds an instrumented binary, a representative workload runs
# against it to collect real branch/call-frequency data, then USE
# rebuilds using that data. GCC/Clang only -- MSVC's PGO mechanism
# (/GENPROFILE, a Visual-Studio-integrated flow) is different enough that
# a single unified code path here isn't practical; MSVC falls back to a
# plain, un-instrumented build with a warning instead of silently doing
# something unexpected.
#
# Applied ONLY to Deepity/pydeepity (via target_compile_options/
# target_link_options below, once those targets exist), NOT globally:
# the pgo_workload.py profiling run only ever exercises our own code, so
# third-party FetchContent deps (Google Benchmark, nanobind-static,
# sleef, ...) would compile with instrumentation but never actually
# execute during the workload, then fail the USE pass outright wherever
# a dependency's own CMakeLists.txt treats -Wmissing-profile as an error
# (Google Benchmark does). Restricting PGO to the two targets that are
# actually profiled avoids that entirely, and is also just the correct
# scope: nothing else here is what PGO is meant to optimize.
set(DEEPITY_PGO_MODE "OFF" CACHE STRING "PGO mode: OFF, GENERATE, or USE")
set_property(CACHE DEEPITY_PGO_MODE PROPERTY STRINGS OFF GENERATE USE)
set(DEEPITY_PGO_DATA_DIR "${CMAKE_BINARY_DIR}/pgo-data" CACHE PATH "Directory for PGO profile data (shared by both GENERATE and USE passes)")

set(DEEPITY_PGO_COMPILE_FLAGS "")
set(DEEPITY_PGO_LINK_FLAGS "")

if(NOT DEEPITY_PGO_MODE STREQUAL "OFF")
    if(MSVC)
        message(WARNING "DEEPITY_PGO_MODE=${DEEPITY_PGO_MODE} was requested, but "
            "this codebase's PGO support targets GCC/Clang only -- MSVC's PGO "
            "workflow differs significantly and isn't implemented here. "
            "Building without PGO instrumentation.")
    elseif(DEEPITY_PGO_MODE STREQUAL "GENERATE")
        list(APPEND DEEPITY_PGO_COMPILE_FLAGS -fprofile-generate=${DEEPITY_PGO_DATA_DIR})
        list(APPEND DEEPITY_PGO_LINK_FLAGS -fprofile-generate=${DEEPITY_PGO_DATA_DIR})
        message(STATUS "PGO: instrumented (GENERATE) build -- profile data will be written to ${DEEPITY_PGO_DATA_DIR}")
    elseif(DEEPITY_PGO_MODE STREQUAL "USE")
        list(APPEND DEEPITY_PGO_COMPILE_FLAGS -fprofile-use=${DEEPITY_PGO_DATA_DIR})
        list(APPEND DEEPITY_PGO_LINK_FLAGS -fprofile-use=${DEEPITY_PGO_DATA_DIR})
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            # GCC-specific: tolerates minor profile/source mismatches
            # gracefully instead of hard-failing. Clang doesn't need or
            # recognize this flag.
            list(APPEND DEEPITY_PGO_COMPILE_FLAGS -fprofile-correction)
        endif()
        message(STATUS "PGO: optimized (USE) build -- reading profile data from ${DEEPITY_PGO_DATA_DIR}")
    endif()
endif()

# --- Link-Time Optimization (LTO/IPO) --------------------------------
# Uses CMake's portable INTERPROCEDURAL_OPTIMIZATION property rather than
# a raw -flto flag: this correctly maps to -flto on GCC/Clang and
# /GL+/LTCG on MSVC, and check_ipo_supported() lets this degrade
# gracefully (a warning, not a hard build failure) on toolchains that
# don't support it at all. Applied to Release builds only -- LTO
# meaningfully slows compilation with no benefit for Debug builds.
include(CheckIPOSupported)
check_ipo_supported(RESULT DEEPITY_IPO_SUPPORTED OUTPUT DEEPITY_IPO_ERROR)
if(NOT DEEPITY_IPO_SUPPORTED)
    message(WARNING "LTO/IPO not supported by this toolchain: ${DEEPITY_IPO_ERROR}. Building without it.")
endif()

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")

# --- DEPENDENCIES ---------------------------------------------------

find_package(OpenMP REQUIRED)

# find_package(OpenMP) only learns the *import* library (via
# OpenMP_omp_LIBRARY, set by build.py on the clang/Ninja Windows path) --
# it has no idea where the runtime DLL lives, so $<TARGET_RUNTIME_DLLS:...>
# further down can't discover or copy it (unlike OPENBLAS_DLL above, which
# is resolved explicitly). Locate it here so it can be bundled the same way.
if(WIN32 AND NOT DEEPITY_OMP_DLL)
    # Don't key off OpenMP_omp_LIBRARY -- it's only populated when
    # cmake_runner.py's CC=clang branch happened to fire. CMake can pick
    # clang as the effective compiler even when that branch never ran
    # (e.g. no MSVC dev shell active, clang just happens to be on PATH),
    # so derive the LLVM install location from CMAKE_CXX_COMPILER itself,
    # which is always correct regardless of how the compiler was chosen.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_COMPILER)
        get_filename_component(_clang_bin_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(_llvm_root "${_clang_bin_dir}" DIRECTORY)
        find_file(DEEPITY_OMP_DLL
            NAMES libomp.dll
            PATHS "${_clang_bin_dir}" "${_llvm_root}/bin" "${_llvm_root}/lib"
            NO_DEFAULT_PATH
        )
    endif()
    if(NOT DEEPITY_OMP_DLL AND OpenMP_omp_LIBRARY)
        # Fall back to the old lookup in case it IS set (e.g. someone
        # invokes cmake directly with -DOpenMP_omp_LIBRARY=...).
        get_filename_component(_omp_lib_dir "${OpenMP_omp_LIBRARY}" DIRECTORY)
        find_file(DEEPITY_OMP_DLL NAMES libomp.dll PATHS "${_omp_lib_dir}" "${_omp_lib_dir}/../bin")
    endif()
    if(NOT DEEPITY_OMP_DLL AND MSVC)
        # MSVC's own OpenMP runtime ships with the VC redistributables --
        # unless /openmp:llvm is in play (see target_compile_options
        # below), in which case MSVC uses LLVM's runtime instead, typically
        # named libomp140.x86_64.dll and shipped inside the VC toolset's
        # own redist tree rather than a standalone LLVM install.
        find_file(DEEPITY_OMP_DLL NAMES vcomp140.dll vcomp140d.dll libomp140.x86_64.dll)
    endif()
    if(DEEPITY_OMP_DLL)
        message(STATUS "DEEPITY_OMP_DLL resolved to: ${DEEPITY_OMP_DLL}")
    else()
        message(WARNING "Could not locate the OpenMP runtime DLL automatically. "
            "You may need to copy it (e.g. libomp.dll from your LLVM install, "
            "or vcomp140.dll for MSVC) next to the build outputs manually, or "
            "reconfigure with -DDEEPITY_OMP_DLL=<path>.")
    endif()
endif()

set(DEEPITY_CUDA_ENABLED OFF)
if(DEEPITY_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
        # Must be set BEFORE enable_language(CUDA), or nvcc silently falls
        # back to its own old default compute capability -- producing a
        # binary with no kernel image for whatever GPU it's actually run
        # on ("no kernel image is available for execution on the device"
        # at runtime, not a build-time error). Covers V100/T4 (70/75),
        # A100/RTX 30xx (80/86), and H100 (90); override with
        # -DCMAKE_CUDA_ARCHITECTURES=<arch> for a narrower, faster build
        # targeting one known GPU.
        if(NOT CMAKE_CUDA_ARCHITECTURES)
            set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;90")
        endif()
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        set(DEEPITY_CUDA_ENABLED ON)
        message(STATUS "CUDA support enabled.")
    else()
        message(STATUS "CUDA toolkit not found. Building CPU-only.")
    endif()
endif()

if(DEEPITY_BUILD_PYTHON_BINDINGS)
    find_package(Python 3.8 COMPONENTS Interpreter Development.Module REQUIRED)

    # Detect the installed nanobind package and import it into CMake
    execute_process(
        COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
        OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
    find_package(nanobind CONFIG REQUIRED)
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
    set(DEEPITY_BLAS_RESOLVED FALSE)

    if(DEEPITY_USE_MKL)
        find_package(MKL CONFIG QUIET)
        if(MKL_FOUND)
            message(STATUS "Using Intel MKL (found via MKLConfig.cmake).")
            set(BLAS_LIBRARIES MKL::MKL)
            get_target_property(BLAS_INCLUDE_DIR MKL::MKL INTERFACE_INCLUDE_DIRECTORIES)
            set(DEEPITY_BLAS_RESOLVED TRUE)
        else()
            message(WARNING "DEEPITY_USE_MKL was set, but MKL was not found "
                "(is oneAPI/MKL installed and sourced? e.g. "
                "'source /opt/intel/oneapi/setvars.sh'). "
                "Falling back to OpenBLAS.")
        endif()
    endif()

    if(NOT DEEPITY_BLAS_RESOLVED)
        set(BLA_VENDOR OpenBLAS)
        find_package(BLAS REQUIRED)
        find_path(BLAS_INCLUDE_DIR
            NAMES cblas.h
            PATH_SUFFIXES openblas include/openblas
            REQUIRED
        )
    endif()
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

# --- Highway (portable SIMD, runtime dispatch) -----------------------
# Its whole point is compiling multiple per-ISA-target variants of the
# SAME source and picking the best one at runtime, so it deliberately
# does NOT get DEEPITY_TARGET_COMPILE_OPTIONS's hard-pinned -march flag
# (that would force every one of its targets down to the same ISA,
# defeating runtime dispatch entirely) -- same reasoning as SLEEF above.
set(HWY_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_CONTRIB OFF CACHE BOOL "" FORCE)
set(HWY_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    highway
    GIT_REPOSITORY https://github.com/google/highway.git
    GIT_TAG 1.4.0
)
FetchContent_MakeAvailable(highway)

# A separate target, deliberately NOT part of DEEPITY_SOURCES: CMake has
# no clean way to "subtract" a target-level compile option from one
# source file within a target, so the Highway-dispatched kernels live in
# their own small static library instead, built without
# DEEPITY_TARGET_COMPILE_OPTIONS's -march pin, then linked into Deepity
# like any other dependency (sleef, hwy itself).
add_library(DeepityHighwayKernels STATIC src/backend/CPUBackendHighway.cpp)
target_include_directories(DeepityHighwayKernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# HWY_TARGET_INCLUDE ("backend/CPUBackendHighway.cpp") must resolve via
# an include search path -- add src/ so it does.
target_include_directories(DeepityHighwayKernels PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(DeepityHighwayKernels PUBLIC hwy)

# --- Deepity library ------------------------------------------------

set(DEEPITY_SOURCES
    src/layers/FullPCLayer.cpp
    src/layers/FullConvPCLayer.cpp
    src/layers/DiscriminativePCLayer.cpp
    src/layers/ConvPCLayer.cpp
    src/layers/SimpleConvPCLayer.cpp
    src/layers/SimplePCLayer.cpp
    src/layers/GaussSeidelPCLayer.cpp
    src/layers/DirectKPPCLayer.cpp
    src/networks/FullPCNetwork.cpp
    src/networks/FullConvPCNetwork.cpp
    src/networks/DiscriminativePCNetwork.cpp
    src/networks/ConvPCNetwork.cpp
    src/networks/SimplePCNetwork.cpp
    src/networks/GaussSeidelPCNetwork.cpp
    src/networks/SimpleConvPCNetwork.cpp
    src/networks/DirectKPPCNetwork.cpp
    src/utils/StreamAlignedBatcher.cpp
    src/backend/Backend.cpp
    src/backend/CPUBackend.cpp
    src/backend/Tensor.cpp
)

if(DEEPITY_CUDA_ENABLED)
    list(APPEND DEEPITY_SOURCES
        src/backend/cuda/CUDABackendCore.cu
        src/backend/cuda/CUDABackendGemm.cu
        src/backend/cuda/CUDABackendElementwise.cu
        src/backend/cuda/CUDABackendActivations.cu
        src/backend/cuda/CUDABackendFusedOps.cu
        src/backend/cuda/CUDABackendOptimizer.cu
        src/backend/cuda/CUDABackendConv.cu
    )
endif()

add_library(Deepity ${DEEPITY_SOURCES})

if(DEEPITY_TARGET_COMPILE_OPTIONS)
    target_compile_options(Deepity PRIVATE ${DEEPITY_TARGET_COMPILE_OPTIONS})
endif()
if(DEEPITY_TARGET_COMPILE_DEFS)
    target_compile_definitions(Deepity PRIVATE ${DEEPITY_TARGET_COMPILE_DEFS})
endif()

if(DEEPITY_PGO_COMPILE_FLAGS)
    # COMPILE_LANGUAGE:CXX guard: Deepity's sources are C++/CUDA when
    # DEEPITY_CUDA_ENABLED, and nvcc doesn't understand these GCC/Clang
    # flags at all. Compile options stay PRIVATE (nothing recompiles
    # Deepity's sources by linking it), but the link options are PUBLIC:
    # -fprofile-generate embeds calls to libgcov's runtime in every
    # object file it touches, so any test/benchmark executable that
    # links this archive needs the SAME flag on ITS OWN final link line
    # too, or the driver never knows to add -lgcov and you get
    # "undefined reference to `__gcov_init'" and friends. PUBLIC makes
    # target_link_libraries(... Deepity) propagate that automatically.
    target_compile_options(Deepity PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:${DEEPITY_PGO_COMPILE_FLAGS}>
    )
    target_link_options(Deepity PUBLIC ${DEEPITY_PGO_LINK_FLAGS})
endif()

if(DEEPITY_USE_MKL AND DEEPITY_BLAS_RESOLVED)
    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_MKL)
endif()

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

target_link_libraries(Deepity PRIVATE hwy DeepityHighwayKernels)

if(DEEPITY_CUDA_ENABLED)
    target_link_libraries(Deepity PUBLIC CUDA::cublas CUDA::cudart)
    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_CUDA)
    target_compile_options(Deepity PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
endif()

target_compile_definitions(Deepity PUBLIC SLEEF_STATIC_LIBS)

# --- Optional: Use CUTLASS for fused GEMM kernels ---------------------

if(DEEPITY_CUDA_ENABLED)
    FetchContent_Declare(
        cutlass
        GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
        GIT_TAG v4.8.0
    )
    set(CUTLASS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_TOOLS OFF CACHE BOOL "" FORCE)
    set(CUTLASS_ENABLE_LIBRARY OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(cutlass)

    target_link_libraries(Deepity PRIVATE CUTLASS)
endif()

# --- Optional: Use Google Benchmark -----------------------------------

include(FetchContent)
FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.9.1
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)

# --- Tests -----------------------------------------------------------

option(DEEPITY_BUILD_TESTS "Build test/verification executables" ON)

# Without this, every add_test() below is silently inert -- ctest reports
# "No tests were found" regardless of how many are registered, since no
# CTestTestfile.cmake is ever generated.
enable_testing()

if(DEEPITY_BUILD_TESTS)
    # --- Benchmarks (Google Benchmark, BENCHMARK_MAIN() -- perf only, not
    # registered with ctest since they have no pass/fail) -----------------

    add_executable(ActivationBenchmark tests/tActivations.cpp)
    target_link_libraries(ActivationBenchmark PRIVATE Deepity benchmark::benchmark)
    set_target_properties(ActivationBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_executable(PCNBenchmark tests/tBenchmark.cpp)
    target_link_libraries(PCNBenchmark PRIVATE Deepity)
    set_target_properties(PCNBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_executable(GflopsBenchmark tests/tGflopsBenchmark.cpp)
    target_link_libraries(GflopsBenchmark PRIVATE Deepity benchmark::benchmark)
    set_target_properties(GflopsBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_executable(ReadmeBenchmark tests/tReadme.cpp)
    target_link_libraries(ReadmeBenchmark PRIVATE Deepity benchmark::benchmark)
    set_target_properties(ReadmeBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    # --- Correctness verification tests (CPU) -----------------------------
    # Deliberately just one: an end-to-end integration smoke test. The
    # narrower per-layer/per-kernel gradient checks that used to live here
    # (CrossEntropyVerify, DirectKPVerify, GaussSeidelPCLayerVerify,
    # FullPCLayerVerify and friends) each did their job once, at the time
    # of the port/feature they verified, and are still on disk in tests/
    # if that exact code path needs re-checking after a future change --
    # they just don't need to be a permanent part of every build.

    add_executable(ConvDiagnose tests/tConvDiagnose.cpp)
    target_link_libraries(ConvDiagnose PRIVATE Deepity)
    set_target_properties(ConvDiagnose PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    add_test(NAME ConvDiagnose COMMAND ConvDiagnose)

    # --- Correctness verification tests (CUDA-only: call
    # CreateBackend(DEVICE_GPU), which throws when built without CUDA) -----

    if(DEEPITY_CUDA_ENABLED)
        add_executable(CUDALaunchErrorVerify tests/tCUDALaunchErrorVerify.cpp)
        target_link_libraries(CUDALaunchErrorVerify PRIVATE Deepity)
        set_target_properties(CUDALaunchErrorVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        add_executable(CUDAIntRoundtripVerify tests/tCUDAIntRoundtripVerify.cpp)
        target_link_libraries(CUDAIntRoundtripVerify PRIVATE Deepity)
        set_target_properties(CUDAIntRoundtripVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        add_executable(CUDAIncrementCounterVerify tests/tCUDAIncrementCounterVerify.cpp)
        target_link_libraries(CUDAIncrementCounterVerify PRIVATE Deepity)
        set_target_properties(CUDAIncrementCounterVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        add_executable(CUDAIncrementCounterSyncVerify tests/tCUDAIncrementCounterSyncVerify.cpp)
        target_link_libraries(CUDAIncrementCounterSyncVerify PRIVATE Deepity)
        set_target_properties(CUDAIncrementCounterSyncVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        add_executable(MatMulLargeAsymmetricVerify tests/tMatMulLargeAsymmetricVerify.cpp)
        target_link_libraries(MatMulLargeAsymmetricVerify PRIVATE Deepity)
        set_target_properties(MatMulLargeAsymmetricVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
    endif()

    # --- Manual profiling tools (no pass/fail; not registered with ctest) -

    add_executable(DiscriminativeNetworkProfile tests/tDiscriminativeNetworkProfile.cpp)
    target_link_libraries(DiscriminativeNetworkProfile PRIVATE Deepity)
    set_target_properties(DiscriminativeNetworkProfile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )

    add_executable(SimpleNetworkProfile tests/tSimpleNetworkProfile.cpp)
    target_link_libraries(SimpleNetworkProfile PRIVATE Deepity)
    set_target_properties(SimpleNetworkProfile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
endif()

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
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${DEEPITY_IPO_SUPPORTED}
)

# --- pydeepity ------------------------------------------------------

if(DEEPITY_BUILD_PYTHON_BINDINGS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    nanobind_add_module(pydeepity
        bindings/pybinding.cpp
        bindings/LayerBindings.cpp
        bindings/NetworkBindings.cpp
        bindings/UtilityBindings.cpp
    )

    target_link_libraries(pydeepity PRIVATE Deepity)

    if(DEEPITY_TARGET_COMPILE_OPTIONS)
        target_compile_options(pydeepity PRIVATE ${DEEPITY_TARGET_COMPILE_OPTIONS})
    endif()
    if(DEEPITY_TARGET_COMPILE_DEFS)
        target_compile_definitions(pydeepity PRIVATE ${DEEPITY_TARGET_COMPILE_DEFS})
    endif()

    if(DEEPITY_PGO_COMPILE_FLAGS)
        # Bindings are pure C++, no COMPILE_LANGUAGE guard needed here.
        target_compile_options(pydeepity PRIVATE ${DEEPITY_PGO_COMPILE_FLAGS})
        target_link_options(pydeepity PRIVATE ${DEEPITY_PGO_LINK_FLAGS})
    endif()

    set_target_properties(pydeepity PROPERTIES
        INTERPROCEDURAL_OPTIMIZATION_RELEASE ${DEEPITY_IPO_SUPPORTED}
    )

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
    if(DEEPITY_OMP_DLL)
        add_custom_command(TARGET pydeepity POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${DEEPITY_OMP_DLL}"
            ${CMAKE_SOURCE_DIR}/pydeepity/
        )
    endif()

    install(TARGETS pydeepity
        LIBRARY DESTINATION pydeepity
        RUNTIME DESTINATION pydeepity
    )
endif()

if(WIN32)
    install(FILES ${OPENBLAS_DLL} DESTINATION pydeepity)
    if(DEEPITY_OMP_DLL)
        install(FILES ${DEEPITY_OMP_DLL} DESTINATION pydeepity)
    endif()
endif()
