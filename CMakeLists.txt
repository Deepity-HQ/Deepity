# 3.28 is Halide's own declared floor (see DEEPITY_USE_HALIDE below);
# CMake 4.x is what's actually installed on every machine this project
# builds on today, so this costs nothing in practice.
cmake_minimum_required(VERSION 3.28)

# Must be set before project()/enable_language() -- CMake only consults
# CMAKE_<LANG>_COMPILER_LAUNCHER at that point, not later. Wraps every
# compile invocation (including inside FetchContent subbuilds like
# oneDNN/SLEEF/Highway, which don't override this themselves) so a
# second clean build of unchanged sources is a cache hit instead of a
# full recompile -- a no-op if ccache isn't installed.
find_program(DEEPITY_CCACHE_PROGRAM ccache)
if(DEEPITY_CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${DEEPITY_CCACHE_PROGRAM}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${DEEPITY_CCACHE_PROGRAM}")
endif()

project(deepity C CXX)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

option(DEEPITY_ENABLE_CUDA "Build with CUDA support" ON)
option(DEEPITY_USE_CUDNN "Use cuDNN for ConvPCLayer-family convolutions instead of the im2col+GEMM path (CUDA only; falls back automatically if cuDNN isn't found)" ON)
option(DEEPITY_BUILD_PYTHON_BINDINGS "Build the Python bindings target" ON)
# Recommended if on Intel CPUs
option(DEEPITY_USE_MKL "Use Intel MKL instead of OpenBLAS (Intel CPUs only -- MKL has a documented history of deliberately worse performance on non-Intel CPUs via runtime dispatch)" OFF)
# Fuses SimplePCLayer's MatMul + AddBiasBroadcast into one oneDNN
# matmul-with-bias call (CPU only; falls back to the existing
# separate-call path automatically if OFF or oneDNN isn't found). ON by
# default -- this is the GEMM path every benchmark number in the README
# was measured against.
option(DEEPITY_USE_ONEDNN "Use Intel oneDNN for a fused GEMM+bias epilogue, avoiding AddBiasBroadcast's separate memory pass" ON)
# Building oneDNN from source (the default, via FetchContent below) takes
# ~90 minutes of single-threaded compile time by itself (measured via
# .ninja_log on this project's own CI-equivalent build -- one file,
# ref_rnn.cpp, alone costs 250s, despite DNNL_ENABLE_PRIMITIVE=MATMUL
# below not needing RNN at all; oneDNN's own build doesn't exclude it
# from compilation, only from the runtime primitive table). ON trades
# that away for a system-installed, dynamically-linked oneDNN instead
# (e.g. Fedora/RHEL's `onednn-devel` package) via find_package(dnnl
# CONFIG) -- falls back to the FetchContent source build automatically
# if not found. Meant for local dev iteration, not for the wheel-building
# CI job, which wants the fully static, self-contained build this
# defaults to.
option(DEEPITY_ONEDNN_SYSTEM "Prefer a system-installed oneDNN (via find_package) over building one from source" OFF)
# Experimental: JIT-compiles shape-specialized small-matrix GEMM kernels
# for SimplePCLayer's feedback and weight-gradient GEMMs (both bias-free,
# unlike the mu GEMM above), avoiding cblas_sgemm's generic-large-GEMM
# dispatch overhead at these small shapes (CPU only; falls back to the
# existing MatMul() path automatically if OFF or dispatch fails).
option(DEEPITY_USE_LIBXSMM "Use libxsmm for small-matrix GEMMs instead of cblas_sgemm (experimental)" OFF)
# Experimental, measured SLOWER -- kept OFF by default, see below. Fuses
# SimplePCLayer::UpdateState()'s feedback GEMM (feedbackScratch = e_above
# @ W) directly into its own elementwise state update (z +=
# ir*(f'(z)*feedbackScratch - e)) via an ahead-of-time-compiled Halide
# kernel, so feedbackScratch is never materialized to memory at all -- a
# structural fusion oneDNN/libxsmm can't express (both are GEMM
# dispatchers, not schedulable compilers). Building Halide itself requires
# LLVM (uses the system install via find_package, not a from-source build
# -- see cmake/PatchHalideForNewerLLVM.cmake's file comment for the one
# compatibility patch that needs).
#
# Verified correct (direct reference comparison at the real MNIST shapes,
# plus a finite-difference gradient check through a real UpdateWeights()
# call). Two rounds measured:
#
# Round 1 (plain RDom reduction, no tiling): ~4x SLOWER than the oneDNN
# baseline for this GEMM bucket alone (1.54s vs. 0.24s+0.13s combined, same
# 40-batch run) -- oneDNN's cache-blocked multi-threaded GEMM comfortably
# beat an untiled hand-scheduled reduction, same as the libxsmm result
# above.
#
# Round 2 (register-tiled schedule adapted from Halide's own
# test/performance/matrix_multiplication.cpp, see FeedbackUpdateGenerator.cpp):
# closed almost all of that gap in isolation -- a standalone GEMM-only probe
# at the real (M,N,K) shapes measured this schedule at parity with MKL
# single-threaded and only ~1.6-2x behind multi-threaded. But wiring it into
# the real training loop made things *worse* than round 1's raw number would
# suggest (total time ~1.8-1.9x the oneDNN baseline), and crucially degraded
# *unrelated* oneDNN GEMM calls elsewhere in the same loop (muGemmSeconds,
# which never touches Halide, rose ~40-60%) -- i.e. the regression is not
# (only) schedule quality anymore, it's Halide's own thread pool and
# oneDNN's thread pool oversubscribing the same cores when interleaved in
# one hot loop. Forcing Halide down to 1 thread (HL_NUM_THREADS=1) restored
# the unrelated oneDNN buckets to baseline but made the fused GEMM itself
# ~13x slower solo -- every thread-count tried nets out worse than baseline
# end to end. A real fix would mean routing Halide's parallel-for through
# oneDNN's own thread pool instead of spinning up a second one
# (halide_do_par_for override) -- a substantially bigger undertaking than
# schedule tuning, not attempted yet.
#
# Kept as an experimental, off-by-default option (same convention as
# DEEPITY_USE_LIBXSMM above) rather than deleted. CPU only; falls back to
# the existing separate-calls path automatically if OFF, dispatch fails,
# the activation isn't SIGMOID (the only one with an AOT kernel today), or
# nextSize < 64 (the tiled schedule's reduction-tile overhead doesn't
# amortize for the output layer's short nextSize=10 feedback GEMM).
option(DEEPITY_USE_HALIDE "Use an AOT Halide kernel to fuse the feedback GEMM with its state update (experimental, measured slower than the oneDNN baseline)" OFF)

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

# check_ipo_supported() only confirms the COMPILER can emit LTO objects --
# it says nothing about whether the LINKER driving the final executable
# can read them. Clang emits LLVM bitcode, which GNU ld (bfd), the
# default system linker on most Linux distros, cannot read at all
# ("file format not recognized") -- only lld (or gold+LLVMgold.so) can.
# GCC's own bitcode format has the same bfd-plugin requirement but bfd
# distros normally ship GCC's LTO plugin by default, which is why this
# only bites Clang. Without this check, a machine where Clang happens to
# be the detected compiler links every executable target successfully
# right up until this exact failure, with no earlier signal.
set(DEEPITY_IPO_DISABLED_BY_LINKER_CHECK FALSE)
if(DEEPITY_IPO_SUPPORTED AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    find_program(DEEPITY_LLD_PROGRAM NAMES ld.lld lld)
    if(DEEPITY_LLD_PROGRAM)
        message(STATUS "Clang + LTO detected: using lld (${DEEPITY_LLD_PROGRAM}) as the "
            "linker, since the default system linker can't read Clang's LTO bitcode.")
        add_link_options(-fuse-ld=lld)
    else()
        set(DEEPITY_IPO_SUPPORTED FALSE)
        set(DEEPITY_IPO_DISABLED_BY_LINKER_CHECK TRUE)
        message(WARNING "Clang was detected as the compiler, but lld (the only linker "
            "that can read Clang's LTO bitcode objects on this platform) was not found. "
            "Disabling LTO for this build -- otherwise every executable target would "
            "fail to link with \"file format not recognized\". Install lld (e.g. "
            "'dnf install lld') and reconfigure to get LTO working with Clang.")
    endif()
endif()

if(NOT DEEPITY_IPO_SUPPORTED AND NOT DEEPITY_IPO_DISABLED_BY_LINKER_CHECK)
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
        if(DEEPITY_CCACHE_PROGRAM)
            set(CMAKE_CUDA_COMPILER_LAUNCHER "${DEEPITY_CCACHE_PROGRAM}")
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

# cuDNN ships separately from the CUDA toolkit itself (no CUDAToolkit::
# imported target for it), so it needs its own find_path/find_library --
# checked against an HPC module's own CUDNN_ROOT env var first (Lmod
# modules on clusters like OSC commonly export one, rather than dropping
# cudnn into the CUDA toolkit's own tree), then a -DCUDNN_ROOT=<path>
# override, then the CUDA toolkit's install tree, then standard system
# paths.
set(DEEPITY_CUDNN_ENABLED OFF)
if(DEEPITY_CUDA_ENABLED AND DEEPITY_USE_CUDNN)
    find_path(DEEPITY_CUDNN_INCLUDE_DIR cudnn.h
        HINTS $ENV{CUDNN_ROOT} ${CUDNN_ROOT} ${CUDAToolkit_INCLUDE_DIRS}
        PATH_SUFFIXES include)
    find_library(DEEPITY_CUDNN_LIBRARY cudnn
        HINTS $ENV{CUDNN_ROOT} ${CUDNN_ROOT} ${CUDAToolkit_LIBRARY_DIR}
        PATH_SUFFIXES lib lib64 lib/x64)
    if(DEEPITY_CUDNN_INCLUDE_DIR AND DEEPITY_CUDNN_LIBRARY)
        set(DEEPITY_CUDNN_ENABLED ON)
        message(STATUS "cuDNN found: ${DEEPITY_CUDNN_LIBRARY}")
    else()
        message(WARNING "DEEPITY_USE_CUDNN is ON, but cuDNN was not found "
            "(set -DCUDNN_ROOT=<path> if it's installed somewhere "
            "nonstandard). Falling back to the im2col+GEMM convolution path.")
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

            if(UNIX AND NOT APPLE)
                # MKL's default threading layer links against Intel's OWN
                # OpenMP runtime (libiomp5), a SEPARATE oneAPI component
                # (.../compiler/<ver>/lib) from MKL itself
                # (.../mkl/<ver>/lib) -- pulled in via MKL's own link
                # flags rather than an IMPORTED_LOCATION-bearing CMake
                # target, so CMake's automatic build-tree RPATH
                # propagation (the mechanism that already puts MKL's own
                # lib dir on every linked binary's RPATH for free, no
                # code needed) never sees it. Without this, EVERY binary
                # linking MKL -- pydeepity and every C++ test executable
                # alike -- fails at load time ("libiomp5.so: cannot open
                # shared object file") unless LD_LIBRARY_PATH happens to
                # be set by hand. Same class of gap DEEPITY_OMP_DLL above
                # already works around for Windows' OpenMP DLL.
                if(NOT DEEPITY_IOMP5_LIB)
                    get_filename_component(_mkl_cmake_dir "${MKL_DIR}" DIRECTORY)
                    get_filename_component(_mkl_lib_dir "${_mkl_cmake_dir}" DIRECTORY)
                    get_filename_component(_mkl_ver_dir "${_mkl_lib_dir}" DIRECTORY)
                    get_filename_component(_mkl_root_dir "${_mkl_ver_dir}" DIRECTORY)
                    get_filename_component(_oneapi_root "${_mkl_root_dir}" DIRECTORY)

                    file(GLOB _oneapi_compiler_lib_dirs "${_oneapi_root}/compiler/*/lib")
                    find_library(DEEPITY_IOMP5_LIB
                        NAMES iomp5
                        PATHS "${_oneapi_root}/compiler/latest/lib" ${_oneapi_compiler_lib_dirs}
                        NO_DEFAULT_PATH
                    )
                    if(NOT DEEPITY_IOMP5_LIB)
                        # Normal system search, covers a non-oneAPI MKL
                        # install (e.g. a distro package) where libiomp5
                        # might already be on the default linker path.
                        find_library(DEEPITY_IOMP5_LIB NAMES iomp5)
                    endif()
                endif()

                if(DEEPITY_IOMP5_LIB)
                    get_filename_component(DEEPITY_IOMP5_LIB_DIR "${DEEPITY_IOMP5_LIB}" DIRECTORY)
                    message(STATUS "Intel OpenMP runtime (libiomp5) resolved to: ${DEEPITY_IOMP5_LIB}")
                    # CMake's own intended global knob for exactly this --
                    # unlike the per-target compile-flag scoping used
                    # elsewhere in this file for ISA portability, RPATH
                    # has no "defeats runtime SIMD dispatch" concern, so a
                    # directory-wide default here is the right tool, not
                    # a repeat of that earlier anti-pattern.
                    list(APPEND CMAKE_BUILD_RPATH "${DEEPITY_IOMP5_LIB_DIR}")
                    list(APPEND CMAKE_INSTALL_RPATH "${DEEPITY_IOMP5_LIB_DIR}")
                else()
                    message(WARNING "DEEPITY_USE_MKL is ON, but Intel's OpenMP runtime "
                        "(libiomp5) could not be located automatically. Built "
                        "binaries (including pydeepity) will fail to load at "
                        "runtime unless libiomp5's directory is on the loader's "
                        "search path (e.g. LD_LIBRARY_PATH), or reconfigure with "
                        "-DDEEPITY_IOMP5_LIB=<path/to/libiomp5.so>.")
                endif()
            endif()
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

        if(UNIX AND NOT APPLE)
            # Some distros (Fedora/RHEL) package OpenBLAS as separate
            # serial/pthread/openmp builds behind one generic
            # "libopenblas.so" alternative, which find_package(BLAS)
            # above resolves blindly to whatever that alternative
            # currently points to. On this project's own dev machine
            # that silently resolved to the SERIAL build: every GEMM
            # call -- the majority of a train step's wall-clock time --
            # ran on a single core while this engine's own OpenMP
            # elementwise kernels used several (confirmed via
            # openblas_get_num_threads() returning 1 regardless of what
            # was requested). Prefer an explicitly-named threaded
            # variant over whatever the generic alternative happens to
            # point to. OpenMP-threaded, not pthread, to match this
            # project's own threading model instead of running two
            # separate thread pools against each other.
            find_library(DEEPITY_OPENBLAS_THREADED_LIB NAMES openblaso)
            if(DEEPITY_OPENBLAS_THREADED_LIB)
                message(STATUS "Using explicitly multi-threaded OpenBLAS: "
                    "${DEEPITY_OPENBLAS_THREADED_LIB} (overriding find_package(BLAS)'s "
                    "generic result: ${BLAS_LIBRARIES})")
                set(BLAS_LIBRARIES "${DEEPITY_OPENBLAS_THREADED_LIB}")
            else()
                message(WARNING "Could not find an explicitly multi-threaded OpenBLAS "
                    "variant (e.g. libopenblaso.so); falling back to find_package(BLAS)'s "
                    "generic result: ${BLAS_LIBRARIES}. If that resolves to a SERIAL "
                    "OpenBLAS build, every GEMM call runs on a single CPU core -- verify "
                    "with openblas_get_num_threads() if train steps seem slow.")
            endif()
        endif()

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

if(DEEPITY_USE_ONEDNN)
    set(DEEPITY_DNNL_RESOLVED FALSE)

    if(DEEPITY_ONEDNN_SYSTEM)
        find_package(dnnl CONFIG QUIET)
        if(TARGET DNNL::dnnl)
            message(STATUS "Using system oneDNN (found via find_package(dnnl)).")
            set(DEEPITY_DNNL_TARGET DNNL::dnnl)
            set(DEEPITY_DNNL_RESOLVED TRUE)
        else()
            message(WARNING "DEEPITY_ONEDNN_SYSTEM is ON, but no system oneDNN "
                "install was found (e.g. 'dnf install onednn-devel' on Fedora/RHEL). "
                "Falling back to building oneDNN from source.")
        endif()
    endif()

    if(NOT DEEPITY_DNNL_RESOLVED)
        # Only the forward matmul primitive is needed -- Deepity computes
        # its own backward/feedback pass manually, never via a oneDNN
        # primitive -- so both the primitive set and the workload
        # (training-specific primitives need backward variants too) are
        # restricted to keep the build small.
        set(DNNL_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(DNNL_BUILD_TESTS OFF CACHE BOOL "" FORCE)
        set(DNNL_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
        set(DNNL_ENABLE_WORKLOAD "INFERENCE" CACHE STRING "" FORCE)
        set(DNNL_ENABLE_PRIMITIVE "MATMUL" CACHE STRING "" FORCE)
        # Graph API (whole-subgraph pattern fusion) can't coexist with
        # restricting ONEDNN_ENABLE_PRIMITIVE -- not needed here, we're
        # only calling the matmul primitive directly.
        set(ONEDNN_BUILD_GRAPH OFF CACHE BOOL "" FORCE)
        set(DNNL_GPU_RUNTIME "NONE" CACHE STRING "" FORCE)
        # Shares Intel's iomp5 runtime with MKL rather than standing up a
        # third thread pool alongside OpenBLAS's own (see this session's
        # earlier, ultimately-false lead chasing oversubscription via perf).
        set(DNNL_CPU_RUNTIME "OMP" CACHE STRING "" FORCE)

        FetchContent_Declare(
            onednn
            GIT_REPOSITORY https://github.com/oneapi-src/oneDNN.git
            GIT_TAG v3.13.4
        )
        FetchContent_MakeAvailable(onednn)
        set(DEEPITY_DNNL_TARGET dnnl)
    endif()
endif()

if(DEEPITY_USE_LIBXSMM)
    # libxsmm's own CMake configure step shells out to a Python script
    # (scripts/libxsmm_config.py) to generate libxsmm_version.h.
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    # Not needed and pulls in a gfortran requirement for nothing -- we
    # only use the C dispatch API.
    set(LIBXSMM_FORTRAN OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        libxsmm
        GIT_REPOSITORY https://github.com/libxsmm/libxsmm.git
        GIT_TAG 2.1.0
    )
    FetchContent_MakeAvailable(libxsmm)

    # Isolated into its own target, deliberately never linked against
    # MKL: libxsmm's CMake build (unlike its GNU Make build) hardcodes
    # LP64 (32-bit libxsmm_blasint) with no option to request ILP64
    # through this path, and its headers refuse to compile at all if
    # MKL_ILP64 is visible alongside that mismatch -- see
    # CPUBackendLibxsmm.h's file comment for the full reasoning. Mirrors
    # DeepityHighwayKernels' existing use of a separate target for the
    # same kind of per-translation-unit compilation isolation.
    add_library(DeepityLibxsmm STATIC src/backend/CPUBackendLibxsmm.cpp)
    target_include_directories(DeepityLibxsmm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    # OpenMP, not MKL: libxsmm's dispatched kernels are deliberately
    # single-threaded per call (see CPUBackendLibxsmm.cpp) -- the
    # caller supplies parallelism itself, here via a row-blocked
    # #pragma omp parallel for across the batch dimension.
    target_link_libraries(DeepityLibxsmm PRIVATE libxsmm::libxsmm OpenMP::OpenMP_CXX)
endif()

if(DEEPITY_USE_HALIDE)
    # Not needed and would otherwise pull in flatbuffers/wabt for nothing --
    # Deepity only uses Halide's AOT Generator workflow (plain
    # add_halide_library() static libs), never JIT, serialization, the
    # WASM backend, or the Python bindings.
    set(WITH_SERIALIZATION OFF CACHE BOOL "" FORCE)
    set(Halide_WASM_BACKEND OFF CACHE STRING "" FORCE)
    set(WITH_PYTHON_BINDINGS OFF CACHE BOOL "" FORCE)
    set(WITH_TESTS OFF CACHE BOOL "" FORCE)
    set(WITH_TUTORIALS OFF CACHE BOOL "" FORCE)
    set(WITH_DOCS OFF CACHE BOOL "" FORCE)

    FetchContent_Declare(
        halide
        GIT_REPOSITORY https://github.com/halide/Halide.git
        GIT_TAG v21.0.0
        PATCH_COMMAND ${CMAKE_COMMAND}
            "-DDEEPITY_HALIDE_SRC_DIR=${CMAKE_BINARY_DIR}/_deps/halide-src"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/PatchHalideForNewerLLVM.cmake"
        UPDATE_DISCONNECTED TRUE
    )
    FetchContent_MakeAvailable(halide)

    # The generator program: runs at BUILD time only (links Halide +
    # LLVM), never shipped. Its output (below) has no such dependency --
    # see IComputeBackend.h's TryFusedFeedbackUpdate comment.
    add_executable(DeepityHalideGenerators src/backend/halide/FeedbackUpdateGenerator.cpp)
    target_link_libraries(DeepityHalideGenerators PRIVATE Halide::Generator)

    add_halide_library(deepity_feedback_update_sigmoid
        FROM DeepityHalideGenerators
        GENERATOR deepity_feedback_update_sigmoid
        FUNCTION_NAME deepity_feedback_update_sigmoid
    )
endif()

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
        src/backend/cuda/CUDABackendCudnnConv.cu
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

if(DEEPITY_CUDNN_ENABLED)
    target_include_directories(Deepity PRIVATE ${DEEPITY_CUDNN_INCLUDE_DIR})
    target_link_libraries(Deepity PRIVATE ${DEEPITY_CUDNN_LIBRARY})
    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_CUDNN)
endif()

if(DEEPITY_USE_ONEDNN)
    target_link_libraries(Deepity PRIVATE ${DEEPITY_DNNL_TARGET})
    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_ONEDNN)
endif()

if(DEEPITY_USE_LIBXSMM)
    target_link_libraries(Deepity PRIVATE DeepityLibxsmm)
    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_LIBXSMM)
endif()

if(DEEPITY_USE_HALIDE)
    target_link_libraries(Deepity PRIVATE deepity_feedback_update_sigmoid)
    target_compile_definitions(Deepity PUBLIC DEEPITY_USE_HALIDE)
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
# Google Benchmark defaults this ON and applies -Werror/-pedantic-errors
# to ITS OWN sources -- on a newer Clang, that turns its (harmless,
# widely-supported) use of __COUNTER__ into a hard failure under
# -Wc2y-extensions, since __COUNTER__ isn't standardized until C2y. Not
# our code, not our warning bar -- just let this dependency build.
set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(benchmark)
# BENCHMARK_ENABLE_WERROR above doesn't cover this: benchmark's own
# CMakeLists.txt applies -pedantic-errors to itself UNCONDITIONALLY (not
# gated by that option), which alone is enough to promote the
# __COUNTER__ diagnostic above to a hard error. A later -Wno-* on the
# command line overrides an earlier -pedantic-errors for that specific
# category, so suppress it directly on the already-defined target
# rather than patching benchmark's vendored source.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(benchmark PRIVATE -Wno-c2y-extensions)
    target_compile_options(benchmark_main PRIVATE -Wno-c2y-extensions)
endif()

# --- Tests -----------------------------------------------------------

option(DEEPITY_BUILD_TESTS "Build test/verification executables" ON)

# Without this, every add_test() below is silently inert -- ctest reports
# "No tests were found" regardless of how many are registered, since no
# CTestTestfile.cmake is ever generated.
enable_testing()

if(DEEPITY_BUILD_TESTS)
    # Every target added below is appended here, then given the same
    # -march/-mtune (and, on MSVC, the same __AVX2__-style compile-time
    # defs) that Deepity/pydeepity get, in one loop at the end of this
    # block. These executables pull in Deepity's activation kernels as
    # `static inline` header functions, so each TU recompiles its own
    # copy -- without this, they silently fall back to the compiler's
    # untouched default target (SSE2-only on this toolchain) instead of
    # the arch build.py actually selected, making any benchmark here
    # measure the wrong kernel entirely.
    set(DEEPITY_TEST_TARGETS "")

    # --- Benchmarks (Google Benchmark, BENCHMARK_MAIN() -- perf only, not
    # registered with ctest since they have no pass/fail) -----------------

    add_executable(ActivationBenchmark tests/tActivations.cpp)
    target_link_libraries(ActivationBenchmark PRIVATE Deepity benchmark::benchmark)
    set_target_properties(ActivationBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    list(APPEND DEEPITY_TEST_TARGETS ActivationBenchmark)

    add_executable(PCNBenchmark tests/tBenchmark.cpp)
    target_link_libraries(PCNBenchmark PRIVATE Deepity)
    set_target_properties(PCNBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    list(APPEND DEEPITY_TEST_TARGETS PCNBenchmark)

    add_executable(GflopsBenchmark tests/tGflopsBenchmark.cpp)
    target_link_libraries(GflopsBenchmark PRIVATE Deepity benchmark::benchmark)
    set_target_properties(GflopsBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    list(APPEND DEEPITY_TEST_TARGETS GflopsBenchmark)

    add_executable(ReadmeBenchmark tests/tReadme.cpp)
    target_link_libraries(ReadmeBenchmark PRIVATE Deepity benchmark::benchmark)
    set_target_properties(ReadmeBenchmark PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    list(APPEND DEEPITY_TEST_TARGETS ReadmeBenchmark)

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
    list(APPEND DEEPITY_TEST_TARGETS ConvDiagnose)

    # --- Correctness verification tests (CUDA-only: call
    # CreateBackend(DEVICE_GPU), which throws when built without CUDA) -----

    if(DEEPITY_CUDA_ENABLED)
        add_executable(CUDALaunchErrorVerify tests/tCUDALaunchErrorVerify.cpp)
        target_link_libraries(CUDALaunchErrorVerify PRIVATE Deepity)
        set_target_properties(CUDALaunchErrorVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
        list(APPEND DEEPITY_TEST_TARGETS CUDALaunchErrorVerify)

        add_executable(CUDAIntRoundtripVerify tests/tCUDAIntRoundtripVerify.cpp)
        target_link_libraries(CUDAIntRoundtripVerify PRIVATE Deepity)
        set_target_properties(CUDAIntRoundtripVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
        list(APPEND DEEPITY_TEST_TARGETS CUDAIntRoundtripVerify)

        add_executable(CUDAIncrementCounterVerify tests/tCUDAIncrementCounterVerify.cpp)
        target_link_libraries(CUDAIncrementCounterVerify PRIVATE Deepity)
        set_target_properties(CUDAIncrementCounterVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
        list(APPEND DEEPITY_TEST_TARGETS CUDAIncrementCounterVerify)

        add_executable(CUDAIncrementCounterSyncVerify tests/tCUDAIncrementCounterSyncVerify.cpp)
        target_link_libraries(CUDAIncrementCounterSyncVerify PRIVATE Deepity)
        set_target_properties(CUDAIncrementCounterSyncVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
        list(APPEND DEEPITY_TEST_TARGETS CUDAIncrementCounterSyncVerify)

        add_executable(MatMulLargeAsymmetricVerify tests/tMatMulLargeAsymmetricVerify.cpp)
        target_link_libraries(MatMulLargeAsymmetricVerify PRIVATE Deepity)
        set_target_properties(MatMulLargeAsymmetricVerify PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )
        list(APPEND DEEPITY_TEST_TARGETS MatMulLargeAsymmetricVerify)
    endif()

    # --- Manual profiling tools (no pass/fail; not registered with ctest) -

    add_executable(DiscriminativeNetworkProfile tests/tDiscriminativeNetworkProfile.cpp)
    target_link_libraries(DiscriminativeNetworkProfile PRIVATE Deepity)
    set_target_properties(DiscriminativeNetworkProfile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    list(APPEND DEEPITY_TEST_TARGETS DiscriminativeNetworkProfile)

    add_executable(SimpleNetworkProfile tests/tSimpleNetworkProfile.cpp)
    target_link_libraries(SimpleNetworkProfile PRIVATE Deepity)
    set_target_properties(SimpleNetworkProfile PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    list(APPEND DEEPITY_TEST_TARGETS SimpleNetworkProfile)

    # Same arch/SIMD flags as Deepity/pydeepity (see the comment at the
    # top of this block) -- without this, every target above silently
    # compiles Deepity's header-only kernels against the compiler's
    # untouched default ISA instead of whatever build.py selected.
    foreach(DEEPITY_TEST_TARGET ${DEEPITY_TEST_TARGETS})
        if(DEEPITY_TARGET_COMPILE_OPTIONS)
            target_compile_options(${DEEPITY_TEST_TARGET} PRIVATE ${DEEPITY_TARGET_COMPILE_OPTIONS})
        endif()
        if(DEEPITY_TARGET_COMPILE_DEFS)
            target_compile_definitions(${DEEPITY_TEST_TARGET} PRIVATE ${DEEPITY_TARGET_COMPILE_DEFS})
        endif()
    endforeach()
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
