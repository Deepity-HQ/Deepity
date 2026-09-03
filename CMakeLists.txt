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
    else()
        add_compile_definitions(__SSE2__)
    endif()
else()
    separate_arguments(ARCH_FLAGS_LIST UNIX_COMMAND ${DEEPITY_ARCH_FLAGS})
    add_compile_options(${ARCH_FLAGS_LIST})
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
set(DEEPITY_PGO_MODE "OFF" CACHE STRING "PGO mode: OFF, GENERATE, or USE")
set_property(CACHE DEEPITY_PGO_MODE PROPERTY STRINGS OFF GENERATE USE)
set(DEEPITY_PGO_DATA_DIR "${CMAKE_BINARY_DIR}/pgo-data" CACHE PATH "Directory for PGO profile data (shared by both GENERATE and USE passes)")

if(NOT DEEPITY_PGO_MODE STREQUAL "OFF")
    if(MSVC)
        message(WARNING "DEEPITY_PGO_MODE=${DEEPITY_PGO_MODE} was requested, but "
            "this codebase's PGO support targets GCC/Clang only -- MSVC's PGO "
            "workflow differs significantly and isn't implemented here. "
            "Building without PGO instrumentation.")
    elseif(DEEPITY_PGO_MODE STREQUAL "GENERATE")
        add_compile_options(-fprofile-generate=${DEEPITY_PGO_DATA_DIR})
        add_link_options(-fprofile-generate=${DEEPITY_PGO_DATA_DIR})
        message(STATUS "PGO: instrumented (GENERATE) build -- profile data will be written to ${DEEPITY_PGO_DATA_DIR}")
    elseif(DEEPITY_PGO_MODE STREQUAL "USE")
        add_compile_options(-fprofile-use=${DEEPITY_PGO_DATA_DIR})
        add_link_options(-fprofile-use=${DEEPITY_PGO_DATA_DIR})
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            # GCC-specific: tolerates minor profile/source mismatches
            # gracefully instead of hard-failing. Clang doesn't need or
            # recognize this flag.
            add_compile_options(-fprofile-correction)
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

if(DEEPITY_ENABLE_CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
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
        # Modern, recommended integration: Intel's own MKLConfig.cmake,
        # present if oneAPI/MKL is properly installed and sourced (e.g.
        # via `source /opt/intel/oneapi/setvars.sh`). Exposes a single
        # target (MKL::MKL) with both include dirs and link libraries.
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

# --- Deepity library ------------------------------------------------

add_library(Deepity
    src/DiscriminativePCLayer.cpp
    src/RBLayer.cpp
    src/ConvPCLayer.cpp
    src/SimpleConvPCLayer.cpp
    src/SimplePCLayer.cpp
    src/GaussSeidelPCLayer.cpp
    src/DirectKPPCLayer.cpp
    src/DiscriminativePCNetwork.cpp
    src/ConvPCNetwork.cpp
    src/SimplePCNetwork.cpp
    src/GaussSeidelPCNetwork.cpp
    src/SimpleConvPCNetwork.cpp
    src/DirectKPPCNetwork.cpp
    src/ModelIO.cpp
    src/StreamAlignedBatcher.cpp
)

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
    INTERPROCEDURAL_OPTIMIZATION_RELEASE ${DEEPITY_IPO_SUPPORTED}
)

# --- pydeepity ------------------------------------------------------

if(DEEPITY_BUILD_PYTHON_BINDINGS AND NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    nanobind_add_module(pydeepity bindings/pybinding.cpp)

    target_link_libraries(pydeepity PRIVATE Deepity)

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
