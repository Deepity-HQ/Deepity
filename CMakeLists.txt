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

message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "CXX implicit link dirs: ${CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES}")
message(STATUS "CXX implicit link libs: ${CMAKE_CXX_IMPLICIT_LINK_LIBRARIES}")

# --------------------------------------------------------------------
# Dependencies
# --------------------------------------------------------------------

# Clang on Windows often ships libomp.lib/libomp.dll but FindOpenMP can't
# auto-detect the flags/paths for it (a known CMake/Clang-on-Windows gap --
# NOT an issue on Linux/GCC or with MSVC, both of which find_package(OpenMP)
# already handles natively). Nothing here is hardcoded to a specific
# install location -- every path is derived from the compiler's own
# `-print-resource-dir` output, computed ONCE (LLVM_ROOT_DIR) and reused
# for both the import library (.lib) and the runtime DLL, so this works
# regardless of where LLVM happens to be installed on the machine.
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND WIN32)
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-resource-dir
        OUTPUT_VARIABLE CLANG_RESOURCE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "Clang resource dir: ${CLANG_RESOURCE_DIR}")

    # Resource dir is <LLVM_ROOT>/lib/clang/<version> -- three levels up
    # is <LLVM_ROOT> itself, from which both lib/ (import libraries) and
    # bin/ (runtime DLLs) are standard, predictable subdirectories.
    get_filename_component(LLVM_ROOT_DIR "${CLANG_RESOURCE_DIR}/../../.." ABSOLUTE)
    message(STATUS "Derived LLVM root: ${LLVM_ROOT_DIR}")

    set(OpenMP_CXX_FLAGS "-fopenmp")
    set(OpenMP_CXX_LIB_NAMES "libomp")
    set(OpenMP_CXX_INCLUDE_DIR "${CLANG_RESOURCE_DIR}/include")

    find_library(OpenMP_libomp_LIBRARY
        NAMES libomp
        PATHS "${LLVM_ROOT_DIR}/lib"
    )
    message(STATUS "Found libomp import library at: ${OpenMP_libomp_LIBRARY}")

    # The .lib alone doesn't tell CMake where the runtime .dll lives --
    # without this, $<TARGET_RUNTIME_DLLS:...> (the generator expression
    # that auto-copies dependency DLLs next to each .exe) has no way to
    # know libomp.dll exists at all, and silently omits it from every
    # build -- the exact gap that caused DeepityTests.exe to fail to
    # launch until libomp.dll was copied in by hand.
    find_file(OpenMP_libomp_DLL
        NAMES libomp.dll
        PATHS "${LLVM_ROOT_DIR}/bin"
    )
    message(STATUS "Found libomp runtime DLL at: ${OpenMP_libomp_DLL}")

    if (OpenMP_libomp_DLL)
        add_library(libomp_dll SHARED IMPORTED)
        set_target_properties(libomp_dll PROPERTIES
            IMPORTED_LOCATION "${OpenMP_libomp_DLL}"
            IMPORTED_IMPLIB "${OpenMP_libomp_LIBRARY}"
        )
    endif()
endif()

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

# --------------------------------------------------------------------
# OpenBLAS (self-built, NOT vcpkg)
# --------------------------------------------------------------------
#
# vcpkg's prebuilt openblas:x64-windows always selects the "generic"
# kernel at RUNTIME (confirmed via openblas_get_corename() -- a ~45x
# slowdown, isolated entirely to GEMM calls, on real hardware). This is
# deliberate on vcpkg's part: prebuilt binaries default to the safest,
# lowest-common-denominator kernel so they don't crash on someone else's
# older CPU.
#
# DYNAMIC_ARCH (bundle many kernels, pick one at runtime) would be the
# "portable to any machine" fix, but (a) vcpkg's own openblas[dynamic-arch]
# feature is explicitly unsupported on the standard x64-windows triplet,
# confirmed directly, and (b) it's solving a problem this project doesn't
# actually have here -- we're building directly ON the target machine, not
# shipping a binary to someone else's unknown hardware. Plain (non-dynamic)
# OpenBLAS auto-detects the CPU ONCE at build time and compiles a single
# kernel tuned for it -- much faster to configure/build (one kernel, not
# dozens), and correctly targets this specific machine since that's all
# that's needed.
#
# Building OpenBLAS directly via its own CMake build system sidesteps
# vcpkg's port-level restriction entirely -- same FetchContent pattern
# already used for SLEEF below.

set(BUILD_SHARED_LIBS ON CACHE BOOL "" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
set(NOFORTRAN ON CACHE BOOL "" FORCE)  # C-only build, no Fortran toolchain required on Windows

set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

FetchContent_Declare(
    openblas_src
    GIT_REPOSITORY https://github.com/OpenMathLib/OpenBLAS.git
    GIT_TAG v0.3.28
)

set(CMAKE_REQUIRED_QUIET ON)
FetchContent_MakeAvailable(openblas_src)
set(CMAKE_REQUIRED_QUIET OFF)

unset(CMAKE_POLICY_VERSION_MINIMUM)

# OpenBLAS's own CMake build exports a target named "openblas" -- this
# REPLACES every downstream use of ${BLAS_LIBRARIES}/${BLAS_INCLUDE_DIR}
# (both previously populated by find_package(BLAS)/find_path()) with this
# single real CMake target instead. cblas.h is generated into the BINARY
# dir at build time (not the source dir), hence openblas_BINARY_DIR here.
set(BLAS_LIBRARIES openblas)
set(BLAS_INCLUDE_DIR "${openblas_BINARY_DIR}/generated" "${openblas_SOURCE_DIR}")

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

# Links the imported libomp DLL target (if it was located above) into
# Deepity's own dependency graph, so $<TARGET_RUNTIME_DLLS:...> discovers
# it and copies it next to every executable/module that links Deepity --
# no more manual copying after each clean rebuild.
if (TARGET libomp_dll)
    target_link_libraries(Deepity PUBLIC libomp_dll)
endif()

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
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:deepity>>,copy_if_different,true>
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
    add_executable(DeepityTests tests/tSimpleConvVerify.cpp)

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
            COMMAND ${CMAKE_COMMAND} -E $<IF:$<BOOL:$<TARGET_RUNTIME_DLLS:DeepityTests>>,copy_if_different,true>
            $<TARGET_RUNTIME_DLLS:DeepityTests>
            $<TARGET_FILE_DIR:DeepityTests>
            COMMAND_EXPAND_LISTS
        )
    endif()
endif()

# --------------------------------------------------------------------
# Profiling (Windows slowdown investigation)
# --------------------------------------------------------------------

if(DEEPITY_BUILD_TESTS)
    # PCN_TIME() calls live INSIDE the .cpp files -- defining PCN_PROFILE
    # only on the test executable does nothing, since Deepity (the
    # library actually containing those calls) is a separately-compiled
    # static library. Needs its OWN profiled build, same sources.
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
    if (TARGET libomp_dll)
        target_link_libraries(DeepityProfiled PUBLIC libomp_dll)
    endif()

    target_compile_definitions(DeepityProfiled PUBLIC SLEEF_STATIC_LIBS PCN_PROFILE)

    if(WIN32)
        target_compile_definitions(DeepityProfiled PUBLIC NOMINMAX)
    endif()

    if(MSVC)
        target_compile_options(DeepityProfiled PRIVATE /O2)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(DeepityProfiled PRIVATE -O3 ${ARCH_FLAGS} -ffast-math -fvectorize -fslp-vectorize)
    else()
        target_compile_options(DeepityProfiled PRIVATE -O3 ${ARCH_FLAGS} -ffast-math -ftree-vectorize -ftree-slp-vectorize)
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