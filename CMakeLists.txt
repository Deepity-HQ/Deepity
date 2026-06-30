cmake_minimum_required(VERSION 4.3.0)
project(deepity CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_path(CBLAS_INCLUDE_DIR
    cblas.h
    PATH_SUFFIXES openblas
    REQUIRED
)

include(FetchContent)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
)

set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(benchmark)

add_library(Deepity STATIC
    src/PCLayer.cpp
    src/RBLayer.cpp
    src/PCNetwork.cpp
)

target_include_directories(Deepity PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/includes
    ${CBLAS_INCLUDE_DIR}
)

target_link_libraries(Deepity PUBLIC
    ${BLAS_LIBRARIES}
    OpenMP::OpenMP_CXX
)

if(MSVC)
    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:/O2 /arch:AVX2>
        $<$<CONFIG:RelWithDebInfo>:/O2 /Zi /arch:AVX2>
        $<$<CONFIG:Debug>:/Od /Zi>
    )
else()
    target_compile_options(Deepity PRIVATE
        $<$<CONFIG:Release>:-O3 -march=native>
        $<$<CONFIG:RelWithDebInfo>:-O3 -g -march=native>
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
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
    POSITION_INDEPENDENT_CODE on
)

find_package(pybind11 REQUIRED)

pybind11_add_module(deepity pybind/binding.cpp)
target_link_libraries(deepity PRIVATE Deepity)
set_target_properties(deepity PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)

add_executable(DeepityTests
    tests/tNetwork.cpp # Change based on test
)
target_link_libraries(DeepityTests PRIVATE Deepity benchmark::benchmark)
set_target_properties(DeepityTests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)