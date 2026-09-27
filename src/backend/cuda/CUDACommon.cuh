#pragma once

#include <iostream>

/**
 * @file CUDACommon.cuh
 * @brief Shared helper(s) for the CUDABackend*.cu translation units. Only
 * ever included from a .cu file compiled under DEEPITY_USE_CUDA -- never
 * from host-only (.cpp) code.
 */

/// @brief Logs the most recent CUDA error (if any) to stderr, tagged with
/// the call site. Used right after every raw `<<<...>>>` kernel launch --
/// cuBLAS/CUTLASS calls check their own returned status instead.
#define CHECK_CUDA_LAUNCH()                                                                        \
  do                                                                                               \
  {                                                                                                \
    cudaError_t err = cudaGetLastError();                                                          \
    if (err != cudaSuccess)                                                                        \
    {                                                                                              \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " -> "                       \
                << cudaGetErrorString(err) << std::endl;                                           \
    }                                                                                              \
  } while (0)
