#pragma once
#include <cstdint>

/**
 * @file ActivationType.h
 * @brief Just the ActivationType enum, split out of Activations.h,
 * with zero dependency on <immintrin.h>/<sleef.h>/<omp.h>.
 */

namespace Deep
{
/// @brief Identifies an activation function or its derivative, for
/// dispatch tables and layer/backend APIs that need to name an
/// activation without depending on its SIMD implementation.
enum class ActivationType : uint8_t
{
  RELU,
  dRELU,
  GELU,
  dGELU,
  SIGMOID,
  dSIGMOID,
  eSIGMOID,
  d_eSIGMOID,
  TANH,
  dTANH,
  LINEAR,
  dLINEAR,
  HARD_TANH,
  dHARD_TANH,
  NONE
};
} // namespace Deep