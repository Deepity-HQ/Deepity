#pragma once
#include <algorithm>
#include <cstddef>
#include <deepity/utils/activations/ActivationMacros.h>

/**
 * @file HardTanh.h
 * @brief hard_tanh and its derivative: clamp(x,-1,1), a piecewise-
 * linear approximation to tanh, bounded above and below unlike ReLU.
 *
 * Scalar only; no SIMD path.
 */

namespace Deep
{
/// @brief hard_tanh(x) = clamp(x, -1, 1).
static inline void hardTanh(float* x, size_t n) noexcept
{
  for (size_t i = 0; i < n; i++)
    x[i] = std::clamp(x[i], -1.0f, 1.0f);
}

/// @brief In-place derivative: 1 where the value is strictly inside
/// (-1, 1), else 0. Exact whether computed from the pre-activation z
/// or the already-activated value.
static inline void dHardTanh(float* x, size_t n, [[maybe_unused]] bool activated = false) noexcept
{
  for (size_t i = 0; i < n; i++)
    x[i] = (x[i] > -1.0f && x[i] < 1.0f) ? 1.0f : 0.0f;
}

/// @brief Two-buffer variant: reads src, writes the derivative into dst.
static inline void dHardTanhInto(float* RESTRICT dst, const float* RESTRICT src, size_t n) noexcept
{
  for (size_t i = 0; i < n; i++)
    dst[i] = (src[i] > -1.0f && src[i] < 1.0f) ? 1.0f : 0.0f;
}
} // namespace Deep
