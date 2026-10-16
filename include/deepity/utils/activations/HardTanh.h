#pragma once
#include <algorithm>
#include <cstddef>
#include <deepity/utils/activations/ActivationMacros.h>

/**
 * @file HardTanh.h
 * @brief hard_tanh and its derivative: clamp(x,-1,1), the piecewise-
 * linear approximation to tanh PCX uses for its PC-CE/iPC variants.
 * Bounded above and below, unlike ReLU -- the property this was added
 * for (ReLU's unbounded growth was implicated in the energy blow-ups
 * seen when porting PCX's VGG-7 config with ReLU substituted in for
 * PCX's own hard_tanh).
 *
 * Scalar only, deliberately: correctness over speed for now, matching
 * Linear.h's own precedent rather than Relu.h's full SIMD treatment.
 * Add AVX paths later if this becomes a measured bottleneck.
 */

namespace Deep
{
    /// @brief hard_tanh(x) = clamp(x, -1, 1).
    static inline void hardTanh(float *x, size_t n) noexcept
    {
        for (size_t i = 0; i < n; i++)
            x[i] = std::clamp(x[i], -1.0f, 1.0f);
    }

    /// @brief In-place derivative: 1 where the value is strictly inside
    /// (-1, 1), else 0 (the clamp boundaries have zero gradient). Exact
    /// whether computed from the pre-activation z or the already-
    /// activated value -- hard_tanh's derivative only depends on
    /// whether clamping happened, which clamp(z,-1,1) itself already
    /// tells you from the activated value alone (unlike GELU, this has
    /// no from-activated ambiguity).
    static inline void dHardTanh(float *x, size_t n, [[maybe_unused]] bool activated = false) noexcept
    {
        for (size_t i = 0; i < n; i++)
            x[i] = (x[i] > -1.0f && x[i] < 1.0f) ? 1.0f : 0.0f;
    }

    /// @brief Two-buffer variant: reads src, writes the derivative into dst.
    static inline void dHardTanhInto(float *RESTRICT dst, const float *RESTRICT src, size_t n) noexcept
    {
        for (size_t i = 0; i < n; i++)
            dst[i] = (src[i] > -1.0f && src[i] < 1.0f) ? 1.0f : 0.0f;
    }
} // namespace Deep
