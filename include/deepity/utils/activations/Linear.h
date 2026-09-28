#pragma once
#include <algorithm>
#include <cstddef>
#include <deepity/utils/activations/ActivationMacros.h>

/**
 * @file Linear.h
 * @brief The identity activation and its (constant) derivative.
 */

namespace Deep
{
    /// @brief Identity activation: f(x) = x. A no-op, kept as a real
    /// function so it can be used interchangeably with the other
    /// activations through ActivationFn.
    static inline void linear([[maybe_unused]] float *x, [[maybe_unused]] size_t n) noexcept {}

    /// @brief Derivative of linear(): a constant 1 everywhere.
    static inline void dLinear(float *x, size_t n, [[maybe_unused]] bool activated = false) noexcept
    {
        std::fill(x, x + n, 1.0f);
    }

    /// @brief Two-buffer variant of dLinear, src is unused (the
    /// derivative of a linear function is a constant), kept for
    /// signature consistency with To_dFn2's dispatch table.
    static inline void dLinearInto(float *RESTRICT dst, [[maybe_unused]] const float *RESTRICT src, size_t n) noexcept
    {
        std::fill(dst, dst + n, 1.0f);
    }
} // namespace Deep
