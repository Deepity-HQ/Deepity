#pragma once
#include <cstdint>

/**
 * @file ActivationType.h
 * @brief Just the ActivationType enum, deliberately split out of
 * Activations.h, zero dependency on <immintrin.h>/<sleef.h>/<omp.h>.
 *
 * Why this exists: IComputeBackend.h's methods take ActivationType as a
 * plain enum parameter and never touch the actual CPU SIMD function
 * implementations (ActivationFn/DerivativeFn/To_Fn/etc. all live in the
 * full Activations.h), so IComputeBackend.h, and everything that
 * includes it (crucially CUDABackend.h/.cu, compiled by nvcc), never
 * needed the SIMD-intrinsics half of Activations.h at all.
*/

namespace Deep
{
    /// @brief Identifies an activation function or its derivative, for
    /// dispatch tables and layer/backend APIs that need to name an
    /// activation without depending on the SIMD implementations
    /// themselves (see the file-level note above on why this is split
    /// out of Activations.h).
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
        NONE
    };
}