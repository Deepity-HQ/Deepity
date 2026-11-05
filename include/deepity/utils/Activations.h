#pragma once

/**
 * @file Activations.h
 * @brief Defines the activation functions of a predictive coding model.
 *
 * This is an umbrella header: the actual implementations live in
 * deepity/utils/activations/ (ActivationDispatch.h for the ActivationType
 * dispatch tables, then one header per activation family: Relu.h, Gelu.h,
 * Tanh.h, Sigmoid.h, Linear.h, HardTanh.h, VectorMath.h).
 *
 * @code{.cpp}
 * #include <deepity/utils/Activations.h>
 *
 * Deep::tanh(array, arraysize)
 * @endcode
 *
 * @note Separate implementations exist for AVX512F, AVX2, SSE, and naive.
 *       Memory does NOT need to be aligned.
 * @version 1.1
 * @date 2026-08-23
 * @author Jack Rose
 */

#include <deepity/utils/activations/ActivationDispatch.h>
#include <deepity/utils/activations/Gelu.h>
#include <deepity/utils/activations/HardTanh.h>
#include <deepity/utils/activations/Linear.h>
#include <deepity/utils/activations/Relu.h>
#include <deepity/utils/activations/Sigmoid.h>
#include <deepity/utils/activations/Tanh.h>
#include <deepity/utils/activations/VectorMath.h>
#include <sleef.h>

namespace Deep
{
/// @brief Single-element forward activation, for hot loops that need
/// f(z) inline, one element at a time, rather than through a
/// whole-buffer ActivationFn/ActivationInto call. Mirrors each
/// family's own vectorized forward formula exactly.
/// @param type A forward-flavored ActivationType (e.g. TANH, not
/// dTANH). Anything else (including NONE) falls back to the identity.
/// @param z The pre-activation value.
static inline float ActivationScalar(ActivationType type, float z) noexcept
{
  switch (type)
  {
  case ActivationType::RELU:
    return z > 0.0f ? z : 0.0f;
  case ActivationType::GELU:
  {
    float zsq = z * z;
    float inner = MAGIC_GELU_1 * z + (MAGIC_GELU_2 * MAGIC_GELU_1) * zsq * z;
    float t = Sleef_tanhf_u10(inner);
    return 0.5f * z * (1.0f + t);
  }
  case ActivationType::SIGMOID:
    return 1.0f / (1.0f + std::exp(-z));
  case ActivationType::eSIGMOID:
    return 0.5f + 0.5f * (z / (1.0f + std::fabs(z)));
  case ActivationType::TANH:
    return Sleef_tanhf_u10(z);
  case ActivationType::HARD_TANH:
    return std::clamp(z, -1.0f, 1.0f);
  case ActivationType::LINEAR:
  case ActivationType::NONE:
  default:
    return z;
  }
}

/// @brief Single-element activation derivative, for hot loops that
/// need f'(z) inline, one element at a time, rather than through a
/// whole-buffer DerivativeFn/DerivativeFn2 call. Mirrors the scalar
/// tail of each family's own vectorized derivative exactly.
/// @param dType A derivative-flavored ActivationType (e.g. dTANH, not
/// TANH). Anything else (including NONE) falls back to 1.0.
/// @param z The pre-activation value.
static inline float ActivationDerivativeScalar(ActivationType dType, float z) noexcept
{
  switch (dType)
  {
  case ActivationType::dRELU:
    return (z > 0.0f) ? 1.0f : 0.0f;
  case ActivationType::dGELU:
  {
    float zsq = z * z;
    float inner = MAGIC_GELU_1 * z + (MAGIC_GELU_2 * MAGIC_GELU_1) * zsq * z;
    float t = Sleef_tanhf_u10(inner);
    float gprime = MAGIC_GELU_1 + (3.0f * MAGIC_GELU_2 * MAGIC_GELU_1) * zsq;
    return 0.5f * (1.0f + t) + 0.5f * z * gprime * (1.0f - t * t);
  }
  case ActivationType::dSIGMOID:
  {
    float sig = 1.0f / (1.0f + std::exp(-z));
    return sig * (1.0f - sig);
  }
  case ActivationType::d_eSIGMOID:
  {
    float a = 1.0f + std::fabs(z);
    return 0.5f / (a * a);
  }
  case ActivationType::dTANH:
  {
    float t = Sleef_tanhf_u10(z);
    return 1.0f - t * t;
  }
  case ActivationType::dHARD_TANH:
    return (z > -1.0f && z < 1.0f) ? 1.0f : 0.0f;
  case ActivationType::dLINEAR:
  case ActivationType::NONE:
  default:
    return 1.0f;
  }
}

/// @brief Single-element activation derivative computed from an
/// already-activated value (e.g. sig*(1-sig) given sig, not given the
/// pre-activation z), the closed-form shortcut every family except
/// GELU supports. Mirrors each family's own dXxx(buf, n,
/// activated=true) in-place formula.
/// @warning GELU has no such closed form; its derivative can't be
/// recovered from gelu(z) alone. The dGELU case treats `activated` as
/// if it were the raw pre-activation z instead.
/// @param dType A derivative-flavored ActivationType. Anything else
/// (including NONE) falls back to 1.0.
/// @param activated The already-activated value (or, for dGELU only,
/// see the warning above, the raw pre-activation value).
static inline float ActivationDerivativeFromActivatedScalar(ActivationType dType,
                                                            float activated) noexcept
{
  switch (dType)
  {
  case ActivationType::dRELU:
    return (activated > 0.0f) ? 1.0f : 0.0f;
  case ActivationType::dGELU:
    return ActivationDerivativeScalar(dType, activated);
  case ActivationType::dSIGMOID:
    return activated * (1.0f - activated);
  case ActivationType::d_eSIGMOID:
    return 2.0f * activated * (1.0f - activated);
  case ActivationType::dTANH:
    return 1.0f - activated * activated;
  case ActivationType::dHARD_TANH:
    return (activated > -1.0f && activated < 1.0f) ? 1.0f : 0.0f;
  case ActivationType::dLINEAR:
  case ActivationType::NONE:
  default:
    return 1.0f;
  }
}

/// @brief Generic (non-SIMD) counterpart to ActivationScalar(), over
/// any floating-point T, using std::tanh/std::exp instead of Sleef's
/// float-only intrinsics, so this is correct for T=double. Used by
/// CPUBackendT<T, AccumT>'s non-float elementwise fallback path.
template <typename T>
static inline T ActivationScalarT(ActivationType type, T z) noexcept
{
  switch (type)
  {
  case ActivationType::RELU:
    return z > T(0) ? z : T(0);
  case ActivationType::GELU:
  {
    constexpr T c1 = T(0.7978845608028654), c2 = T(0.044715);
    T zsq = z * z;
    T inner = c1 * z + (c2 * c1) * zsq * z;
    T t = std::tanh(inner);
    return T(0.5) * z * (T(1) + t);
  }
  case ActivationType::SIGMOID:
    return T(1) / (T(1) + std::exp(-z));
  case ActivationType::eSIGMOID:
    return T(0.5) + T(0.5) * (z / (T(1) + std::fabs(z)));
  case ActivationType::TANH:
    return std::tanh(z);
  case ActivationType::HARD_TANH:
    return std::clamp(z, T(-1), T(1));
  case ActivationType::LINEAR:
  case ActivationType::NONE:
  default:
    return z;
  }
}

/// @brief Generic (non-SIMD) counterpart to ActivationDerivativeScalar().
template <typename T>
static inline T ActivationDerivativeScalarT(ActivationType dType, T z) noexcept
{
  switch (dType)
  {
  case ActivationType::dRELU:
    return (z > T(0)) ? T(1) : T(0);
  case ActivationType::dGELU:
  {
    constexpr T c1 = T(0.7978845608028654), c2 = T(0.044715);
    T zsq = z * z;
    T inner = c1 * z + (c2 * c1) * zsq * z;
    T t = std::tanh(inner);
    T gprime = c1 + (T(3) * c2 * c1) * zsq;
    return T(0.5) * (T(1) + t) + T(0.5) * z * gprime * (T(1) - t * t);
  }
  case ActivationType::dSIGMOID:
  {
    T sig = T(1) / (T(1) + std::exp(-z));
    return sig * (T(1) - sig);
  }
  case ActivationType::d_eSIGMOID:
  {
    T a = T(1) + std::fabs(z);
    return T(0.5) / (a * a);
  }
  case ActivationType::dTANH:
  {
    T t = std::tanh(z);
    return T(1) - t * t;
  }
  case ActivationType::dHARD_TANH:
    return (z > T(-1) && z < T(1)) ? T(1) : T(0);
  case ActivationType::dLINEAR:
  case ActivationType::NONE:
  default:
    return T(1);
  }
}
} // namespace Deep
