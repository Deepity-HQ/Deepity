#pragma once
#include <cstddef>
#include <deepity/utils/ActivationType.h>
#include <deepity/utils/activations/ActivationMacros.h>

/**
 * @file ActivationDispatch.h
 * @brief Function-pointer types and ActivationType <-> function dispatch
 * tables shared by every activation family. Forward-declares the concrete
 * activation functions (defined in the sibling Relu.h/Gelu.h/Tanh.h/
 * Sigmoid.h/Linear.h) so the dispatchers below can take their addresses
 * without depending on inclusion order.
 */

namespace Deep
{
    /// @brief Pointer to an in-place activation function: f(buf, n).
    using ActivationFn = void (*)(float *, size_t);
    /// @brief Pointer to an in-place activation derivative: f(buf, n, activated).
    using DerivativeFn = void (*)(float *, size_t, bool);
    // Two-buffer variant: reads from src, writes the derivative directly
    // into dst, in one pass. Eliminates the copy-then-derive pattern
    // (cblas_scopy + DerivativeFn) callers currently need when they can't
    // afford to mutate src in place, src stays untouched throughout.
    /// @brief Pointer to a two-buffer activation derivative: reads src,
    /// writes the derivative into dst, leaving src untouched.
    using DerivativeFn2 = void (*)(float *RESTRICT, const float *RESTRICT, size_t);

    // Forward declarations only, the real definitions (and their full
    // docs) live in the sibling Relu.h/Gelu.h/Tanh.h/Sigmoid.h/Linear.h,
    // one activation family per header. @copydoc isn't usable here: with
    // both this declaration and the real one in the same Deep namespace,
    // Doxygen resolves "Deep::relu" back to whichever declaration it's
    // attached to, so it self-references instead of finding the other one.

    /// @brief See Relu.h.
    static inline void relu(float *, size_t) noexcept;
    /// @brief See Gelu.h.
    static inline void gelu(float *, size_t) noexcept;
    /// @brief See Sigmoid.h.
    static inline void sigmoid(float *, size_t) noexcept;
    /// @brief See Sigmoid.h.
    static inline void e_sigmoid(float *, size_t) noexcept;
    /// @brief See Tanh.h.
    static inline void tanh(float *, size_t) noexcept;
    /// @brief See Linear.h.
    static inline void linear(float *, size_t) noexcept;
    /// @brief See HardTanh.h.
    static inline void hardTanh(float *, size_t) noexcept;

    /// @brief See Relu.h.
    static inline void dRelu(float *, size_t, bool) noexcept;
    /// @brief See Gelu.h.
    static inline void dGelu(float *, size_t, bool) noexcept;
    /// @brief See Sigmoid.h.
    static inline void dSigmoid(float *, size_t, bool) noexcept;
    /// @brief See Sigmoid.h.
    static inline void d_eSigmoid(float *, size_t, bool) noexcept;
    /// @brief See Tanh.h.
    static inline void dTanh(float *, size_t, bool) noexcept;
    /// @brief See Linear.h.
    static inline void dLinear(float *, size_t, bool) noexcept;
    /// @brief See HardTanh.h.
    static inline void dHardTanh(float *, size_t, bool) noexcept;

    /// @brief See Relu.h.
    static inline void dReluInto(float *RESTRICT, const float *RESTRICT, size_t) noexcept;
    /// @brief See Gelu.h.
    static inline void dGeluInto(float *RESTRICT, const float *RESTRICT, size_t) noexcept;
    /// @brief See Sigmoid.h.
    static inline void dSigmoidInto(float *RESTRICT, const float *RESTRICT, size_t) noexcept;
    /// @brief See Sigmoid.h.
    static inline void d_eSigmoidInto(float *RESTRICT, const float *RESTRICT, size_t) noexcept;
    /// @brief See Tanh.h.
    static inline void dTanhInto(float *RESTRICT, const float *RESTRICT, size_t) noexcept;
    /// @brief See Linear.h.
    static inline void dLinearInto(float *RESTRICT, const float *RESTRICT, size_t) noexcept;
    /// @brief See HardTanh.h.
    static inline void dHardTanhInto(float *RESTRICT, const float *RESTRICT, size_t) noexcept;

    /// @brief Maps an ActivationType to its concrete function pointer.
    /// @return nullptr for ActivationType::NONE or an unrecognized value.
    static inline ActivationFn To_Fn(ActivationType type)
    {
        switch (type)
        {
        case ActivationType::RELU:
            return relu;
        case ActivationType::GELU:
            return gelu;
        case ActivationType::SIGMOID:
            return sigmoid;
        case ActivationType::eSIGMOID:
            return e_sigmoid;
        case ActivationType::TANH:
            return tanh;
        case ActivationType::LINEAR:
            return linear;
        case ActivationType::HARD_TANH:
            return hardTanh;
        case ActivationType::NONE:
        default:
            return nullptr;
        }
    }

    /// @brief Maps an ActivationType (a derivative variant, e.g. dRELU)
    /// to its concrete function pointer.
    /// @return nullptr for ActivationType::NONE or an unrecognized value.
    static inline DerivativeFn To_dFn(ActivationType dType)
    {
        switch (dType)
        {
        case ActivationType::dRELU:
            return dRelu;
        case ActivationType::dGELU:
            return dGelu;
        case ActivationType::dSIGMOID:
            return dSigmoid;
        case ActivationType::d_eSIGMOID:
            return d_eSigmoid;
        case ActivationType::dTANH:
            return dTanh;
        case ActivationType::dLINEAR:
            return dLinear;
        case ActivationType::dHARD_TANH:
            return dHardTanh;
        case ActivationType::NONE:
        default:
            return nullptr;
        }
    }

    /// @brief Dispatches to the two-buffer (dst, src, n) derivative
    /// variant for the given activation type, see DerivativeFn2.
    static inline DerivativeFn2 To_dFn2(ActivationType dType)
    {
        switch (dType)
        {
        case ActivationType::dRELU:
            return dReluInto;
        case ActivationType::dGELU:
            return dGeluInto;
        case ActivationType::dSIGMOID:
            return dSigmoidInto;
        case ActivationType::d_eSIGMOID:
            return d_eSigmoidInto;
        case ActivationType::dTANH:
            return dTanhInto;
        case ActivationType::dLINEAR:
            return dLinearInto;
        case ActivationType::dHARD_TANH:
            return dHardTanhInto;
        case ActivationType::NONE:
        default:
            return nullptr;
        }
    }

    /// @brief Reverse lookup of To_Fn(): maps a function pointer back to
    /// its ActivationType, by identity comparison against every known
    /// activation function.
    /// @return ActivationType::NONE if fn doesn't match any known activation.
    static inline ActivationType To_AType(ActivationFn fn)
    {
        if (fn == relu)
            return ActivationType::RELU;
        if (fn == gelu)
            return ActivationType::GELU;
        if (fn == sigmoid)
            return ActivationType::SIGMOID;
        if (fn == e_sigmoid)
            return ActivationType::eSIGMOID;
        if (fn == tanh)
            return ActivationType::TANH;
        if (fn == linear)
            return ActivationType::LINEAR;
        if (fn == hardTanh)
            return ActivationType::HARD_TANH;
        return ActivationType::NONE;
    }

    /// @brief Same as the ActivationFn overload above, for derivative
    /// function pointers.
    /// @return ActivationType::NONE if dfn doesn't match any known derivative.
    static inline ActivationType To_AType(DerivativeFn dfn)
    {
        if (dfn == dRelu)
            return ActivationType::dRELU;
        if (dfn == dGelu)
            return ActivationType::dGELU;
        if (dfn == dSigmoid)
            return ActivationType::dSIGMOID;
        if (dfn == d_eSigmoid)
            return ActivationType::d_eSIGMOID;
        if (dfn == dTanh)
            return ActivationType::dTANH;
        if (dfn == dLinear)
            return ActivationType::dLINEAR;
        if (dfn == dHardTanh)
            return ActivationType::dHARD_TANH;
        return ActivationType::NONE;
    }

    /// @brief Maps a forward ActivationType to its derivative-flavored
    /// counterpart (e.g. TANH -> dTANH). The single canonical version of
    /// this mapping; every layer should use this rather than maintaining
    /// its own copy.
    /// @return ActivationType::NONE for LINEAR, NONE, or an unrecognized
    /// value (LINEAR's derivative is the constant-1 dLINEAR handler,
    /// which every ActivationDerivativeScalar/Device dispatch already
    /// maps NONE to as well, so returning NONE here is not a gap).
    static inline ActivationType ToDerivativeType(ActivationType fwd)
    {
        switch (fwd)
        {
        case ActivationType::RELU:
            return ActivationType::dRELU;
        case ActivationType::GELU:
            return ActivationType::dGELU;
        case ActivationType::SIGMOID:
            return ActivationType::dSIGMOID;
        case ActivationType::eSIGMOID:
            return ActivationType::d_eSIGMOID;
        case ActivationType::TANH:
            return ActivationType::dTANH;
        case ActivationType::LINEAR:
            return ActivationType::dLINEAR;
        case ActivationType::HARD_TANH:
            return ActivationType::dHARD_TANH;
        case ActivationType::NONE:
        default:
            return ActivationType::NONE;
        }
    }
} // namespace Deep
