#pragma once
#include <cstddef>
#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <sleef.h>
#include <omp.h>

#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

/**
 * @file Activations.h
 * @brief Defines the activation functions of a predictive coding model.
 *
 * This header includes implementations of ReLU and TanH.
 *
 * Usage:
 *  #include <Activations.h>
 *
 * Example:
 *  Deep::tanh(array, arraysize)
 *
 * @note Separate implementations exist for AVX512F, AVX2, SSE, and naive.
 * @version 1.0
 * @date 2026-06-21
 * @author Jack Rose
 */

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

namespace Deep
{
    using ActivationFn = void (*)(float *, size_t);
    using DerivativeFn = void (*)(float *, size_t, bool);

    enum class ActivationType : uint8_t
    {
        RELU,
        dRELU,
        SIGMOID,
        dSIGMOID,
        TANH,
        dTANH,
        LINEAR,
        dLINEAR,
        NONE
    };

    static inline void relu(float *, size_t) noexcept;
    static inline void sigmoid(float *, size_t) noexcept;
    static inline void tanh(float *, size_t) noexcept;
    static inline void linear(float *, size_t) noexcept;

    static inline void dRelu(float *, size_t, bool) noexcept;
    static inline void dSigmoid(float *, size_t, bool) noexcept;
    static inline void dTanh(float *, size_t, bool) noexcept;
    static inline void dLinear(float *, size_t, bool) noexcept;

    static inline ActivationFn To_Fn(ActivationType type)
    {
        switch (type)
        {
        case ActivationType::RELU:
            return relu;
        case ActivationType::SIGMOID:
            return sigmoid;
        case ActivationType::TANH:
            return tanh;
        case ActivationType::LINEAR:
            return linear;
        case ActivationType::NONE:
        default:
            return nullptr;
        }
    }

    static inline DerivativeFn To_dFn(ActivationType dType)
    {
        switch (dType)
        {
        case ActivationType::dRELU:
            return dRelu;
        case ActivationType::dSIGMOID:
            return dSigmoid;
        case ActivationType::dTANH:
            return dTanh;
        case ActivationType::dLINEAR:
            return dLinear;
        case ActivationType::NONE:
        default:
            return nullptr;
        }
    }

    static inline ActivationType To_AType(ActivationFn fn)
    {
        if (fn == relu)
            return ActivationType::RELU;
        if (fn == sigmoid)
            return ActivationType::SIGMOID;
        if (fn == tanh)
            return ActivationType::TANH;
        if (fn == linear)
            return ActivationType::LINEAR;
        return ActivationType::NONE;
    }

    static inline ActivationType To_AType(DerivativeFn dfn)
    {
        if (dfn == dRelu)
            return ActivationType::dRELU;
        if (dfn == dSigmoid)
            return ActivationType::dSIGMOID;
        if (dfn == dTanh)
            return ActivationType::dTANH;
        if (dfn == dLinear)
            return ActivationType::dLINEAR;
        return ActivationType::NONE;
    }

    inline void expf_v(float *x, size_t n)
    {
        size_t simd_end = 0;

#if defined(__AVX512F__)
        simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);

            __m512 res = Sleef_expf16_u10avx512f(x_512);

            _mm512_store_ps(x + i, res);
        }

#elif defined(__AVX2__)
        simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);

            __m256 res = Sleef_expf8_u10avx2(x_256);

            _mm256_store_ps(x + i, res);
        }

#elif defined(__SSE2__)
        simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);

            __m128 res = Sleef_expf4_u10sse2(x_128);

            _mm_store_ps(x + i, res);
        }

#endif

        for (size_t i = simd_end; i < n; ++i)
        {
            x[i] = Sleef_expf_u10(x[i]);
        }
    }

    inline void logf_v(float *x, size_t n)
    {
        size_t simd_end = 0;

#if defined(__AVX512F__)
        simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);

            __m512 res = Sleef_logf16_u10avx512f(x_512);

            _mm512_store_ps(x + i, res);
        }

#elif defined(__AVX2__)
        simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);

            __m256 res = Sleef_logf8_u10avx2(x_256);

            _mm256_store_ps(x + i, res);
        }

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);
            __m128 res = Sleef_logf4_u10(x_128);

            _mm_store_ps(x + i, res);
        }
#endif
        for (size_t i = simd_end; i < n; ++i)
        {
            x[i] = Sleef_logf_u10(x[i]);
        }
    }

#pragma region relu
    /// @brief RELU(x) = MAX(0, x) for all x
    /// @param x array, \em assumed to be properly aligned
    /// @param n x length
    inline void relu(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t simd_end = 0;

#if defined(__AVX512F__)
        __m512 zeros = _mm512_setzero_ps();
        size_t simd_end4 = n - (n % 64);
        
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end4; i += 64)
        {
            __m512 x0 = _mm512_load_ps(x + i);
            __m512 x1 = _mm512_load_ps(x + i + 16);
            __m512 x2 = _mm512_load_ps(x + i + 32);
            __m512 x3 = _mm512_load_ps(x + i + 48);

            x0 = _mm512_max_ps(zeros, x0);
            x1 = _mm512_max_ps(zeros, x1);
            x2 = _mm512_max_ps(zeros, x2);
            x3 = _mm512_max_ps(zeros, x3);

            _mm512_store_ps(x + i, x0);
            _mm512_store_ps(x + i + 16, x1);
            _mm512_store_ps(x + i + 32, x2);
            _mm512_store_ps(x + i + 48, x3);
        }
        
        simd_end = n - (n % 16);
        for (size_t i = simd_end4; i < simd_end; i += 16)
        {
            _mm512_store_ps(x + i, _mm512_max_ps(zeros, _mm512_load_ps(x + i)));
        }

#elif defined(__AVX2__) || defined(__AVX__)
        __m256 zeros = _mm256_setzero_ps();
        size_t simd_end4 = n - (n % 32);

#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end4; i += 32)
        {
            __m256 x0 = _mm256_load_ps(x + i);
            __m256 x1 = _mm256_load_ps(x + i + 8);
            __m256 x2 = _mm256_load_ps(x + i + 16);
            __m256 x3 = _mm256_load_ps(x + i + 24);

            x0 = _mm256_max_ps(zeros, x0);
            x1 = _mm256_max_ps(zeros, x1);
            x2 = _mm256_max_ps(zeros, x2);
            x3 = _mm256_max_ps(zeros, x3);

            _mm256_store_ps(x + i, x0);
            _mm256_store_ps(x + i + 8, x1);
            _mm256_store_ps(x + i + 16, x2);
            _mm256_store_ps(x + i + 24, x3);
        }
        
        simd_end = n - (n % 8);
        for (size_t i = simd_end4; i < simd_end; i += 8)
        {
            _mm256_store_ps(x + i, _mm256_max_ps(zeros, _mm256_load_ps(x + i)));
        }

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 zeros = _mm_setzero_ps();
        size_t simd_end4 = n - (n % 16);

#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end4; i += 16)
        {
            __m128 x0 = _mm_load_ps(x + i);
            __m128 x1 = _mm_load_ps(x + i + 4);
            __m128 x2 = _mm_load_ps(x + i + 8);
            __m128 x3 = _mm_load_ps(x + i + 12);

            x0 = _mm_max_ps(zeros, x0);
            x1 = _mm_max_ps(zeros, x1);
            x2 = _mm_max_ps(zeros, x2);
            x3 = _mm_max_ps(zeros, x3);

            _mm_store_ps(x + i, x0);
            _mm_store_ps(x + i + 4, x1);
            _mm_store_ps(x + i + 8, x2);
            _mm_store_ps(x + i + 12, x3);
        }
        
        simd_end = n - (n % 4);
        for (size_t i = simd_end4; i < simd_end; i += 4)
        {
            _mm_store_ps(x + i, _mm_max_ps(zeros, _mm_load_ps(x + i)));
        }
#endif

        for (size_t i = simd_end; i < n; i++)
        {
            x[i] = MAX(0.0f, x[i]);
        }
    }

    inline void dRelu(float *RESTRICT x, const size_t n, [[maybe_unused]] bool activated = false) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t simd_end = 0;

#if defined(__AVX512F__)
        __m512 ones = _mm512_set1_ps(1.0f);
        __m512 zeros = _mm512_setzero_ps();
        size_t simd_end4 = n - (n % 64);
        
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end4; i += 64)
        {
            __m512 x0 = _mm512_load_ps(x + i);
            __m512 x1 = _mm512_load_ps(x + i + 16);
            __m512 x2 = _mm512_load_ps(x + i + 32);
            __m512 x3 = _mm512_load_ps(x + i + 48);

            __mmask16 m0 = _mm512_cmp_ps_mask(x0, zeros, _CMP_GT_OQ);
            __mmask16 m1 = _mm512_cmp_ps_mask(x1, zeros, _CMP_GT_OQ);
            __mmask16 m2 = _mm512_cmp_ps_mask(x2, zeros, _CMP_GT_OQ);
            __mmask16 m3 = _mm512_cmp_ps_mask(x3, zeros, _CMP_GT_OQ);

            _mm512_store_ps(x + i,      _mm512_mask_blend_ps(m0, zeros, ones));
            _mm512_store_ps(x + i + 16, _mm512_mask_blend_ps(m1, zeros, ones));
            _mm512_store_ps(x + i + 32, _mm512_mask_blend_ps(m2, zeros, ones));
            _mm512_store_ps(x + i + 48, _mm512_mask_blend_ps(m3, zeros, ones));
        }

        simd_end = n - (n % 16);
        for (size_t i = simd_end4; i < simd_end; i += 16)
        {
            __m512 x0 = _mm512_load_ps(x + i);
            __mmask16 m0 = _mm512_cmp_ps_mask(x0, zeros, _CMP_GT_OQ);
            _mm512_store_ps(x + i, _mm512_mask_blend_ps(m0, zeros, ones));
        }

#elif defined(__AVX2__) || defined(__AVX__)
        __m256 ones = _mm256_set1_ps(1.0f);
        __m256 zeros = _mm256_setzero_ps();
        size_t simd_end4 = n - (n % 32);

#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end4; i += 32)
        {
            __m256 x0 = _mm256_load_ps(x + i);
            __m256 x1 = _mm256_load_ps(x + i + 8);
            __m256 x2 = _mm256_load_ps(x + i + 16);
            __m256 x3 = _mm256_load_ps(x + i + 24);

            x0 = _mm256_and_ps(ones, _mm256_cmp_ps(x0, zeros, _CMP_GT_OQ));
            x1 = _mm256_and_ps(ones, _mm256_cmp_ps(x1, zeros, _CMP_GT_OQ));
            x2 = _mm256_and_ps(ones, _mm256_cmp_ps(x2, zeros, _CMP_GT_OQ));
            x3 = _mm256_and_ps(ones, _mm256_cmp_ps(x3, zeros, _CMP_GT_OQ));

            _mm256_store_ps(x + i, x0);
            _mm256_store_ps(x + i + 8, x1);
            _mm256_store_ps(x + i + 16, x2);
            _mm256_store_ps(x + i + 24, x3);
        }

        simd_end = n - (n % 8);
        for (size_t i = simd_end4; i < simd_end; i += 8)
        {
            __m256 x0 = _mm256_load_ps(x + i);
            _mm256_store_ps(x + i, _mm256_and_ps(ones, _mm256_cmp_ps(x0, zeros, _CMP_GT_OQ)));
        }

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_setzero_ps();
        size_t simd_end4 = n - (n % 16);

#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end4; i += 16)
        {
            __m128 x0 = _mm_load_ps(x + i);
            __m128 x1 = _mm_load_ps(x + i + 4);
            __m128 x2 = _mm_load_ps(x + i + 8);
            __m128 x3 = _mm_load_ps(x + i + 12);

            x0 = _mm_and_ps(ones, _mm_cmpgt_ps(x0, zeros));
            x1 = _mm_and_ps(ones, _mm_cmpgt_ps(x1, zeros));
            x2 = _mm_and_ps(ones, _mm_cmpgt_ps(x2, zeros));
            x3 = _mm_and_ps(ones, _mm_cmpgt_ps(x3, zeros));

            _mm_store_ps(x + i, x0);
            _mm_store_ps(x + i + 4, x1);
            _mm_store_ps(x + i + 8, x2);
            _mm_store_ps(x + i + 12, x3);
        }

        simd_end = n - (n % 4);
        for (size_t i = simd_end4; i < simd_end; i += 4)
        {
            __m128 x0 = _mm_load_ps(x + i);
            _mm_store_ps(x + i, _mm_and_ps(ones, _mm_cmpgt_ps(x0, zeros)));
        }
#endif

        for (size_t i = simd_end; i < n; i++)
        {
            x[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
        }
    }
#pragma endregion

#define TANH_TINYLIMIT 0.000244140625f
#define TANH_POLY_LIMIT 0.625f
#define TANH_BIGLIMIT 6.0f

#define P0 -0.9643991794f
#define P1 -99.28772310f
#define P2 -1614.687684f

#define Q0 112.8116785f
#define Q1 2235.488391f
#define Q2 4844.063053f

#pragma region tanh
    inline void tanh(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t simd_end = 0;

#if defined(__AVX512F__)
        simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);
            // u10 guarantees 1.0 ULP accuracy (highly precise)
            __m512 res = Sleef_tanhf16_u10avx512f(x_512);
            _mm512_store_ps(x + i, res);
        }

#elif defined(__AVX2__)
        simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);
            __m256 res = Sleef_tanhf8_u10avx2(x_256);
            _mm256_store_ps(x + i, res);
        }

#elif defined(__SSE4_1__) || defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
        simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536)
        for (size_t i = 0; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);
            // Fallback to sse4 or sse2 depending on what your SLEEF build exposes
#if defined(__SSE4_1__)
            __m128 res = Sleef_tanhf4_u10sse4(x_128);
#else
            __m128 res = Sleef_tanhf4_u10sse2(x_128);
#endif
            _mm_store_ps(x + i, res);
        }
#endif

        // Scalar remainder
        for (size_t i = simd_end; i < n; i++)
        {
            x[i] = Sleef_tanhf_u10(x[i]);
        }
    }

    inline void dTanh(float *RESTRICT x, const size_t n, bool activated = false) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        if (!activated)
            tanh(x, n);

        size_t i = 0;
        [[maybe_unused]] size_t simd_end;

#if defined(__AVX512F__)
        __m512 ones = _mm512_set1_ps(1.0f);
        simd_end = n - (n % 16);

        for (; i < simd_end; i += 16)
        {
            __m512 t = _mm512_load_ps(x + i);

#ifdef __FMA__
            __m512 res = _mm512_fnmadd_ps(t, t, ones); // 1 - t*t
#else
            __m512 res = _mm512_sub_ps(ones, _mm512_mul_ps(t, t));
#endif
            _mm512_store_ps(x + i, res);
        }

#elif defined(__AVX2__)
        __m256 ones = _mm256_set1_ps(1.0f);
        simd_end = n - (n % 8);

        for (; i < simd_end; i += 8)
        {
            __m256 t = _mm256_load_ps(x + i);
#ifdef __FMA__
            __m256 res = _mm256_fnmadd_ps(t, t, ones); // 1 - t*t
#else
            __m256 res = _mm256_sub_ps(ones, _mm256_mul_ps(t, t));
#endif
            _mm256_store_ps(x + i, res);
        }
#elif defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 ones = _mm_set1_ps(1.0f);
        simd_end = n - (n % 4);

        for (; i < simd_end; i += 4)
        {
            __m128 t = _mm_load_ps(x + i);

#ifdef __FMA__
            __m128 res = _mm_fnmadd_ps(t, t, ones); // 1 - t*t
#else
            __m128 res = _mm_sub_ps(ones, _mm_mul_ps(t, t));
#endif
            _mm_store_ps(x + i, res);
        }
#endif

        for (; i < n; ++i)
            x[i] = 1.0f - x[i] * x[i];
    }

#pragma endregion

#pragma region sigmoid

    /// @brief Implements the \em Elliot \em Sigmoid approximation, i.e. `S(x) = (1/2)((x / (1 + |x|)) + 1)`
    /// @param x array, \em assumed to be 64-bit aligned!
    /// @param n x length
    inline void sigmoid(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 half = _mm512_set1_ps(0.5f);
        __m512 one = _mm512_set1_ps(1.0f);
        size_t r = n % 16;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);
            __m512 den = _mm512_add_ps(
                _mm512_abs_ps(x_512),
                one);
            __m512 div = _mm512_div_ps(x_512, den);
            __m512 sig = _mm512_fmadd_ps(div, half, half);

            _mm512_store_ps(x + i, sig);
        }
#elif defined(__AVX2__)
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // for 256-bit abs
        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);
            __m256 den = _mm256_add_ps(
                _mm256_and_ps(x_256, mask),
                one);
            __m256 div = _mm256_div_ps(x_256, den);
#ifdef __FMA__
            __m256 sig = _mm256_fmadd_ps(div, half, half);
#else
            __m256 sig = _mm256_add_ps(_mm256_mul_ps(div, half), half);
#endif

            _mm256_store_ps(x + i, sig);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 half = _mm_set1_ps(0.5f);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)); // for 256-bit abs
        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);
            __m128 den = _mm_add_ps(
                _mm_and_ps(x_128, mask),
                one);
            __m128 div = _mm_div_ps(x_128, den);
#ifdef __FMA__
            __m128 sig = _mm_fmadd_ps(div, half, half);
#else
            __m128 sig = _mm_add_ps(_mm_mul_ps(div, half), half);
#endif

            _mm_store_ps(x + i, sig);
        }
#endif
        for (; i < n; i++)
        {
            x[i] = 0.5f * (x[i] / (1.0f + fabsf(x[i])) + 1.0f);
        }
    }

    inline void dSigmoid(float *RESTRICT x, const size_t n, bool activated = false) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        if (!activated)
            sigmoid(x, n);
        size_t i = 0;

#if defined(__AVX512F__)

        size_t r = n % 16;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);
            __m512 d = _mm512_fnmadd_ps(x_512, x_512, x_512); // d = x * (1 - x) = x - x^2 = -x*x + x
            _mm512_store_ps(x + i, d);
        }
#elif defined(__AVX2__)

        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);
#ifdef __FMA__
            __m256 d = _mm256_fnmadd_ps(x_256, x_256, x_256); // d = x * (1 - x) = x - x^2 = -x*x + x
#else
            __m256 d = _mm256_sub_ps(x_256, _mm256_mul_ps(x_256, x_256));
#endif
            _mm256_store_ps(x + i, d);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);

#ifdef __FMA__
            __m128 d = _mm_fnmadd_ps(x_128, x_128, x_128); // d = x * (1 - x) = x - x^2 = -x*x + x
#else
            __m128 d = _mm_sub_ps(x_128, _mm_mul_ps(x_128, x_128));
#endif
            _mm_store_ps(x + i, d);
        }
#endif

        for (; i < n; i++)
        {
            x[i] = x[i] * (1.0f - x[i]);
        }
    }
#pragma endregion

    inline void linear(float *x, size_t n) noexcept {}

    inline void dLinear(float *x, size_t n, [[maybe_unused]] bool activated = false) noexcept
    {
        std::fill_n(x, n, 1.0f);
    }
}