#pragma once
#include <cassert>
#include <cstddef>
#include <immintrin.h>
#include <sleef.h>
#include <deepity/utils/activations/ActivationMacros.h>

/**
 * @file Gelu.h
 * @brief GELU (tanh approximation) and its derivative (plain and
 * two-buffer variants), with AVX512/AVX2/SSE paths and a scalar tail.
 */

namespace Deep
{

    /// @brief sqrt(2/pi), the leading constant in the tanh approximation
    /// of GELU. Also used directly by CUDABackend's device kernels.
    constexpr float MAGIC_GELU_1 = 0.7978845608028654f;
    /// @brief The tanh approximation's cubic-term coefficient (0.044715),
    /// paired with MAGIC_GELU_1.
    constexpr float MAGIC_GELU_2 = 0.044715f;

    /// @brief In-place GELU activation (tanh approximation, ~5e-4 max
    /// absolute error vs. true GELU, see tests/tGelu.cpp).
    static inline void gelu(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        ptrdiff_t simd_end = 0;

#if defined(__AVX512F__)
        const __m512 ones512 = _mm512_set1_ps(1.0f);
        const __m512 half512 = _mm512_set1_ps(0.5f);
        const __m512 sqrt2overpi512 = _mm512_set1_ps(MAGIC_GELU_1);               // C1 = sqrt(2/pi)
        const __m512 gelu_coeff512 = _mm512_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1); // C2 = C1 * 0.044715

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)15;
        for (ptrdiff_t i = 0; i < simd_end; i += 16)
        {
            __m512 x512 = _mm512_loadu_ps(x + i);
            __m512 cube512 = _mm512_mul_ps(x512, _mm512_mul_ps(x512, x512));
            __m512 c1x512 = _mm512_mul_ps(sqrt2overpi512, x512);               // C1*x
            __m512 inner512 = _mm512_fmadd_ps(cube512, gelu_coeff512, c1x512); // x^3*C2 + C1*x
            __m512 t512 = Sleef_tanhf16_u10(inner512);
            __m512 res = _mm512_mul_ps(half512, _mm512_mul_ps(x512, _mm512_add_ps(t512, ones512)));
            _mm512_storeu_ps(x + i, res);
        }
#elif defined(__AVX__)
        const __m256 ones256 = _mm256_set1_ps(1.0f);
        const __m256 half256 = _mm256_set1_ps(0.5f);
        const __m256 sqrt2overpi256 = _mm256_set1_ps(MAGIC_GELU_1);               // C1 = sqrt(2/pi)
        const __m256 gelu_coeff256 = _mm256_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1); // C2 = C1 * 0.044715

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)7;
        for (ptrdiff_t i = 0; i < simd_end; i += 8)
        {
            __m256 x256 = _mm256_loadu_ps(x + i);
            __m256 cube256 = _mm256_mul_ps(x256, _mm256_mul_ps(x256, x256));
            __m256 c1x256 = _mm256_mul_ps(sqrt2overpi256, x256); // C1*x
#if defined(__AVX2__)
            __m256 inner256 = _mm256_fmadd_ps(cube256, gelu_coeff256, c1x256); // x^3*C2 + C1*x
#else
            __m256 inner256 = _mm256_add_ps(_mm256_mul_ps(cube256, gelu_coeff256), c1x256);
#endif
            __m256 t256 = Sleef_tanhf8_u10(inner256);
            __m256 res = _mm256_mul_ps(half256, _mm256_mul_ps(x256, _mm256_add_ps(t256, ones256)));
            _mm256_storeu_ps(x + i, res);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        const __m128 ones128 = _mm_set1_ps(1.0f);
        const __m128 half128 = _mm_set1_ps(0.5f);
        const __m128 sqrt2overpi128 = _mm_set1_ps(MAGIC_GELU_1);               // C1 = sqrt(2/pi)
        const __m128 gelu_coeff128 = _mm_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1); // C2 = C1 * 0.044715

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)3;
        for (ptrdiff_t i = 0; i < simd_end; i += 4)
        {
            __m128 x128 = _mm_loadu_ps(x + i);
            __m128 cube128 = _mm_mul_ps(x128, _mm_mul_ps(x128, x128));
            __m128 c1x128 = _mm_mul_ps(sqrt2overpi128, x128); // C1*x
#ifdef __FMA__
            __m128 inner128 = _mm_fmadd_ps(cube128, gelu_coeff128, c1x128); // x^3*C2 + C1*x
#else
            __m128 inner128 = _mm_add_ps(c1x128, _mm_mul_ps(cube128, gelu_coeff128));
#endif
            __m128 t128 = Sleef_tanhf4_u10(inner128);
            __m128 res = _mm_mul_ps(half128, _mm_mul_ps(x128, _mm_add_ps(t128, ones128)));
            _mm_storeu_ps(x + i, res);
        }
#endif
        for (ptrdiff_t i = simd_end; i < n; ++i)
        {
            float xi = x[i];
            float inner = MAGIC_GELU_1 * xi + (MAGIC_GELU_2 * MAGIC_GELU_1) * xi * xi * xi;
            x[i] = 0.5f * xi * (1.0f + Sleef_tanhf_u10(inner));
        }
    }

    /// @brief In-place GELU derivative.
    /// @param x Array to derive in place. Unlike dRelu/dTanh/dSigmoid,
    /// this always treats x as the PRE-activation value, the formula
    /// is recomputed from scratch regardless of `activated`, so calling
    /// this after gelu() has already run in place on the same buffer
    /// gives a wrong result.
    /// @param n Length of x.
    /// @param activated Unused; present only for signature consistency
    /// with DerivativeFn.
    static inline void dGelu(float *RESTRICT x, const size_t n, const bool activated) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        ptrdiff_t simd_end = 0;

#if defined(__AVX512F__)
        const __m512 ones512 = _mm512_set1_ps(1.0f);
        const __m512 half512 = _mm512_set1_ps(0.5f);
        const __m512 c1_512 = _mm512_set1_ps(MAGIC_GELU_1);                          // C1
        const __m512 c2_512 = _mm512_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1);           // C2
        const __m512 c2_x3_512 = _mm512_set1_ps(3.0f * MAGIC_GELU_2 * MAGIC_GELU_1); // 3 * C2

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)15;
        for (ptrdiff_t i = 0; i < simd_end; i += 16)
        {
            __m512 x512 = _mm512_loadu_ps(x + i);
            __m512 xsq512 = _mm512_mul_ps(x512, x512);
            __m512 xcube512 = _mm512_mul_ps(xsq512, x512);
            __m512 c1x512 = _mm512_mul_ps(c1_512, x512);
            __m512 inner512 = _mm512_fmadd_ps(xcube512, c2_512, c1x512);

            __m512 t512 = Sleef_tanhf16_u10(inner512);
            __m512 tsq512 = _mm512_mul_ps(t512, t512);

            __m512 one_minus_tsq512 = _mm512_sub_ps(ones512, tsq512);
            __m512 gprime512 = _mm512_fmadd_ps(xsq512, c2_x3_512, c1_512);

            __m512 term1 = _mm512_mul_ps(half512, _mm512_add_ps(ones512, t512));
            __m512 term2 = _mm512_mul_ps(half512, _mm512_mul_ps(x512, _mm512_mul_ps(gprime512, one_minus_tsq512)));
            __m512 res = _mm512_add_ps(term1, term2);
            _mm512_storeu_ps(x + i, res);
        }
#elif defined(__AVX__)
        const __m256 ones256 = _mm256_set1_ps(1.0f);
        const __m256 half256 = _mm256_set1_ps(0.5f);
        const __m256 c1_256 = _mm256_set1_ps(MAGIC_GELU_1);
        const __m256 c2_256 = _mm256_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1);
        const __m256 c2_x3_256 = _mm256_set1_ps(3.0f * MAGIC_GELU_2 * MAGIC_GELU_1);

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)7;
        for (ptrdiff_t i = 0; i < simd_end; i += 8)
        {
            __m256 x256 = _mm256_loadu_ps(x + i);
            __m256 xsq256 = _mm256_mul_ps(x256, x256);
            __m256 xcube256 = _mm256_mul_ps(xsq256, x256);
            __m256 c1x256 = _mm256_mul_ps(c1_256, x256);

#if defined(__AVX2__)
            __m256 inner256 = _mm256_fmadd_ps(xcube256, c2_256, c1x256);
            __m256 gprime256 = _mm256_fmadd_ps(xsq256, c2_x3_256, c1_256);
#else
            __m256 inner256 = _mm256_add_ps(_mm256_mul_ps(xcube256, c2_256), c1x256);
            __m256 gprime256 = _mm256_add_ps(_mm256_mul_ps(xsq256, c2_x3_256), c1_256);
#endif

            __m256 t256 = Sleef_tanhf8_u10(inner256);
            __m256 tsq256 = _mm256_mul_ps(t256, t256);
            __m256 one_minus_tsq256 = _mm256_sub_ps(ones256, tsq256);

            __m256 term1 = _mm256_mul_ps(half256, _mm256_add_ps(ones256, t256));
            __m256 term2 = _mm256_mul_ps(half256, _mm256_mul_ps(x256, _mm256_mul_ps(gprime256, one_minus_tsq256)));

            _mm256_storeu_ps(x + i, _mm256_add_ps(term1, term2));
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        const __m128 ones128 = _mm_set1_ps(1.0f);
        const __m128 half128 = _mm_set1_ps(0.5f);
        const __m128 c1_128 = _mm_set1_ps(MAGIC_GELU_1);
        const __m128 c2_128 = _mm_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1);
        const __m128 c2_x3_128 = _mm_set1_ps(3.0f * MAGIC_GELU_2 * MAGIC_GELU_1);

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)3;
        for (ptrdiff_t i = 0; i < simd_end; i += 4)
        {
            __m128 x128 = _mm_loadu_ps(x + i);
            __m128 xsq128 = _mm_mul_ps(x128, x128);
            __m128 xcube128 = _mm_mul_ps(xsq128, x128);
            __m128 c1x128 = _mm_mul_ps(c1_128, x128);

#ifdef __FMA__
            __m128 inner128 = _mm_fmadd_ps(xcube128, c2_128, c1x128);
            __m128 gprime128 = _mm_fmadd_ps(xsq128, c2_x3_128, c1_128);
#else
            __m128 inner128 = _mm_add_ps(_mm_mul_ps(xcube128, c2_128), c1x128);
            __m128 gprime128 = _mm_add_ps(_mm_mul_ps(xsq128, c2_x3_128), c1_128);
#endif

            __m128 t128 = Sleef_tanhf4_u10(inner128);
            __m128 tsq128 = _mm_mul_ps(t128, t128);
            __m128 one_minus_tsq128 = _mm_sub_ps(ones128, tsq128);

            __m128 term1 = _mm_mul_ps(half128, _mm_add_ps(ones128, t128));
            __m128 term2 = _mm_mul_ps(half128, _mm_mul_ps(x128, _mm_mul_ps(gprime128, one_minus_tsq128)));

            _mm_storeu_ps(x + i, _mm_add_ps(term1, term2));
        }
#endif
        for (ptrdiff_t i = simd_end; i < n; ++i)
        {
            float xi = x[i];
            float xsq = xi * xi;

            float inner = MAGIC_GELU_1 * xi + (MAGIC_GELU_2 * MAGIC_GELU_1) * xsq * xi;
            float t = Sleef_tanhf_u10(inner);

            float gprime = MAGIC_GELU_1 + (3.0f * MAGIC_GELU_2 * MAGIC_GELU_1) * xsq;

            float term1 = 0.5f * (1.0f + t);
            float term2 = 0.5f * xi * gprime * (1.0f - t * t);

            x[i] = term1 + term2;
        }
    }

    /// @brief Two-buffer, single-pass GELU derivative: reads src (the
    /// pre-activation value), writes the derivative directly into dst.
    static inline void dGeluInto(float *RESTRICT dst, const float *RESTRICT src, const size_t n) noexcept
    {

        ptrdiff_t simd_end = 0;

#if defined(__AVX512F__)
        const __m512 ones512 = _mm512_set1_ps(1.0f);
        const __m512 half512 = _mm512_set1_ps(0.5f);
        const __m512 c1_512 = _mm512_set1_ps(MAGIC_GELU_1);                          // C1
        const __m512 c2_512 = _mm512_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1);           // C2
        const __m512 c2_x3_512 = _mm512_set1_ps(3.0f * MAGIC_GELU_2 * MAGIC_GELU_1); // 3 * C2

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)15;
        for (ptrdiff_t i = 0; i < simd_end; i += 16)
        {
            __m512 x512 = _mm512_loadu_ps(src + i);
            __m512 xsq512 = _mm512_mul_ps(x512, x512);
            __m512 xcube512 = _mm512_mul_ps(xsq512, x512);
            __m512 c1x512 = _mm512_mul_ps(c1_512, x512);
            __m512 inner512 = _mm512_fmadd_ps(xcube512, c2_512, c1x512);

            __m512 t512 = Sleef_tanhf16_u10(inner512);
            __m512 tsq512 = _mm512_mul_ps(t512, t512);

            __m512 one_minus_tsq512 = _mm512_sub_ps(ones512, tsq512);
            __m512 gprime512 = _mm512_fmadd_ps(xsq512, c2_x3_512, c1_512);

            __m512 term1 = _mm512_mul_ps(half512, _mm512_add_ps(ones512, t512));
            __m512 term2 = _mm512_mul_ps(half512, _mm512_mul_ps(x512, _mm512_mul_ps(gprime512, one_minus_tsq512)));
            __m512 res = _mm512_add_ps(term1, term2);
            _mm512_storeu_ps(dst + i, res);
        }
#elif defined(__AVX__)
        const __m256 ones256 = _mm256_set1_ps(1.0f);
        const __m256 half256 = _mm256_set1_ps(0.5f);
        const __m256 c1_256 = _mm256_set1_ps(MAGIC_GELU_1);
        const __m256 c2_256 = _mm256_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1);
        const __m256 c2_x3_256 = _mm256_set1_ps(3.0f * MAGIC_GELU_2 * MAGIC_GELU_1);

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)7;
        for (ptrdiff_t i = 0; i < simd_end; i += 8)
        {
            __m256 x256 = _mm256_loadu_ps(src + i);
            __m256 xsq256 = _mm256_mul_ps(x256, x256);
            __m256 xcube256 = _mm256_mul_ps(xsq256, x256);
            __m256 c1x256 = _mm256_mul_ps(c1_256, x256);

#if defined(__AVX2__)
            __m256 inner256 = _mm256_fmadd_ps(xcube256, c2_256, c1x256);
            __m256 gprime256 = _mm256_fmadd_ps(xsq256, c2_x3_256, c1_256);
#else
            __m256 inner256 = _mm256_add_ps(_mm256_mul_ps(xcube256, c2_256), c1x256);
            __m256 gprime256 = _mm256_add_ps(_mm256_mul_ps(xsq256, c2_x3_256), c1_256);
#endif

            __m256 t256 = Sleef_tanhf8_u10(inner256);
            __m256 tsq256 = _mm256_mul_ps(t256, t256);
            __m256 one_minus_tsq256 = _mm256_sub_ps(ones256, tsq256);

            __m256 term1 = _mm256_mul_ps(half256, _mm256_add_ps(ones256, t256));
            __m256 term2 = _mm256_mul_ps(half256, _mm256_mul_ps(x256, _mm256_mul_ps(gprime256, one_minus_tsq256)));

            _mm256_storeu_ps(dst + i, _mm256_add_ps(term1, term2));
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        const __m128 ones128 = _mm_set1_ps(1.0f);
        const __m128 half128 = _mm_set1_ps(0.5f);
        const __m128 c1_128 = _mm_set1_ps(MAGIC_GELU_1);
        const __m128 c2_128 = _mm_set1_ps(MAGIC_GELU_2 * MAGIC_GELU_1);
        const __m128 c2_x3_128 = _mm_set1_ps(3.0f * MAGIC_GELU_2 * MAGIC_GELU_1);

        simd_end = (ptrdiff_t)n & ~(ptrdiff_t)3;
        for (ptrdiff_t i = 0; i < simd_end; i += 4)
        {
            __m128 x128 = _mm_loadu_ps(src + i);
            __m128 xsq128 = _mm_mul_ps(x128, x128);
            __m128 xcube128 = _mm_mul_ps(xsq128, x128);
            __m128 c1x128 = _mm_mul_ps(c1_128, x128);

#ifdef __FMA__
            __m128 inner128 = _mm_fmadd_ps(xcube128, c2_128, c1x128);
            __m128 gprime128 = _mm_fmadd_ps(xsq128, c2_x3_128, c1_128);
#else
            __m128 inner128 = _mm_add_ps(_mm_mul_ps(xcube128, c2_128), c1x128);
            __m128 gprime128 = _mm_add_ps(_mm_mul_ps(xsq128, c2_x3_128), c1_128);
#endif

            __m128 t128 = Sleef_tanhf4_u10(inner128);
            __m128 tsq128 = _mm_mul_ps(t128, t128);
            __m128 one_minus_tsq128 = _mm_sub_ps(ones128, tsq128);

            __m128 term1 = _mm_mul_ps(half128, _mm_add_ps(ones128, t128));
            __m128 term2 = _mm_mul_ps(half128, _mm_mul_ps(x128, _mm_mul_ps(gprime128, one_minus_tsq128)));

            _mm_storeu_ps(dst + i, _mm_add_ps(term1, term2));
        }
#endif
        for (ptrdiff_t i = simd_end; i < n; ++i)
        {
            float xi = src[i];
            float xsq = xi * xi;

            float inner = MAGIC_GELU_1 * xi + (MAGIC_GELU_2 * MAGIC_GELU_1) * xsq * xi;
            float t = Sleef_tanhf_u10(inner);

            float gprime = MAGIC_GELU_1 + (3.0f * MAGIC_GELU_2 * MAGIC_GELU_1) * xsq;

            float term1 = 0.5f * (1.0f + t);
            float term2 = 0.5f * xi * gprime * (1.0f - t * t);

            dst[i] = term1 + term2;
        }
    }

} // namespace Deep
