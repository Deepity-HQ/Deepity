#pragma once
#include <cassert>
#include <cstddef>
#include <immintrin.h>
#include <omp.h>
#include <deepity/utils/activations/ActivationMacros.h>

/// @brief Guarded so this doesn't clash with a system MAX macro (notably
/// Windows headers) if one is already defined.
#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

/**
 * @file Relu.h
 * @brief ReLU and its derivative (plain and two-buffer variants), with
 * AVX512/AVX2/SSE paths and a scalar tail.
 */

namespace Deep
{
    /// @brief RELU(x) = MAX(0, x) for all x
    /// @param x array
    /// @param n x length
    static inline void relu(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t simd_end = 0;

#if defined(__AVX512F__)
        const __m512 zeros512 = _mm512_setzero_ps();
        simd_end = n - (n % 16);

        for (size_t i = 0; i < simd_end; i += 16)
        {
            __m512 xi = _mm512_loadu_ps(x + i);
            xi = _mm512_max_ps(zeros512, xi);
            _mm512_storeu_ps(x + i, xi);
        }

#elif defined(__AVX2__) || defined(__AVX__)
        const __m256 zeros256 = _mm256_setzero_ps();
        simd_end = n - (n % 8);

        for (size_t i = 0; i < simd_end; i += 8)
        {
            __m256 xi = _mm256_loadu_ps(x + i);
            xi = _mm256_max_ps(zeros256, xi);
            _mm256_storeu_ps(x + i, xi);
        }

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        const __m128 zeros128 = _mm_setzero_ps();
        simd_end = n - (n % 4);

        for (size_t i = 0; i < simd_end; i += 4)
        {
            __m128 xi = _mm_loadu_ps(x + i);
            xi = _mm_max_ps(zeros128, xi);
            _mm_storeu_ps(x + i, xi);
        }
#endif

        for (size_t i = simd_end; i < n; i++)
        {
            x[i] = MAX(0.0f, x[i]);
        }
    }

    /// @brief In-place ReLU derivative: 1 where x > 0, else 0.
    /// @param x Array to derive in place.
    /// @param n Length of x.
    /// @param activated Unused -- ReLU's derivative only depends on sign,
    /// so whether x already holds the activated value makes no difference.
    /// Present only for signature consistency with DerivativeFn.
    static inline void dRelu(float *RESTRICT x, const size_t n, [[maybe_unused]] bool activated = false) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t simd_end = 0;

#if defined(__AVX512F__)
        __m512 ones = _mm512_set1_ps(1.0f);
        __m512 zeros = _mm512_setzero_ps();
        size_t simd_end4 = n - (n % 64);

#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end4); i += 64)
        {
            __m512 x0 = _mm512_loadu_ps(x + i);
            __m512 x1 = _mm512_loadu_ps(x + i + 16);
            __m512 x2 = _mm512_loadu_ps(x + i + 32);
            __m512 x3 = _mm512_loadu_ps(x + i + 48);

            __mmask16 m0 = _mm512_cmp_ps_mask(x0, zeros, _CMP_GT_OQ);
            __mmask16 m1 = _mm512_cmp_ps_mask(x1, zeros, _CMP_GT_OQ);
            __mmask16 m2 = _mm512_cmp_ps_mask(x2, zeros, _CMP_GT_OQ);
            __mmask16 m3 = _mm512_cmp_ps_mask(x3, zeros, _CMP_GT_OQ);

            _mm512_storeu_ps(x + i, _mm512_mask_blend_ps(m0, zeros, ones));
            _mm512_storeu_ps(x + i + 16, _mm512_mask_blend_ps(m1, zeros, ones));
            _mm512_storeu_ps(x + i + 32, _mm512_mask_blend_ps(m2, zeros, ones));
            _mm512_storeu_ps(x + i + 48, _mm512_mask_blend_ps(m3, zeros, ones));
        }

        simd_end = n - (n % 16);
        for (size_t i = simd_end4; i < simd_end; i += 16)
        {
            __m512 x0 = _mm512_loadu_ps(x + i);
            __mmask16 m0 = _mm512_cmp_ps_mask(x0, zeros, _CMP_GT_OQ);
            _mm512_storeu_ps(x + i, _mm512_mask_blend_ps(m0, zeros, ones));
        }

#elif defined(__AVX2__) || defined(__AVX__)
        __m256 ones = _mm256_set1_ps(1.0f);
        __m256 zeros = _mm256_setzero_ps();
        size_t simd_end4 = n - (n % 32);

#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end4); i += 32)
        {
            __m256 x0 = _mm256_loadu_ps(x + i);
            __m256 x1 = _mm256_loadu_ps(x + i + 8);
            __m256 x2 = _mm256_loadu_ps(x + i + 16);
            __m256 x3 = _mm256_loadu_ps(x + i + 24);

            x0 = _mm256_and_ps(ones, _mm256_cmp_ps(x0, zeros, _CMP_GT_OQ));
            x1 = _mm256_and_ps(ones, _mm256_cmp_ps(x1, zeros, _CMP_GT_OQ));
            x2 = _mm256_and_ps(ones, _mm256_cmp_ps(x2, zeros, _CMP_GT_OQ));
            x3 = _mm256_and_ps(ones, _mm256_cmp_ps(x3, zeros, _CMP_GT_OQ));

            _mm256_storeu_ps(x + i, x0);
            _mm256_storeu_ps(x + i + 8, x1);
            _mm256_storeu_ps(x + i + 16, x2);
            _mm256_storeu_ps(x + i + 24, x3);
        }

        simd_end = n - (n % 8);
        for (size_t i = simd_end4; i < simd_end; i += 8)
        {
            __m256 x0 = _mm256_loadu_ps(x + i);
            _mm256_storeu_ps(x + i, _mm256_and_ps(ones, _mm256_cmp_ps(x0, zeros, _CMP_GT_OQ)));
        }

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_setzero_ps();
        size_t simd_end4 = n - (n % 16);

#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end4); i += 16)
        {
            __m128 x0 = _mm_loadu_ps(x + i);
            __m128 x1 = _mm_loadu_ps(x + i + 4);
            __m128 x2 = _mm_loadu_ps(x + i + 8);
            __m128 x3 = _mm_loadu_ps(x + i + 12);

            x0 = _mm_and_ps(ones, _mm_cmpgt_ps(x0, zeros));
            x1 = _mm_and_ps(ones, _mm_cmpgt_ps(x1, zeros));
            x2 = _mm_and_ps(ones, _mm_cmpgt_ps(x2, zeros));
            x3 = _mm_and_ps(ones, _mm_cmpgt_ps(x3, zeros));

            _mm_storeu_ps(x + i, x0);
            _mm_storeu_ps(x + i + 4, x1);
            _mm_storeu_ps(x + i + 8, x2);
            _mm_storeu_ps(x + i + 12, x3);
        }

        simd_end = n - (n % 4);
        for (size_t i = simd_end4; i < simd_end; i += 4)
        {
            __m128 x0 = _mm_loadu_ps(x + i);
            _mm_storeu_ps(x + i, _mm_and_ps(ones, _mm_cmpgt_ps(x0, zeros)));
        }
#endif

        for (size_t i = simd_end; i < n; i++)
        {
            x[i] = (x[i] > 0.0f) ? 1.0f : 0.0f;
        }
    }

    /// @brief Two-buffer variant of dRelu: reads src, writes the
    /// derivative directly into dst.
    static inline void dReluInto(float *RESTRICT dst, const float *RESTRICT src, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(dst != nullptr && "dst must not be null.");
        assert(src != nullptr && "src must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 ones = _mm512_set1_ps(1.0f);
        __m512 zeros = _mm512_setzero_ps();
        size_t simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 16)
        {
            __m512 s = _mm512_loadu_ps(src + j);
            __mmask16 m = _mm512_cmp_ps_mask(s, zeros, _CMP_GT_OQ);
            _mm512_storeu_ps(dst + j, _mm512_mask_blend_ps(m, zeros, ones));
        }
        i = simd_end;
#elif defined(__AVX2__) || defined(__AVX__)
        __m256 ones = _mm256_set1_ps(1.0f);
        __m256 zeros = _mm256_setzero_ps();
        size_t simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 8)
        {
            __m256 s = _mm256_loadu_ps(src + j);
            _mm256_storeu_ps(dst + j, _mm256_and_ps(ones, _mm256_cmp_ps(s, zeros, _CMP_GT_OQ)));
        }
        i = simd_end;
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_setzero_ps();
        size_t simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 4)
        {
            __m128 s = _mm_loadu_ps(src + j);
            _mm_storeu_ps(dst + j, _mm_and_ps(ones, _mm_cmpgt_ps(s, zeros)));
        }
        i = simd_end;
#endif

        for (; i < n; i++)
        {
            dst[i] = (src[i] > 0.0f) ? 1.0f : 0.0f;
        }
    }
} // namespace Deep
