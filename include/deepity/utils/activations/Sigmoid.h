#pragma once
#include <cassert>
#include <cmath>
#include <cstddef>
#include <immintrin.h>
#include <omp.h>
#include <sleef.h>
#include <deepity/utils/activations/ActivationMacros.h>

/**
 * @file Sigmoid.h
 * @brief Logistic sigmoid and Elliot ("fast") sigmoid, and their
 * derivatives (plain and two-buffer variants), with AVX512/AVX2/SSE paths
 * and a scalar tail.
 */

namespace Deep
{

    /// @brief Implements the \em Logistic \em Sigmoid approximation, i.e. `S(x) = 1 / (1 + e^(-x))`
    /// @param x array, \em does not need to be aligned
    /// @param n x length
    static inline void sigmoid(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 neg_one = _mm512_set1_ps(-1.0f);
        size_t r = n % 16;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_loadu_ps(x + i);
            __m512 neg_x = _mm512_mul_ps(x_512, neg_one);
            __m512 exp_neg_x = Sleef_expf16_u10avx512f(neg_x);
            __m512 den = _mm512_add_ps(exp_neg_x, one);
            __m512 sig = _mm512_div_ps(one, den);
            _mm512_storeu_ps(x + i, sig);
        }
#elif defined(__AVX2__)
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_one = _mm256_set1_ps(-1.0f);
        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_loadu_ps(x + i);
            __m256 neg_x = _mm256_mul_ps(x_256, neg_one);
            __m256 exp_neg_x = Sleef_expf8_u10avx2(neg_x);
            __m256 den = _mm256_add_ps(exp_neg_x, one);
            __m256 sig = _mm256_div_ps(one, den);
            _mm256_storeu_ps(x + i, sig);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_one = _mm_set1_ps(-1.0f);
        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_loadu_ps(x + i);
            __m128 neg_x = _mm_mul_ps(x_128, neg_one);
            __m128 exp_neg_x = Sleef_expf4_u10sse2(neg_x);
            __m128 den = _mm_add_ps(exp_neg_x, one);
            __m128 sig = _mm_div_ps(one, den);
            _mm_storeu_ps(x + i, sig);
        }
#endif
        for (; i < n; i++)
        {
            x[i] = 1.0f / (1.0f + std::exp(-x[i]));
        }
    }

    /// @brief Implements the \em Elliot \em Sigmoid approximation, i.e. `S(x) = (1/2)((x / (1 + |x|)) + 1)`
    /// @param x array, \em does not need to be aligned
    /// @param n x length
    inline void e_sigmoid(float *RESTRICT x, const size_t n) noexcept
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
            __m512 x_512 = _mm512_loadu_ps(x + i);
            __m512 den = _mm512_add_ps(
                _mm512_abs_ps(x_512),
                one);
            __m512 div = _mm512_div_ps(x_512, den);
            __m512 sig = _mm512_fmadd_ps(div, half, half);

            _mm512_storeu_ps(x + i, sig);
        }
#elif defined(__AVX2__)
        __m256 half = _mm256_set1_ps(0.5f);
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF)); // for 256-bit abs
        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_loadu_ps(x + i);
            __m256 den = _mm256_add_ps(
                _mm256_and_ps(x_256, mask),
                one);
            __m256 div = _mm256_div_ps(x_256, den);
#ifdef __FMA__
            __m256 sig = _mm256_fmadd_ps(div, half, half);
#else
            __m256 sig = _mm256_add_ps(_mm256_mul_ps(div, half), half);
#endif

            _mm256_storeu_ps(x + i, sig);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 half = _mm_set1_ps(0.5f);
        __m128 one = _mm_set1_ps(1.0f);
        __m128 mask = _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF)); // for 256-bit abs
        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_loadu_ps(x + i);
            __m128 den = _mm_add_ps(
                _mm_and_ps(x_128, mask),
                one);
            __m128 div = _mm_div_ps(x_128, den);
#ifdef __FMA__
            __m128 sig = _mm_fmadd_ps(div, half, half);
#else
            __m128 sig = _mm_add_ps(_mm_mul_ps(div, half), half);
#endif

            _mm_storeu_ps(x + i, sig);
        }
#endif
        for (; i < n; i++)
        {
            x[i] = 0.5f * (x[i] / (1.0f + fabsf(x[i])) + 1.0f);
        }
    }

    /// @brief In-place logistic sigmoid derivative: sigmoid(x)*(1-sigmoid(x)).
    /// @param x Array to derive in place.
    /// @param n Length of x.
    /// @param activated If true, x already holds sigmoid(x); if false, x
    /// holds the pre-activation value and sigmoid(x) is computed first.
    static inline void dSigmoid(float *RESTRICT x, const size_t n, bool activated = false) noexcept
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
            __m512 x_512 = _mm512_loadu_ps(x + i);
            __m512 d = _mm512_fnmadd_ps(x_512, x_512, x_512); // d = x * (1 - x) = x - x^2 = -x*x + x
            _mm512_storeu_ps(x + i, d);
        }
#elif defined(__AVX2__)

        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_loadu_ps(x + i);
#ifdef __FMA__
            __m256 d = _mm256_fnmadd_ps(x_256, x_256, x_256); // d = x * (1 - x) = x - x^2 = -x*x + x
#else
            __m256 d = _mm256_sub_ps(x_256, _mm256_mul_ps(x_256, x_256));
#endif
            _mm256_storeu_ps(x + i, d);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_loadu_ps(x + i);

#ifdef __FMA__
            __m128 d = _mm_fnmadd_ps(x_128, x_128, x_128); // d = x * (1 - x) = x - x^2 = -x*x + x
#else
            __m128 d = _mm_sub_ps(x_128, _mm_mul_ps(x_128, x_128));
#endif
            _mm_storeu_ps(x + i, d);
        }
#endif

        for (; i < n; i++)
        {
            x[i] = x[i] * (1.0f - x[i]);
        }
    }

    /// @brief Two-buffer, single-pass sigmoid derivative: reads src,
    /// writes sigmoid(src)*(1-sigmoid(src)) directly into dst. Computes
    /// the sigmoid into a register and derives from that same register
    /// immediately -- one read of src, one write to dst, no intermediate
    /// array pass.
    static inline void dSigmoidInto(float *RESTRICT dst, const float *RESTRICT src, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(dst != nullptr && "dst must not be null.");
        assert(src != nullptr && "src must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 one = _mm512_set1_ps(1.0f);
        __m512 neg_one = _mm512_set1_ps(-1.0f);
        size_t simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 16)
        {
            __m512 s = _mm512_loadu_ps(src + j);
            __m512 neg_s = _mm512_mul_ps(s, neg_one);
            __m512 exp_neg_s = Sleef_expf16_u10avx512f(neg_s);
            __m512 den = _mm512_add_ps(exp_neg_s, one);
            __m512 sig = _mm512_div_ps(one, den);
            __m512 d = _mm512_fnmadd_ps(sig, sig, sig); // sig - sig^2 = sig*(1-sig)
            _mm512_storeu_ps(dst + j, d);
        }
        i = simd_end;
#elif defined(__AVX2__)
        __m256 one = _mm256_set1_ps(1.0f);
        __m256 neg_one = _mm256_set1_ps(-1.0f);
        size_t simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 8)
        {
            __m256 s = _mm256_loadu_ps(src + j);
            __m256 neg_s = _mm256_mul_ps(s, neg_one);
            __m256 exp_neg_s = Sleef_expf8_u10avx2(neg_s);
            __m256 den = _mm256_add_ps(exp_neg_s, one);
            __m256 sig = _mm256_div_ps(one, den);
#ifdef __FMA__
            __m256 d = _mm256_fnmadd_ps(sig, sig, sig);
#else
            __m256 d = _mm256_sub_ps(sig, _mm256_mul_ps(sig, sig));
#endif
            _mm256_storeu_ps(dst + j, d);
        }
        i = simd_end;
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 one = _mm_set1_ps(1.0f);
        __m128 neg_one = _mm_set1_ps(-1.0f);
        size_t simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 4)
        {
            __m128 s = _mm_loadu_ps(src + j);
            __m128 neg_s = _mm_mul_ps(s, neg_one);
            __m128 exp_neg_s = Sleef_expf4_u10sse2(neg_s);
            __m128 den = _mm_add_ps(exp_neg_s, one);
            __m128 sig = _mm_div_ps(one, den);
#ifdef __FMA__
            __m128 d = _mm_fnmadd_ps(sig, sig, sig);
#else
            __m128 d = _mm_sub_ps(sig, _mm_mul_ps(sig, sig));
#endif
            _mm_storeu_ps(dst + j, d);
        }
        i = simd_end;
#endif

        for (; i < n; i++)
        {
            float sig = 1.0f / (1.0f + std::exp(-src[i]));
            dst[i] = sig * (1.0f - sig);
        }
    }

    /// @brief In-place Elliot sigmoid derivative.
    /// @param x Array to derive in place.
    /// @param n Length of x.
    /// @param activated If true, x already holds e_sigmoid(x); if false,
    /// x holds the pre-activation value and e_sigmoid(x) is computed first.
    static inline void d_eSigmoid(float *RESTRICT x,
                                  const size_t n,
                                  const bool activated = false) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        if (!activated)
            e_sigmoid(x, n);

        size_t i = 0;

#if defined(__AVX512F__)

        const __m512 two = _mm512_set1_ps(2.0f);

        const size_t r = n % 16;
        const size_t simd_end = n - r;

        for (; i < simd_end; i += 16)
        {
            const __m512 x_512 = _mm512_loadu_ps(x + i);

            // d = 2 * x * (1 - x)
            const __m512 d = _mm512_mul_ps(
                _mm512_fnmadd_ps(x_512, x_512, x_512),
                two);

            _mm512_storeu_ps(x + i, d);
        }

#elif defined(__AVX2__)

        const __m256 two = _mm256_set1_ps(2.0f);

        const size_t r = n % 8;
        const size_t simd_end = n - r;

        for (; i < simd_end; i += 8)
        {
            const __m256 x_256 = _mm256_loadu_ps(x + i);

#ifdef __FMA__
            // x * (1 - x) = x - x^2
            const __m256 d = _mm256_mul_ps(
                _mm256_fnmadd_ps(x_256, x_256, x_256),
                two);
#else
            const __m256 d = _mm256_mul_ps(
                _mm256_sub_ps(x_256, _mm256_mul_ps(x_256, x_256)),
                two);
#endif

            _mm256_storeu_ps(x + i, d);
        }

#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)

        const __m128 two = _mm_set1_ps(2.0f);

        const size_t r = n % 4;
        const size_t simd_end = n - r;

        for (; i < simd_end; i += 4)
        {
            const __m128 x_128 = _mm_loadu_ps(x + i);

#ifdef __FMA__
            const __m128 d = _mm_mul_ps(
                _mm_fnmadd_ps(x_128, x_128, x_128),
                two);
#else
            const __m128 d = _mm_mul_ps(
                _mm_sub_ps(x_128, _mm_mul_ps(x_128, x_128)),
                two);
#endif

            _mm_storeu_ps(x + i, d);
        }

#endif

        for (; i < n; ++i)
        {
            x[i] = 2.0f * x[i] * (1.0f - x[i]);
        }
    }

    /// @brief Two-buffer, single-pass Elliot sigmoid derivative: reads
    /// src, writes the derivative directly into dst.
    static inline void d_eSigmoidInto(float *RESTRICT dst,
                                      const float *RESTRICT src,
                                      const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(dst != nullptr && "dst must not be null.");
        assert(src != nullptr && "src must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)

        const __m512 half = _mm512_set1_ps(0.5f);
        const __m512 one = _mm512_set1_ps(1.0f);

        const size_t simd_end = n - (n % 16);

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 16)
        {
            const __m512 s = _mm512_loadu_ps(src + j);
            const __m512 a = _mm512_add_ps(_mm512_abs_ps(s), one);
            const __m512 d = _mm512_div_ps(half, _mm512_mul_ps(a, a));

            _mm512_storeu_ps(dst + j, d);
        }

        i = simd_end;

#elif defined(__AVX2__)

        const __m256 half = _mm256_set1_ps(0.5f);
        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 abs_mask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7FFFFFFF));

        const size_t simd_end = n - (n % 8);

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 8)
        {
            const __m256 s = _mm256_loadu_ps(src + j);
            const __m256 a = _mm256_add_ps(
                _mm256_and_ps(s, abs_mask),
                one);

            const __m256 d = _mm256_div_ps(
                half,
                _mm256_mul_ps(a, a));

            _mm256_storeu_ps(dst + j, d);
        }

        i = simd_end;

#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)

        const __m128 half = _mm_set1_ps(0.5f);
        const __m128 one = _mm_set1_ps(1.0f);
        const __m128 abs_mask =
            _mm_castsi128_ps(_mm_set1_epi32(0x7FFFFFFF));

        const size_t simd_end = n - (n % 4);

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 4)
        {
            const __m128 s = _mm_loadu_ps(src + j);
            const __m128 a = _mm_add_ps(
                _mm_and_ps(s, abs_mask),
                one);

            const __m128 d = _mm_div_ps(
                half,
                _mm_mul_ps(a, a));

            _mm_storeu_ps(dst + j, d);
        }

        i = simd_end;

#endif

        for (; i < n; ++i)
        {
            const float a = 1.0f + std::fabs(src[i]);
            dst[i] = 0.5f / (a * a);
        }
    }

} // namespace Deep
