#pragma once
#include <cassert>
#include <cstddef>
#include <immintrin.h>
#include <omp.h>
#include <sleef.h>
#include <deepity/utils/activations/ActivationMacros.h>

/**
 * @file Tanh.h
 * @brief tanh and its derivative (plain and two-buffer variants), with
 * AVX512/AVX2/SSE paths and a scalar tail.
 */

namespace Deep
{

    /// @brief In-place hyperbolic tangent activation.
    /// @param x Array to activate in place.
    /// @param n Length of x.
    static inline void tanh(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t simd_end = 0;

#if defined(__AVX512F__)

        simd_end = n & ~(size_t)15;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(simd_end); i += 16)
        {
            __m512 v = _mm512_loadu_ps(x + i);
            v = Sleef_tanhf16_u10avx512f(v);
            _mm512_storeu_ps(x + i, v);
        }

#elif defined(__AVX2__)

        simd_end = n & ~(size_t)7;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(simd_end); i += 8)
        {
            __m256 v = _mm256_loadu_ps(x + i);
            v = Sleef_tanhf8_u10avx2(v);
            _mm256_storeu_ps(x + i, v);
        }

#elif defined(__AVX__)

        simd_end = n & ~(size_t)7;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(simd_end); i += 8)
        {
            __m256 v = _mm256_loadu_ps(x + i);
            v = Sleef_tanhf8_u10avx(v);
            _mm256_storeu_ps(x + i, v);
        }

#elif defined(__SSE4_1__) || defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)

        simd_end = n & ~(size_t)3;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < static_cast<ptrdiff_t>(simd_end); i += 4)
        {
            __m128 v = _mm_loadu_ps(x + i);

#if defined(__SSE4_1__)
            v = Sleef_tanhf4_u10sse4(v);
#else
            v = Sleef_tanhf4_u10sse2(v);
#endif

            _mm_storeu_ps(x + i, v);
        }

#endif

        for (size_t i = simd_end; i < n; ++i)
        {
            x[i] = Sleef_tanhf_u10(x[i]);
        }
    }
    /// @brief In-place tanh derivative: 1 - tanh(x)^2.
    /// @param x Array to derive in place.
    /// @param n Length of x.
    /// @param activated If true, x already holds tanh(x) (skips
    /// re-computing it); if false, x holds the pre-activation value and
    /// tanh(x) is computed first.
    static inline void dTanh(float *RESTRICT x,
                             const size_t n,
                             const bool activated = false) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t i = 0;

        if (activated)
        {
#if defined(__AVX512F__)

            const __m512 ones = _mm512_set1_ps(1.0f);
            const size_t simd_end = n & ~(size_t)15;

            for (; i < simd_end; i += 16)
            {
                const __m512 t = _mm512_loadu_ps(x + i);

#ifdef __FMA__
                const __m512 res = _mm512_fnmadd_ps(t, t, ones);
#else
                const __m512 res = _mm512_sub_ps(
                    ones,
                    _mm512_mul_ps(t, t));
#endif

                _mm512_storeu_ps(x + i, res);
            }

#elif defined(__AVX2__)

            const __m256 ones = _mm256_set1_ps(1.0f);
            const size_t simd_end = n & ~(size_t)7;

            for (; i < simd_end; i += 8)
            {
                const __m256 t = _mm256_loadu_ps(x + i);

#ifdef __FMA__
                const __m256 res = _mm256_fnmadd_ps(t, t, ones);
#else
                const __m256 res = _mm256_sub_ps(
                    ones,
                    _mm256_mul_ps(t, t));
#endif

                _mm256_storeu_ps(x + i, res);
            }

#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)

            const __m128 ones = _mm_set1_ps(1.0f);
            const size_t simd_end = n & ~(size_t)3;

            for (; i < simd_end; i += 4)
            {
                const __m128 t = _mm_loadu_ps(x + i);

#ifdef __FMA__
                const __m128 res = _mm_fnmadd_ps(t, t, ones);
#else
                const __m128 res = _mm_sub_ps(
                    ones,
                    _mm_mul_ps(t, t));
#endif

                _mm_storeu_ps(x + i, res);
            }

#endif

            for (; i < n; ++i)
            {
                const float t = x[i];
                x[i] = 1.0f - t * t;
            }

            return;
        }

#if defined(__AVX512F__)

        {
            const __m512 ones = _mm512_set1_ps(1.0f);
            const size_t simd_end = n & ~(size_t)15;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
            for (ptrdiff_t j = 0;
                 j < static_cast<ptrdiff_t>(simd_end);
                 j += 16)
            {
                __m512 t = _mm512_loadu_ps(x + j);
                t = Sleef_tanhf16_u10avx512f(t);

#ifdef __FMA__
                t = _mm512_fnmadd_ps(t, t, ones);
#else
                t = _mm512_sub_ps(
                    ones,
                    _mm512_mul_ps(t, t));
#endif

                _mm512_storeu_ps(x + j, t);
            }

            i = simd_end;
        }

#elif defined(__AVX2__)

        {
            const __m256 ones = _mm256_set1_ps(1.0f);
            const size_t simd_end = n & ~(size_t)7;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
            for (ptrdiff_t j = 0;
                 j < static_cast<ptrdiff_t>(simd_end);
                 j += 8)
            {
                __m256 t = _mm256_loadu_ps(x + j);
                t = Sleef_tanhf8_u10avx2(t);

#ifdef __FMA__
                t = _mm256_fnmadd_ps(t, t, ones);
#else
                t = _mm256_sub_ps(
                    ones,
                    _mm256_mul_ps(t, t));
#endif

                _mm256_storeu_ps(x + j, t);
            }

            i = simd_end;
        }

#elif defined(__AVX__)

        {
            const __m256 ones = _mm256_set1_ps(1.0f);
            const size_t simd_end = n & ~(size_t)7;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
            for (ptrdiff_t j = 0;
                 j < static_cast<ptrdiff_t>(simd_end);
                 j += 8)
            {
                __m256 t = _mm256_loadu_ps(x + j);
                t = Sleef_tanhf8_u10avx(t);

#ifdef __FMA__
                t = _mm256_fnmadd_ps(t, t, ones);
#else
                t = _mm256_sub_ps(
                    ones,
                    _mm256_mul_ps(t, t));
#endif

                _mm256_storeu_ps(x + j, t);
            }

            i = simd_end;
        }

#elif defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)

        {
            const __m128 ones = _mm_set1_ps(1.0f);
            const size_t simd_end = n & ~(size_t)3;

#pragma omp parallel for schedule(static) if (n > 16384 && !omp_in_parallel())
            for (ptrdiff_t j = 0;
                 j < static_cast<ptrdiff_t>(simd_end);
                 j += 4)
            {
                __m128 t = _mm_loadu_ps(x + j);

#if defined(__SSE4_1__)
                t = Sleef_tanhf4_u10sse4(t);
#else
                t = Sleef_tanhf4_u10sse2(t);
#endif

#ifdef __FMA__
                t = _mm_fnmadd_ps(t, t, ones);
#else
                t = _mm_sub_ps(
                    ones,
                    _mm_mul_ps(t, t));
#endif

                _mm_storeu_ps(x + j, t);
            }

            i = simd_end;
        }

#else

        i = 0;

#endif

        /*
         * Scalar tail.
         */
        for (; i < n; ++i)
        {
            const float t = Sleef_tanhf_u10(x[i]);
            x[i] = 1.0f - t * t;
        }
    }

    /// @brief Two-buffer, single-pass tanh derivative: reads src, writes
    /// 1-tanh(src)^2 directly into dst. Unlike dTanh(x, n, false), which
    /// needs two full passes internally (activate in place, then derive
    /// in place) plus a separate copy beforehand if src can't be
    /// mutated, this computes tanh(src[i]) into a register and derives
    /// from that same register immediately, with only one read of src
    /// and one write to dst, no intermediate array pass at all.
    static inline void dTanhInto(float *RESTRICT dst, const float *RESTRICT src, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(dst != nullptr && "dst must not be null.");
        assert(src != nullptr && "src must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 ones = _mm512_set1_ps(1.0f);
        size_t simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 16)
        {
            __m512 s = _mm512_loadu_ps(src + j);
            __m512 t = Sleef_tanhf16_u10avx512f(s);
#ifdef __FMA__
            __m512 res = _mm512_fnmadd_ps(t, t, ones); // 1 - t*t
#else
            __m512 res = _mm512_sub_ps(ones, _mm512_mul_ps(t, t));
#endif
            _mm512_storeu_ps(dst + j, res);
        }
        i = simd_end;
#elif defined(__AVX2__)
        __m256 ones = _mm256_set1_ps(1.0f);
        size_t simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 8)
        {
            __m256 s = _mm256_loadu_ps(src + j);
            __m256 t = Sleef_tanhf8_u10avx2(s);
#ifdef __FMA__
            __m256 res = _mm256_fnmadd_ps(t, t, ones); // 1 - t*t
#else
            __m256 res = _mm256_sub_ps(ones, _mm256_mul_ps(t, t));
#endif
            _mm256_storeu_ps(dst + j, res);
        }
        i = simd_end;
#elif defined(__SSE4_1__) || defined(__SSE2__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 ones = _mm_set1_ps(1.0f);
        size_t simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t j = 0; j < (ptrdiff_t)simd_end; j += 4)
        {
            __m128 s = _mm_loadu_ps(src + j);
#if defined(__SSE4_1__)
            __m128 t = Sleef_tanhf4_u10sse4(s);
#else
            __m128 t = Sleef_tanhf4_u10sse2(s);
#endif
#ifdef __FMA__
            __m128 res = _mm_fnmadd_ps(t, t, ones); // 1 - t*t
#else
            __m128 res = _mm_sub_ps(ones, _mm_mul_ps(t, t));
#endif
            _mm_storeu_ps(dst + j, res);
        }
        i = simd_end;
#endif

        for (; i < n; ++i)
        {
            float t = Sleef_tanhf_u10(src[i]);
            dst[i] = 1.0f - t * t;
        }
    }

} // namespace Deep
