#pragma once
#include <cstddef>
#include <immintrin.h>
#include <omp.h>
#include <sleef.h>

/**
 * @file VectorMath.h
 * @brief SIMD-vectorized exp/log helpers (AVX512/AVX2/SSE with a scalar
 * tail). Currently unused by the rest of the codebase, kept for planned
 * future use (e.g. a vectorized softmax path) rather than deleted.
 */

namespace Deep
{
    /// @brief In-place vectorized exp: x[i] = exp(x[i]) for all i.
    static inline void expf_v(float *x, size_t n)
    {
        size_t simd_end = 0;

#if defined(__AVX512F__)
        simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end); i += 16)
        {
            __m512 x_512 = _mm512_loadu_ps(x + i);
            __m512 res = Sleef_expf16_u10avx512f(x_512);
            _mm512_storeu_ps(x + i, res);
        }

#elif defined(__AVX2__)
        simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end); i += 8)
        {
            __m256 x_256 = _mm256_loadu_ps(x + i);
            __m256 res = Sleef_expf8_u10avx2(x_256);
            _mm256_storeu_ps(x + i, res);
        }

#elif defined(__SSE2__)
        simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end); i += 4)
        {
            __m128 x_128 = _mm_loadu_ps(x + i);
            __m128 res = Sleef_expf4_u10sse2(x_128);
            _mm_storeu_ps(x + i, res);
        }
#endif

        for (size_t i = simd_end; i < n; ++i)
        {
            x[i] = Sleef_expf_u10(x[i]);
        }
    }

    /// @brief In-place vectorized log: x[i] = log(x[i]) for all i.
    inline void logf_v(float *x, size_t n)
    {
        size_t simd_end = 0;

#if defined(__AVX512F__)
        simd_end = n - (n % 16);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end); i += 16)
        {
            __m512 x_512 = _mm512_loadu_ps(x + i);
            __m512 res = Sleef_logf16_u10avx512f(x_512);
            _mm512_storeu_ps(x + i, res);
        }

#elif defined(__AVX2__)
        simd_end = n - (n % 8);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end); i += 8)
        {
            __m256 x_256 = _mm256_loadu_ps(x + i);
            __m256 res = Sleef_logf8_u10avx2(x_256);
            _mm256_storeu_ps(x + i, res);
        }

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        simd_end = n - (n % 4);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
        for (ptrdiff_t i = 0; i < (ptrdiff_t)(simd_end); i += 4)
        {
            __m128 x_128 = _mm_loadu_ps(x + i);
            __m128 res = Sleef_logf4_u10sse2(x_128);
            _mm_storeu_ps(x + i, res);
        }
#endif
        for (size_t i = simd_end; i < n; ++i)
        {
            x[i] = Sleef_logf_u10(x[i]);
        }
    }
} // namespace Deep
