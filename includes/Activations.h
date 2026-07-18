#pragma once
#include <cstddef>
#include <cassert>
#include <cmath>
#include <immintrin.h>
#include <sleef.h>

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
    inline void expf_v(float *x, size_t n)
    {
        size_t i = 0;

#if defined(__AVX512F__)
        size_t simd_end = n - (n % 16);

        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);

            __m512 res = Sleef_expf16_u10avx512f(x_512);

            _mm512_store_ps(x + i, res);
        }

#elif defined(__AVX2__)
        size_t simd_end = n - (n % 8);

        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);

            __m256 res = Sleef_expf8_u10avx2(x_256);

            _mm256_store_ps(x + i, res);
        }

#elif defined(__SSE2__)
        size_t simd_end = n - (n % 4);

        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);

            __m128 res = Sleef_expf4_u10sse2(x_128);

            _mm_store_ps(x + i, res);
        }

#endif

        for (; i < n; ++i)
        {
            x[i] = Sleef_expf_u10(x[i]);
        }
    }
#pragma region relu
    /// @brief RELU(x) = MAX(0, x) for all x
    /// @param x array, \em assumed to be 64-bit aligned!
    /// @param n x length
    inline void relu(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");
        size_t i = 0;

#if defined(__AVX512F__)
        __m512 zeros_512 = _mm512_setzero_ps();
        size_t r = n % 16;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);
            __m512 clamped = _mm512_max_ps(zeros_512, x_512);
            _mm512_store_ps(x + i, clamped);
        }
#elif defined(__AVX2__) || defined(__AVX__)
        __m256 zeros_256 = _mm256_setzero_ps();
        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);
            __m256 clamped = _mm256_max_ps(zeros_256, x_256);
            _mm256_store_ps(x + i, clamped);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 zeros_128 = _mm_setzero_ps();
        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);
            __m128 clamped = _mm_max_ps(zeros_128, x_128);
            _mm_store_ps(x + i, clamped);
        }
#endif

        for (; i < n; i++)
        {
            x[i] = MAX(0.0f, x[i]);
        }
    }

    inline void dRelu(float *RESTRICT x, const size_t n, [[maybe_unused]] bool activated = false) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 ones = _mm512_set1_ps(1.0f);
        __m512 zeros = _mm512_setzero_ps();
        size_t simd_end = n - (n % 16);
        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);
            __mmask16 mask = _mm512_cmp_ps_mask(x_512, zeros, _CMP_GT_OQ);
            // mask blend: select 'ones' where mask=1, 'zeros' where mask=0
            __m512 result = _mm512_mask_blend_ps(mask, zeros, ones);
            _mm512_store_ps(x + i, result);
        }

#elif defined(__AVX2__) || defined(__AVX__)
        __m256 ones = _mm256_set1_ps(1.0f);
        __m256 zeros = _mm256_setzero_ps();
        size_t simd_end = n - (n % 8);
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);
            // AVX comparison: full vector mask (all-1s or all-0s per lane)
            __m256 mask = _mm256_cmp_ps(x_256, zeros, _CMP_GT_OQ);
            // AND: keeps 'ones' where mask=FFFFFFFF, zeros out where mask=00000000
            __m256 result = _mm256_and_ps(ones, mask);
            _mm256_store_ps(x + i, result);
        }

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 ones = _mm_set1_ps(1.0f);
        __m128 zeros = _mm_setzero_ps();
        size_t simd_end = n - (n % 4);
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);
            // SSE comparison: full vector mask (baseline SSE, no predicates needed)
            __m128 mask = _mm_cmpgt_ps(x_128, zeros);
            __m128 result = _mm_and_ps(ones, mask);
            _mm_store_ps(x + i, result);
        }
#endif

        for (; i < n; i++)
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
        assert(n != 0);
        assert(x != nullptr);

        size_t i = 0;

#if defined(__AVX2__)

        const __m256 tiny = _mm256_set1_ps(TANH_TINYLIMIT);
        const __m256 poly_limit = _mm256_set1_ps(TANH_POLY_LIMIT);
        const __m256 big = _mm256_set1_ps(TANH_BIGLIMIT);

        const __m256 one = _mm256_set1_ps(1.0f);
        const __m256 two = _mm256_set1_ps(2.0f);
        const __m256 zero = _mm256_setzero_ps();

        const __m256 p0 = _mm256_set1_ps(P0);
        const __m256 p1 = _mm256_set1_ps(P1);
        const __m256 p2 = _mm256_set1_ps(P2);

        const __m256 q0 = _mm256_set1_ps(Q0);
        const __m256 q1 = _mm256_set1_ps(Q1);
        const __m256 q2 = _mm256_set1_ps(Q2);

        const __m256 sign_mask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x80000000));

        const __m256 abs_mask =
            _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));

        for (; i + 8 <= n; i += 8)
        {
            __m256 x0 = _mm256_load_ps(x + i);

            __m256 ax = _mm256_and_ps(x0, abs_mask);

            __m256 big_mask =
                _mm256_cmp_ps(ax, big, _CMP_GE_OQ);

            __m256 poly_mask =
                _mm256_cmp_ps(ax, poly_limit, _CMP_LT_OQ);

            __m256 tiny_mask =
                _mm256_cmp_ps(ax, tiny, _CMP_LT_OQ);

            //
            // Polynomial path
            //
            __m256 s = _mm256_mul_ps(x0, x0);

#ifdef __FMA__

            __m256 p = _mm256_fmadd_ps(p0, s, p1);
            p = _mm256_fmadd_ps(p, s, p2);

            __m256 q = _mm256_add_ps(s, q0);
            q = _mm256_fmadd_ps(q, s, q1);
            q = _mm256_fmadd_ps(q, s, q2);

#else

            __m256 p = _mm256_add_ps(
                _mm256_mul_ps(p0, s),
                p1);

            p = _mm256_add_ps(
                _mm256_mul_ps(p, s),
                p2);

            __m256 q = _mm256_add_ps(s, q0);

            q = _mm256_add_ps(
                _mm256_mul_ps(q, s),
                q1);

            q = _mm256_add_ps(
                _mm256_mul_ps(q, s),
                q2);

#endif

            __m256 poly =
                _mm256_add_ps(
                    x0,
                    _mm256_mul_ps(
                        _mm256_mul_ps(x0, s),
                        _mm256_div_ps(p, q)));

            //
            // Exponential path
            //
            alignas(32) float exp_input[8];
            alignas(32) float exp_output[8];

            alignas(32) float values[8];

            _mm256_store_ps(values, ax);

            bool need_exp = false;

            for (int j = 0; j < 8; j++)
            {
                if (!((reinterpret_cast<uint32_t *>(&big_mask)[j])))
                {
                    exp_input[j] = values[j] * 2.0f;
                    need_exp = true;
                }
                else
                {
                    exp_input[j] = 0.0f;
                }
            }

            if (need_exp)
            {
                expf_v(exp_input, 8);

                for (int j = 0; j < 8; j++)
                {
                    exp_output[j] =
                        1.0f -
                        2.0f / (exp_input[j] + 1.0f);
                }
            }

            __m256 exp_result =
                _mm256_load_ps(exp_output);

            //
            // restore sign
            //
            exp_result =
                _mm256_or_ps(
                    _mm256_and_ps(x0, sign_mask),
                    _mm256_and_ps(exp_result, abs_mask));

            //
            // Select result
            //
            __m256 result = x0;

            result =
                _mm256_blendv_ps(
                    result,
                    exp_result,
                    _mm256_andnot_ps(big_mask, poly_mask));

            result =
                _mm256_blendv_ps(
                    result,
                    poly,
                    poly_mask);

            result =
                _mm256_blendv_ps(
                    result,
                    _mm256_or_ps(
                        _mm256_and_ps(x0, sign_mask),
                        one),
                    big_mask);

            result =
                _mm256_blendv_ps(
                    result,
                    x0,
                    tiny_mask);

            _mm256_store_ps(x + i, result);
        }

#endif

        //
        // Scalar remainder
        //
        for (; i < n; i++)
        {
            float v = x[i];
            float av = fabsf(v);

            if (av < TANH_TINYLIMIT)
            {
                continue;
            }
            else if (av >= TANH_BIGLIMIT)
            {
                x[i] = copysignf(1.0f, v);
            }
            else if (av < TANH_POLY_LIMIT)
            {
                float s = v * v;

                x[i] =
                    v +
                    v * s *
                        (((P0 * s + P1) * s + P2) /
                         (((s + Q0) * s + Q1) * s + Q2));
            }
            else
            {
                float e = Sleef_expf_u10(2.0f * av);
                x[i] =
                    copysignf(1.0f - 2.0f / (e + 1.0f), v);
            }
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
#pragma omp parallel for
        for (size_t i = 0; i < n; i++)
        {
            x[i] = 1.0f;
        }
    }
}