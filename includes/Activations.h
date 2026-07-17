#pragma once
#include <cstddef>
#include <cassert>
#include <cmath>
#include <immintrin.h>

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
    inline void dRelu(float *RESTRICT x, const size_t n) noexcept
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
#define TANH_BIGLIMIT 9.0f
#define p1 -0.3333328194f
#define p2 -0.0015662060f
#define p3 -0.0001658424f
#define q1 0.6031357917f
#define q2 0.0146431154f

#pragma region tanh
    /// @brief TANH(x) = (exp(x) - exp(-x))/(exp(x) + exp(-x))
    /// @param x array, \em assumed to be 64-bit aligned!
    /// @param n x length
    inline void tanh(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        size_t i = 0;

#if defined(__AVX512F__)
        size_t r = n % 16;
        size_t simd_end = n - r;

        __m512 pos_small_check = _mm512_set1_ps(TANH_TINYLIMIT);
        __m512 neg_small_check = _mm512_set1_ps(-TANH_TINYLIMIT);
        __m512 pos_big_check = _mm512_set1_ps(TANH_BIGLIMIT);
        __m512 neg_big_check = _mm512_set1_ps(-TANH_BIGLIMIT);
        __m512 negative1 = _mm512_set1_ps(-1.0f);
        __m512 positive1 = _mm512_set1_ps(1.0f);
        __m512 zeros = _mm512_setzero_ps();
        __m512 mmp1 = _mm512_set1_ps(p1);
        __m512 mmp2 = _mm512_set1_ps(p2);
        __m512 mmp3 = _mm512_set1_ps(p3);
        __m512 mmq1 = _mm512_set1_ps(q1);
        __m512 mmq2 = _mm512_set1_ps(q2);

        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);

            __mmask16 bigmaskpo = _mm512_cmp_ps_mask(x_512, pos_big_check, _CMP_GT_OQ);
            __mmask16 bigmaskne = _mm512_cmp_ps_mask(x_512, neg_big_check, _CMP_LT_OQ);
            __mmask16 tinymask = _mm512_cmp_ps_mask(x_512, neg_small_check, _CMP_GT_OQ) & _mm512_cmp_ps_mask(x_512, pos_small_check, _CMP_LT_OQ);
            __mmask16 bigmask = bigmaskpo | bigmaskne;
            __mmask16 midmask = (~bigmask) & (~tinymask);

            __m512 y = x_512;
            y = _mm512_mask_blend_ps(bigmaskne, y, negative1);
            y = _mm512_mask_blend_ps(bigmaskpo, y, positive1);

            __m512 g = _mm512_mask_mov_ps(zeros, midmask, y);
            __m512 g2 = _mm512_mul_ps(g, g);

            __m512 num = _mm512_fmadd_ps(mmp3, g2, mmp2);
            num = _mm512_fmadd_ps(num, g2, mmp1);

            __m512 den = _mm512_add_ps(g2, mmq2);
            den = _mm512_fmadd_ps(den, g2, mmq1);
            den = _mm512_fmadd_ps(den, g2, positive1);

            __m512 poly = _mm512_fmadd_ps(g, _mm512_mul_ps(g2, _mm512_div_ps(num, den)), g);

            y = _mm512_mask_mov_ps(y, midmask, poly);
            _mm512_store_ps(x + i, y);
        }

#elif defined(__AVX2__)
        size_t r = n % 8;
        size_t simd_end = n - r;

        __m256 pos_small_check = _mm256_set1_ps(TANH_TINYLIMIT);
        __m256 neg_small_check = _mm256_set1_ps(-TANH_TINYLIMIT);
        __m256 pos_big_check = _mm256_set1_ps(TANH_BIGLIMIT);
        __m256 neg_big_check = _mm256_set1_ps(-TANH_BIGLIMIT);
        __m256 negative1 = _mm256_set1_ps(-1.0f);
        __m256 positive1 = _mm256_set1_ps(1.0f);
        __m256 allones = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        __m256 mmp1 = _mm256_set1_ps(p1);
        __m256 mmp2 = _mm256_set1_ps(p2);
        __m256 mmp3 = _mm256_set1_ps(p3);
        __m256 mmq1 = _mm256_set1_ps(q1);
        __m256 mmq2 = _mm256_set1_ps(q2);

        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);

            __m256 bigmaskpo = _mm256_cmp_ps(x_256, pos_big_check, _CMP_GT_OQ);
            __m256 bigmaskne = _mm256_cmp_ps(x_256, neg_big_check, _CMP_LT_OQ);
            __m256 tinymask = _mm256_and_ps(
                _mm256_cmp_ps(x_256, neg_small_check, _CMP_GT_OQ),
                _mm256_cmp_ps(x_256, pos_small_check, _CMP_LT_OQ));
            __m256 bigmask = _mm256_or_ps(bigmaskpo, bigmaskne);
            __m256 midmask = _mm256_andnot_ps(_mm256_or_ps(bigmask, tinymask), allones);

            __m256 y = x_256;
            y = _mm256_blendv_ps(y, negative1, bigmaskne);
            y = _mm256_blendv_ps(y, positive1, bigmaskpo);

            __m256 g = _mm256_and_ps(y, midmask);
            __m256 g2 = _mm256_mul_ps(g, g);

            __m256 ratio;
#ifdef __FMA__
            __m256 num = _mm256_fmadd_ps(mmp3, g2, mmp2);
            num = _mm256_fmadd_ps(num, g2, mmp1);

            __m256 den = _mm256_add_ps(g2, mmq2);
            den = _mm256_fmadd_ps(den, g2, mmq1);
            den = _mm256_fmadd_ps(den, g2, positive1);

            ratio = _mm256_mul_ps(g2, _mm256_div_ps(num, den));
            __m256 poly = _mm256_fmadd_ps(g, ratio, g);
#else
            __m256 num = _mm256_add_ps(_mm256_mul_ps(mmp3, g2), mmp2);
            num = _mm256_add_ps(_mm256_mul_ps(num, g2), mmp1);

            __m256 den = _mm256_add_ps(g2, mmq2);
            den = _mm256_add_ps(_mm256_mul_ps(den, g2), mmq1);
            den = _mm256_add_ps(_mm256_mul_ps(den, g2), positive1);

            ratio = _mm256_mul_ps(g2, _mm256_div_ps(num, den));
            __m256 poly = _mm256_add_ps(_mm256_mul_ps(g, ratio), g);
#endif

            y = _mm256_blendv_ps(y, poly, midmask);
            _mm256_store_ps(x + i, y);
        }

#elif defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)
        size_t r = n % 4;
        size_t simd_end = n - r;

        __m128 pos_small_check = _mm_set1_ps(TANH_TINYLIMIT);
        __m128 neg_small_check = _mm_set1_ps(-TANH_TINYLIMIT);
        __m128 pos_big_check = _mm_set1_ps(TANH_BIGLIMIT);
        __m128 neg_big_check = _mm_set1_ps(-TANH_BIGLIMIT);
        __m128 negative1 = _mm_set1_ps(-1.0f);
        __m128 positive1 = _mm_set1_ps(1.0f);
        __m128 allones = _mm_castsi128_ps(_mm_set1_epi32(-1));
        __m128 mmp1 = _mm_set1_ps(p1);
        __m128 mmp2 = _mm_set1_ps(p2);
        __m128 mmp3 = _mm_set1_ps(p3);
        __m128 mmq1 = _mm_set1_ps(q1);
        __m128 mmq2 = _mm_set1_ps(q2);

        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);

            __m128 bigmaskpo = _mm_cmpgt_ps(x_128, pos_big_check);
            __m128 bigmaskne = _mm_cmplt_ps(x_128, neg_big_check);
            __m128 tinymask = _mm_and_ps(
                _mm_cmpgt_ps(x_128, neg_small_check),
                _mm_cmplt_ps(x_128, pos_small_check));
            __m128 bigmask = _mm_or_ps(bigmaskpo, bigmaskne);
            __m128 midmask = _mm_andnot_ps(_mm_or_ps(bigmask, tinymask), allones);

            __m128 y = x_128;
            y = _mm_blendv_ps(y, negative1, bigmaskne);
            y = _mm_blendv_ps(y, positive1, bigmaskpo);

            __m128 g = _mm_and_ps(y, midmask);
            __m128 g2 = _mm_mul_ps(g, g);

            __m128 ratio;
#ifdef __FMA__
            __m128 num = _mm_fmadd_ps(mmp3, g2, mmp2);
            num = _mm_fmadd_ps(num, g2, mmp1);

            __m128 den = _mm_add_ps(g2, mmq2);
            den = _mm_fmadd_ps(den, g2, mmq1);
            den = _mm_fmadd_ps(den, g2, positive1);

            ratio = _mm_mul_ps(g2, _mm_div_ps(num, den));
            __m128 poly = _mm_fmadd_ps(g, ratio, g);
#else
            __m128 num = _mm_add_ps(_mm_mul_ps(mmp3, g2), mmp2);
            num = _mm_add_ps(_mm_mul_ps(num, g2), mmp1);

            __m128 den = _mm_add_ps(g2, mmq2);
            den = _mm_add_ps(_mm_mul_ps(den, g2), mmq1);
            den = _mm_add_ps(_mm_mul_ps(den, g2), positive1);

            ratio = _mm_mul_ps(g2, _mm_div_ps(num, den));
            __m128 poly = _mm_add_ps(_mm_mul_ps(g, ratio), g);
#endif

            y = _mm_blendv_ps(y, poly, midmask);
            _mm_store_ps(x + i, y);
        }
#endif

        for (; i < n; i++)
        {
            float v = x[i];
            if (v < -TANH_BIGLIMIT)
                x[i] = -1.0f;
            else if (v > TANH_BIGLIMIT)
                x[i] = 1.0f;
            else if (v < -TANH_TINYLIMIT || v > TANH_TINYLIMIT)
            {
                float g = v * v;
                x[i] = v + v * g * (((p3 * g + p2) * g + p1) / (((g + q2) * g + q1) * g + 1.0f));
            }
        }
    }

    inline void dTanh(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

        tanh(x, n);
        size_t i = 0;
        [[maybe_unused]] size_t simd_end;

#if defined(__AVX512F__)
        __m512 n_ones = _mm512_set1_ps(-1.0f);
        simd_end = n - (n % 16);
        for (; i < simd_end; i += 16)
        {
            __m512 x_512 = _mm512_load_ps(x + i);
            __m512 res = _mm512_fmadd_ps(x_512, x_512, n_ones);
            _mm512_store_ps(x + i, res);
        }
#elif defined(__AVX2__)
        __m256 n_ones = _mm256_set1_ps(-1.0f);
        simd_end = n - (n % 8);
        for (; i < simd_end; i += 8)
        {
            __m256 x_256 = _mm256_load_ps(x + i);
#ifdef __FMA__
            __m256 res = _mm256_fmadd_ps(x_256, x_256, n_ones);
#else
            __m256 res = _mm256_add_ps(_mm256_mul_ps(x_256, x_256), n_ones);
#endif
            _mm256_store_ps(x + i, res);
        }
#elif defined(__SSE4_1__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 n_ones = _mm_set1_ps(-1.0f);
        simd_end = n - (n % 4);
        for (; i < simd_end; i += 4)
        {
            __m128 x_128 = _mm_load_ps(x + i);
#ifdef __FMA__
            __m128 res = _mm_fmadd_ps(x_128, x_128, n_ones);
#else
            __m128 res = _mm_mul_ps(x_128, x_128);
            res = _mm_add_ps(res, n_ones);
#endif
            _mm_store_ps(x + i, res);
        }
#endif
        for (; i < n; i++)
        {
            x[i] = 1 - x[i] * x[i];
        }
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

    inline void dSigmoid(float *RESTRICT x, const size_t n) noexcept
    {
        assert(n != 0 && "n must not be 0.");
        assert(x != nullptr && "x must not be null.");

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
}