#pragma once
#include <cstddef>
#include <cmath>
#include <immintrin.h>
#include <sleef.h>
#include <cstring>
#if defined(_MSC_VER)
#define RESTRICT __restrict
#else
#define RESTRICT __restrict__
#endif

/**
 * @file AdamOptimizer.h
 * @brief Vectorized AdamW update, operating per-element over an array of
 * weights (or biases) -- NOT the single-scalar-averaging design from the
 * earlier draft, which would have collapsed many distinct per-weight
 * gradients into one shared update and thrown away exactly the
 * per-parameter information Adam depends on.
 *
 * IMPORTANT -- this requires the RAW, unscaled gradient, not the
 * already-lr-scaled delta the existing UpdateWeights() GEMM calls produce.
 * The GEMM's alpha parameter needs to change from `lr/batchSize` to `1.0`,
 * writing into a scratch gradient buffer, with THIS function handling all
 * scaling (lr, momentum, bias correction) afterward. Adam's moment
 * estimates (m, v) are only meaningful if they're tracking the raw
 * gradient's own magnitude consistently across timesteps -- baking lr
 * scaling into the gradient before Adam sees it would corrupt that.
 *
 * lr is a genuine runtime parameter, not a compile-time constant -- this
 * session has repeatedly found lr needs to be tunable per-layer and per
 * schedule (base rate, decay, and a found-empirically-necessary correction
 * factor for at least one training regime); hardcoding it would silently
 * break all of that.
 */

namespace Deep
{
    enum class OptimizerType {
        SGD,
        ADAM,
        ADAMW
    };
    inline OptimizerType toOpType(const char *s) {
        if (strncmp(s, "SGD", 3) == 0) return OptimizerType::SGD;
        else if (strncmp(s, "ADAM", 4) == 0) return OptimizerType::ADAM;
        else if (strncmp(s, "ADAMW", 5) == 0) return OptimizerType::ADAMW;
        else return OptimizerType::SGD;
    }

    /// @brief Vectorized AdamW update over `n` elements. `w`, `grad`, `m`,
    /// `v` are all arrays of length `n` -- ONE m/v pair PER WEIGHT, not one
    /// shared scalar. `t` is the layer's own timestep counter (increment
    /// ONCE per UpdateWeights() call, not per element -- shared across the
    /// whole array, which is why beta1^t/beta2^t are computed once here,
    /// not per element).
    ///
    /// AdamW's decoupled weight decay (`w -= lr*lambda*w`, applied
    /// separately from the gradient-based step) replaces this codebase's
    /// existing `cblas_sscal(..., 1.0f-lmbda, W, 1)` weight-decay call --
    /// don't apply both, or decay will be applied twice.
    inline void AdamWUpdate(float *RESTRICT w, const float *RESTRICT grad,
                            float *RESTRICT m, float *RESTRICT v,
                            size_t n, int t,
                            float lr, float lambda,
                            float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept
    {
        // Bias-correction terms depend only on t, beta1, beta2 -- identical
        // for every element in this call. Computed ONCE (scalar), not
        // per-element (would otherwise mean n redundant Sleef_powf calls
        // for a value that never changes within this call).
        const float beta1_t = 1.0f - Sleef_powf_u10(beta1, (float)t);
        const float beta2_t = 1.0f - Sleef_powf_u10(beta2, (float)t);
        const float inv_beta1_t = 1.0f / beta1_t;
        const float inv_beta2_t = 1.0f / beta2_t;

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 beta1_v = _mm512_set1_ps(beta1);
        __m512 one_minus_beta1 = _mm512_set1_ps(1.0f - beta1);
        __m512 beta2_v = _mm512_set1_ps(beta2);
        __m512 one_minus_beta2 = _mm512_set1_ps(1.0f - beta2);
        __m512 inv_beta1_t_v = _mm512_set1_ps(inv_beta1_t);
        __m512 inv_beta2_t_v = _mm512_set1_ps(inv_beta2_t);
        __m512 lr_v = _mm512_set1_ps(lr);
        __m512 eps_v = _mm512_set1_ps(eps);
        __m512 decay_v = _mm512_set1_ps(1.0f - lr * lambda); // AdamW decoupled decay factor

        size_t r = n % 16;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 16)
        {
            __m512 g = _mm512_loadu_ps(&grad[i]);
            __m512 m_old = _mm512_loadu_ps(&m[i]);
            __m512 v_old = _mm512_loadu_ps(&v[i]);

            __m512 m_new = _mm512_fmadd_ps(beta1_v, m_old, _mm512_mul_ps(one_minus_beta1, g));
            __m512 v_new = _mm512_fmadd_ps(beta2_v, v_old, _mm512_mul_ps(one_minus_beta2, _mm512_mul_ps(g, g)));

            __m512 m_hat = _mm512_mul_ps(m_new, inv_beta1_t_v);
            __m512 v_hat = _mm512_mul_ps(v_new, inv_beta2_t_v);

            __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps_v);
            __m512 step = _mm512_div_ps(_mm512_mul_ps(lr_v, m_hat), denom);

            __m512 w_old = _mm512_loadu_ps(&w[i]);
            // AdamW: decoupled decay applied to w directly, THEN the
            // gradient-based step -- not folded into the gradient itself.
            __m512 w_new = _mm512_sub_ps(_mm512_mul_ps(w_old, decay_v), step);

            _mm512_storeu_ps(&w[i], w_new);
            _mm512_storeu_ps(&m[i], m_new);
            _mm512_storeu_ps(&v[i], v_new);
        }
#elif defined(__AVX2__) || defined(__AVX__)
        __m256 beta1_v = _mm256_set1_ps(beta1);
        __m256 one_minus_beta1 = _mm256_set1_ps(1.0f - beta1);
        __m256 beta2_v = _mm256_set1_ps(beta2);
        __m256 one_minus_beta2 = _mm256_set1_ps(1.0f - beta2);
        __m256 inv_beta1_t_v = _mm256_set1_ps(inv_beta1_t);
        __m256 inv_beta2_t_v = _mm256_set1_ps(inv_beta2_t);
        __m256 lr_v = _mm256_set1_ps(lr);
        __m256 eps_v = _mm256_set1_ps(eps);
        __m256 decay_v = _mm256_set1_ps(1.0f - lr * lambda);

        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 g = _mm256_loadu_ps(&grad[i]);
            __m256 m_old = _mm256_loadu_ps(&m[i]);
            __m256 v_old = _mm256_loadu_ps(&v[i]);

#ifdef __FMA__
            __m256 m_new = _mm256_fmadd_ps(beta1_v, m_old, _mm256_mul_ps(one_minus_beta1, g));
            __m256 v_new = _mm256_fmadd_ps(beta2_v, v_old, _mm256_mul_ps(one_minus_beta2, _mm256_mul_ps(g, g)));
#else
            __m256 m_new = _mm256_add_ps(_mm256_mul_ps(beta1_v, m_old), _mm256_mul_ps(one_minus_beta1, g));
            __m256 v_new = _mm256_add_ps(_mm256_mul_ps(beta2_v, v_old), _mm256_mul_ps(one_minus_beta2, _mm256_mul_ps(g, g)));
#endif

            __m256 m_hat = _mm256_mul_ps(m_new, inv_beta1_t_v);
            __m256 v_hat = _mm256_mul_ps(v_new, inv_beta2_t_v);

            __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v);
            __m256 step = _mm256_div_ps(_mm256_mul_ps(lr_v, m_hat), denom);

            __m256 w_old = _mm256_loadu_ps(&w[i]);
            __m256 w_new = _mm256_sub_ps(_mm256_mul_ps(w_old, decay_v), step);

            _mm256_storeu_ps(&w[i], w_new);
            _mm256_storeu_ps(&m[i], m_new);
            _mm256_storeu_ps(&v[i], v_new);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 beta1_v = _mm_set1_ps(beta1);
        __m128 one_minus_beta1 = _mm_set1_ps(1.0f - beta1);
        __m128 beta2_v = _mm_set1_ps(beta2);
        __m128 one_minus_beta2 = _mm_set1_ps(1.0f - beta2);
        __m128 inv_beta1_t_v = _mm_set1_ps(inv_beta1_t);
        __m128 inv_beta2_t_v = _mm_set1_ps(inv_beta2_t);
        __m128 lr_v = _mm_set1_ps(lr);
        __m128 eps_v = _mm_set1_ps(eps);
        __m128 decay_v = _mm_set1_ps(1.0f - lr * lambda);

        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 g = _mm_loadu_ps(&grad[i]);
            __m128 m_old = _mm_loadu_ps(&m[i]);
            __m128 v_old = _mm_loadu_ps(&v[i]);

#ifdef __FMA__
            __m128 m_new = _mm_fmadd_ps(beta1_v, m_old, _mm_mul_ps(one_minus_beta1, g));
            __m128 v_new = _mm_fmadd_ps(beta2_v, v_old, _mm_mul_ps(one_minus_beta2, _mm_mul_ps(g, g)));
#else
            __m128 m_new = _mm_add_ps(_mm_mul_ps(beta1_v, m_old), _mm_mul_ps(one_minus_beta1, g));
            __m128 v_new = _mm_add_ps(_mm_mul_ps(beta2_v, v_old), _mm_mul_ps(one_minus_beta2, _mm_mul_ps(g, g)));
#endif

            __m128 m_hat = _mm_mul_ps(m_new, inv_beta1_t_v);
            __m128 v_hat = _mm_mul_ps(v_new, inv_beta2_t_v);

            __m128 denom = _mm_add_ps(_mm_sqrt_ps(v_hat), eps_v);
            __m128 step = _mm_div_ps(_mm_mul_ps(lr_v, m_hat), denom);

            __m128 w_old = _mm_loadu_ps(&w[i]);
            __m128 w_new = _mm_sub_ps(_mm_mul_ps(w_old, decay_v), step);

            _mm_storeu_ps(&w[i], w_new);
            _mm_storeu_ps(&m[i], m_new);
            _mm_storeu_ps(&v[i], v_new);
        }
#endif
        // Scalar tail
        float decay = 1.0f - lr * lambda;
        for (; i < n; ++i)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

            float m_hat = m[i] * inv_beta1_t;
            float v_hat = v[i] * inv_beta2_t;

            w[i] = w[i] * decay - lr * m_hat / (Sleef_sqrtf_u05(v_hat) + eps);
        }
    }

    /// @brief Vectorized PLAIN Adam update over `n` elements -- NO weight
    /// decay term at all, unlike AdamWUpdate above. This is deliberate,
    /// not a missing feature: "Adam with L2 regularization" (folding decay
    /// into the gradient before the moment estimates see it) is the
    /// well-known WORSE approach AdamW was specifically invented to fix --
    /// large-gradient weights get decayed less than they should, since the
    /// adaptive step size shrinks the decay term along with everything
    /// else. If you want regularization alongside plain Adam, apply this
    /// codebase's existing decay convention separately (e.g.
    /// cblas_sscal(n, 1.0f-lmbda, w, 1) before calling this) -- fully
    /// composable, and avoids re-introducing the exact problem AdamW
    /// exists to solve.
    ///
    /// Same per-element m/v semantics as AdamWUpdate: one m/v pair PER
    /// WEIGHT, t is the layer's own shared timestep counter (increment
    /// ONCE per UpdateWeights() call, not per element).
    inline void AdamUpdate(float *RESTRICT w, const float *RESTRICT grad,
                           float *RESTRICT m, float *RESTRICT v,
                           size_t n, int t,
                           float lr,
                           float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept
    {
        const float beta1_t = 1.0f - Sleef_powf_u10(beta1, (float)t);
        const float beta2_t = 1.0f - Sleef_powf_u10(beta2, (float)t);
        const float inv_beta1_t = 1.0f / beta1_t;
        const float inv_beta2_t = 1.0f / beta2_t;

        size_t i = 0;

#if defined(__AVX512F__)
        __m512 beta1_v = _mm512_set1_ps(beta1);
        __m512 one_minus_beta1 = _mm512_set1_ps(1.0f - beta1);
        __m512 beta2_v = _mm512_set1_ps(beta2);
        __m512 one_minus_beta2 = _mm512_set1_ps(1.0f - beta2);
        __m512 inv_beta1_t_v = _mm512_set1_ps(inv_beta1_t);
        __m512 inv_beta2_t_v = _mm512_set1_ps(inv_beta2_t);
        __m512 lr_v = _mm512_set1_ps(lr);
        __m512 eps_v = _mm512_set1_ps(eps);

        size_t r = n % 16;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 16)
        {
            __m512 g = _mm512_loadu_ps(&grad[i]);
            __m512 m_old = _mm512_loadu_ps(&m[i]);
            __m512 v_old = _mm512_loadu_ps(&v[i]);

            __m512 m_new = _mm512_fmadd_ps(beta1_v, m_old, _mm512_mul_ps(one_minus_beta1, g));
            __m512 v_new = _mm512_fmadd_ps(beta2_v, v_old, _mm512_mul_ps(one_minus_beta2, _mm512_mul_ps(g, g)));

            __m512 m_hat = _mm512_mul_ps(m_new, inv_beta1_t_v);
            __m512 v_hat = _mm512_mul_ps(v_new, inv_beta2_t_v);

            __m512 denom = _mm512_add_ps(_mm512_sqrt_ps(v_hat), eps_v);
            __m512 step = _mm512_div_ps(_mm512_mul_ps(lr_v, m_hat), denom);

            __m512 w_old = _mm512_loadu_ps(&w[i]);
            __m512 w_new = _mm512_sub_ps(w_old, step); // no decay term -- plain Adam

            _mm512_storeu_ps(&w[i], w_new);
            _mm512_storeu_ps(&m[i], m_new);
            _mm512_storeu_ps(&v[i], v_new);
        }
#elif defined(__AVX2__) || defined(__AVX__)
        __m256 beta1_v = _mm256_set1_ps(beta1);
        __m256 one_minus_beta1 = _mm256_set1_ps(1.0f - beta1);
        __m256 beta2_v = _mm256_set1_ps(beta2);
        __m256 one_minus_beta2 = _mm256_set1_ps(1.0f - beta2);
        __m256 inv_beta1_t_v = _mm256_set1_ps(inv_beta1_t);
        __m256 inv_beta2_t_v = _mm256_set1_ps(inv_beta2_t);
        __m256 lr_v = _mm256_set1_ps(lr);
        __m256 eps_v = _mm256_set1_ps(eps);

        size_t r = n % 8;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 8)
        {
            __m256 g = _mm256_loadu_ps(&grad[i]);
            __m256 m_old = _mm256_loadu_ps(&m[i]);
            __m256 v_old = _mm256_loadu_ps(&v[i]);

#ifdef __FMA__
            __m256 m_new = _mm256_fmadd_ps(beta1_v, m_old, _mm256_mul_ps(one_minus_beta1, g));
            __m256 v_new = _mm256_fmadd_ps(beta2_v, v_old, _mm256_mul_ps(one_minus_beta2, _mm256_mul_ps(g, g)));
#else
            __m256 m_new = _mm256_add_ps(_mm256_mul_ps(beta1_v, m_old), _mm256_mul_ps(one_minus_beta1, g));
            __m256 v_new = _mm256_add_ps(_mm256_mul_ps(beta2_v, v_old), _mm256_mul_ps(one_minus_beta2, _mm256_mul_ps(g, g)));
#endif

            __m256 m_hat = _mm256_mul_ps(m_new, inv_beta1_t_v);
            __m256 v_hat = _mm256_mul_ps(v_new, inv_beta2_t_v);

            __m256 denom = _mm256_add_ps(_mm256_sqrt_ps(v_hat), eps_v);
            __m256 step = _mm256_div_ps(_mm256_mul_ps(lr_v, m_hat), denom);

            __m256 w_old = _mm256_loadu_ps(&w[i]);
            __m256 w_new = _mm256_sub_ps(w_old, step);

            _mm256_storeu_ps(&w[i], w_new);
            _mm256_storeu_ps(&m[i], m_new);
            _mm256_storeu_ps(&v[i], v_new);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 beta1_v = _mm_set1_ps(beta1);
        __m128 one_minus_beta1 = _mm_set1_ps(1.0f - beta1);
        __m128 beta2_v = _mm_set1_ps(beta2);
        __m128 one_minus_beta2 = _mm_set1_ps(1.0f - beta2);
        __m128 inv_beta1_t_v = _mm_set1_ps(inv_beta1_t);
        __m128 inv_beta2_t_v = _mm_set1_ps(inv_beta2_t);
        __m128 lr_v = _mm_set1_ps(lr);
        __m128 eps_v = _mm_set1_ps(eps);

        size_t r = n % 4;
        size_t simd_end = n - r;
        for (; i < simd_end; i += 4)
        {
            __m128 g = _mm_loadu_ps(&grad[i]);
            __m128 m_old = _mm_loadu_ps(&m[i]);
            __m128 v_old = _mm_loadu_ps(&v[i]);

#ifdef __FMA__
            __m128 m_new = _mm_fmadd_ps(beta1_v, m_old, _mm_mul_ps(one_minus_beta1, g));
            __m128 v_new = _mm_fmadd_ps(beta2_v, v_old, _mm_mul_ps(one_minus_beta2, _mm_mul_ps(g, g)));
#else
            __m128 m_new = _mm_add_ps(_mm_mul_ps(beta1_v, m_old), _mm_mul_ps(one_minus_beta1, g));
            __m128 v_new = _mm_add_ps(_mm_mul_ps(beta2_v, v_old), _mm_mul_ps(one_minus_beta2, _mm_mul_ps(g, g)));
#endif

            __m128 m_hat = _mm_mul_ps(m_new, inv_beta1_t_v);
            __m128 v_hat = _mm_mul_ps(v_new, inv_beta2_t_v);

            __m128 denom = _mm_add_ps(_mm_sqrt_ps(v_hat), eps_v);
            __m128 step = _mm_div_ps(_mm_mul_ps(lr_v, m_hat), denom);

            __m128 w_old = _mm_loadu_ps(&w[i]);
            __m128 w_new = _mm_sub_ps(w_old, step);

            _mm_storeu_ps(&w[i], w_new);
            _mm_storeu_ps(&m[i], m_new);
            _mm_storeu_ps(&v[i], v_new);
        }
#endif
        // Scalar tail
        for (; i < n; ++i)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;

            float m_hat = m[i] * inv_beta1_t;
            float v_hat = v[i] * inv_beta2_t;

            w[i] -= lr * m_hat / (Sleef_sqrtf_u05(v_hat) + eps);
        }
    }

}