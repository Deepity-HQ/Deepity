#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/Optimize.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <sleef.h>
#include <cstring>

#define ALIGN64(n) (((n) + 63) & ~63)

namespace Deep
{
    DiscriminativePCLayer::DiscriminativePCLayer(int size, int nextSize, int batchSize,
                                                 float learningRate, float inferenceRate, float precisionRate, float lmbda,
                                                 void (*act)(float *, size_t),
                                                 void (*dAct)(float *, size_t, bool))
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), pr(precisionRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct), activationType(ActivationType::RELU), opt(OptimizerType::SGD)
    {
        this->size = size;
        this->nextSize = nextSize;
        DynamicThread(batchSize);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    DiscriminativePCLayer::DiscriminativePCLayer(int size, int nextSize, int batchSize,
                                                 float learningRate, float inferenceRate, float precisionRate, float lmbda,
                                                 ActivationType aType, ActivationType dType)
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), pr(precisionRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activationType(aType), opt(OptimizerType::SGD)
    {
        this->activation = To_Fn(aType);
        this->activationDerivative = To_dFn(dType);
        this->size = size;
        this->nextSize = nextSize;
        DynamicThread(batchSize);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    void DiscriminativePCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = (size_t)size * nextSize;
        float limit = std::sqrt(2.0f / (size + nextSize));

        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(seedGenerator);

#pragma omp parallel
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
            std::normal_distribution<float> dist(0.0f, limit);

#pragma omp for
            for (ptrdiff_t i = 0; i < (ptrdiff_t)Wsz; ++i)
                W[i] = dist(rng);
        }
    }

    float DiscriminativePCLayer::CalculateState() noexcept
    {
        const size_t N = (size_t)batchSize * size;

        if (layerBelow == nullptr)
        {
            std::memset(e, 0, N * sizeof(float));
            float totalEnergy = 0.0f;
            for (size_t i=0; i < size; ++i)
                totalEnergy -= 0.5f * log_p[i] * batchSize;
            
            if (nextSize > 0) ComputeMuOnly();
            return totalEnergy;
        }

        cblas_scopy(N, z, 1, e, 1);
        cblas_saxpy(N, -1.0f, layerBelow->mu, 1, e, 1);

        float totalEnergy = 0.0f;
#pragma omp parallel for schedule(static) reduction(+ : totalEnergy)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            const size_t offset = (size_t)batch * size;
            size_t i = 0;

#if defined(__AVX512F__)
            __m512 epsFloor = _mm512_set1_ps(1e-8f);
            __m512 half = _mm512_set1_ps(0.5f);
            __m512 energy = _mm512_setzero_ps();

            size_t r = size % 16;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 16)
            {
                __m512 p512 = _mm512_load_ps(&p[i]);
                __m512 e512 = _mm512_loadu_ps(&e[offset + i]);
                __m512 precision = _mm512_max_ps(p512, epsFloor);
                __m512 logp512 = _mm512_loadu_ps(&log_p[i]);

                __m512 m1 = _mm512_mul_ps(
                    half,
                    _mm512_mul_ps(precision,
                                  _mm512_mul_ps(e512, e512)));
                energy = _mm512_fnmadd_ps(half, logp512, energy);
                energy = _mm512_add_ps(energy, m1);
            }
            totalEnergy += _mm512_reduce_add_ps(energy);

#elif defined(__AVX2__) || defined(__AVX__)
            __m256 epsFloor = _mm256_set1_ps(1e-8f);
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 energy = _mm256_setzero_ps();

            size_t r = size % 8;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 8)
            {
                __m256 p256 = _mm256_load_ps(&p[i]);
                __m256 e256 = _mm256_loadu_ps(&e[offset + i]);
                __m256 precision = _mm256_max_ps(p256, epsFloor);
                __m256 logp256 = _mm256_loadu_ps(&log_p[i]);

                __m256 m1 = _mm256_mul_ps(
                    half,
                    _mm256_mul_ps(precision,
                                  _mm256_mul_ps(e256, e256)));
                energy = _mm256_fnmadd_ps(half, logp256, energy);
                energy = _mm256_add_ps(energy, m1);
            }
            totalEnergy += hsum256_ps(energy);

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
            __m128 epsFloor = _mm_set1_ps(1e-8f);
            __m128 half = _mm_set1_ps(0.5f);
            __m128 energy = _mm_setzero_ps();

            size_t r = size % 4;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 4)
            {
                __m128 p256 = _mm_load_ps(&p[i]);
                __m128 e256 = _mm_loadu_ps(&e[offset + i]);
                __m128 precision = _mm_max_ps(p256, epsFloor);
                __m128 logp128 = _mm_loadu_ps(&log_p[i]);

                __m128 m1 = _mm_mul_ps(
                    half,
                    _mm_mul_ps(precision,
                               _mm_mul_ps(e256, e256)));

#ifdef __FMA__
                energy = _mm_fnmadd_ps(half, logp128, energy);
#else
                __m128 m2 = _mm_mul_ps(half, logp128);
                energy = _mm_sub_ps(energy, m2);
#endif
                energy = _mm_add_ps(energy, m1);
            }
            totalEnergy += hsum128_ps(energy);

#endif
            for (; i < size; ++i)
            {
                float precision = std::max(p[i], 1e-8f);
                float err = e[offset + i];

                totalEnergy += 0.5f * precision * err * err;
                totalEnergy -= 0.5f * log_p[i];
            }
        }

        if (nextSize > 0)
            ComputeMuOnly();

        return totalEnergy;
    }

    void DiscriminativePCLayer::ComputeMuOnly() noexcept
    {
        if (nextSize == 0) return;

        size_t Nout = (size_t)batchSize * nextSize;
        size_t N = (size_t)batchSize * size;

        if (isClamped && muCacheValid) {
            cblas_scopy((int)Nout, cachedMu, 1, mu, 1);
            return;
        }

        cblas_scopy((int)N, z, 1, zF, 1);
        activation(zF, N);

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            batchSize, nextSize, size,
            1.0f, zF, size, W, size, 0.0f, mu, nextSize);

        #pragma omp parallel for schedule(static) if(batchSize > 4 && !omp_in_parallel())
        for (int batch = 0; batch < batchSize; ++batch)
            cblas_saxpy(nextSize, 1.0f, b, 1, mu + batch * nextSize, 1);

        if (isClamped) {
            cblas_scopy((int)Nout, mu, 1, cachedMu, 1);
            muCacheValid = true;
        }
    }

    void DiscriminativePCLayer::UpdateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

        if (isClamped)
            return;

        if (layerAbove != nullptr && nextSize > 0)
        {
            const float *e_above = layerAbove->GetErrors();
            const float *p_above = layerAbove->GetPrecisions();

            cblas_scopy((int)N, z, 1, zFDeriv, 1);
            activationDerivative(zFDeriv, N, false);

            #pragma omp parallel for schedule(static) if(batchSize > 4 && !omp_in_parallel())
            for (int batch = 0; batch < batchSize; ++batch) {
                size_t offset = (size_t)batch * nextSize;
                for (size_t f = 0; f < nextSize; ++f) {
                    bottom_up[offset + f] = e_above[offset + f] * p_above[f];
                }
            }

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, nextSize,
                1.0f, bottom_up, nextSize, W, size,
                0.0f, feedbackScratch, size);

            #pragma omp parallel for schedule(static) if(batchSize > 4 && !omp_in_parallel())
            for (int batch = 0; batch < batchSize; ++batch) {
                size_t offset = (size_t)batch * size;
                for (size_t i = 0; i < size; ++i) {
                    size_t idx = offset + i;
                    dz_dt[idx] = (feedbackScratch[idx] * zFDeriv[idx]) - (p[i] * e[idx]);
                    z[idx] += ir * dz_dt[idx];
                }
            }
        }
        else // Output Layer
        {
            #pragma omp parallel for schedule(static) if(batchSize > 4 && !omp_in_parallel())
            for (int batch = 0; batch < batchSize; ++batch) {
                size_t offset = (size_t)batch * size;
                for (size_t i = 0; i < size; ++i) {
                    size_t idx = offset + i;
                    dz_dt[idx] = -(p[i] * e[idx]);
                    z[idx] += ir * dz_dt[idx];
                }
            }
        }
    }

    void DiscriminativePCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || nextSize == 0)
            return;

        const float *e_above = layerAbove->GetErrors();
        const float *p_above = layerAbove->GetPrecisions();
        float *local_grad = bottom_up; 

        #pragma omp parallel for schedule(static) if(batchSize > 4 && !omp_in_parallel())
        for (int batch = 0; batch < batchSize; ++batch) {
            size_t offset = (size_t)batch * nextSize;
            for (size_t f = 0; f < nextSize; ++f) {
                local_grad[offset + f] = e_above[offset + f] * p_above[f];
            }
        }

        switch (opt)
        {
        case OptimizerType::SGD:
        {
            if (lmbda > 0.0f)
                cblas_sscal((size_t)nextSize * size, 1.0f - lmbda, W, 1);

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                nextSize, size, batchSize,
                lr / batchSize, local_grad, nextSize, zF, size,
                1.0f, W, size);

            float lr_batch = lr / batchSize;
            for (int batch = 0; batch < batchSize; batch++)
                cblas_saxpy(nextSize, lr_batch, local_grad + batch * nextSize, 1, b, 1);

            break;
        }
        case OptimizerType::ADAM:
        case OptimizerType::ADAMW:
        {
            t++;

            size_t num_weights = (size_t)nextSize * size;
            float grad_scale = -1.0f;

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                nextSize, size, batchSize,
                grad_scale, local_grad, nextSize, zF, size,
                0.0f, grad_W, size);

            std::memset(grad_b, 0, nextSize * sizeof(float));
            for (int batch = 0; batch < batchSize; batch++)
                cblas_saxpy(nextSize, grad_scale, local_grad + batch * nextSize, 1, grad_b, 1);

            if (opt == OptimizerType::ADAMW)
            {
                Deep::AdamWUpdate(W, grad_W, m_W, v_W, num_weights, t, lr, lmbda);
            }
            else
            {
                Deep::AdamUpdate(W, grad_W, m_W, v_W, num_weights, t, lr);
            }

            Deep::AdamUpdate(b, grad_b, m_b, v_b, nextSize, t, lr);
            break;
        }
        }
    }

    void DiscriminativePCLayer::UpdatePrecision() noexcept
    {
        if (layerBelow == nullptr)
            return;

        size_t simd_end = 0;

#if defined(__AVX512F__)
        __m512 neg_one = _mm512_set1_ps(-1.0f);
        __m512 half = _mm512_set1_ps(0.5f);
        float pr_inv_bs = pr / static_cast<float>(batchSize);
        __m512 pr_inv_bs512 = _mm512_set1_ps(pr_inv_bs);
        __m512 min_val = _mm512_set1_ps(-5.0f);
        __m512 max_val = _mm512_set1_ps(5.0f);

        size_t r = size % 16;
        simd_end = size - r;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < simd_end; i += 16)
        {
            __m512 grad = _mm512_setzero_ps();
            __m512 p512 = _mm512_load_ps(&p[i]);
            __m512 logp512 = _mm512_load_ps(&log_p[i]);

            for (int batch = 0; batch < batchSize; batch++)
            {
                __m512 e512 = _mm512_loadu_ps(&e[(size_t)batch * size + i]);

                __m512 err_sq = _mm512_mul_ps(e512, e512);
                __m512 p_err_sq_minus_1 = _mm512_fmadd_ps(p512, err_sq, neg_one);
                grad = _mm512_fmadd_ps(half, p_err_sq_minus_1, grad);
            }

            logp512 = _mm512_fnmadd_ps(pr_inv_bs512, grad, logp512);
            logp512 = _mm512_max_ps(min_val, _mm512_min_ps(logp512, max_val));
            p512 = Sleef_expf16_u10avx512f(logp512);

            _mm512_store_ps(&p[i], p512);
            _mm512_store_ps(&log_p[i], logp512);
        }
#elif defined(__AVX2__) || defined(__AVX__)
        __m256 neg_one = _mm256_set1_ps(-1.0f);
        __m256 half = _mm256_set1_ps(0.5f);
        float pr_inv_bs = pr / static_cast<float>(batchSize);
        __m256 pr_inv_bs256 = _mm256_set1_ps(pr_inv_bs);
        __m256 min_val = _mm256_set1_ps(-5.0f);
        __m256 max_val = _mm256_set1_ps(5.0f);

        size_t r = size % 8;
        simd_end = size - r;
#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)simd_end; i += 8)
        {
            __m256 grad = _mm256_setzero_ps();
            __m256 p256 = _mm256_load_ps(&p[i]);
            __m256 logp256 = _mm256_load_ps(&log_p[i]);

            for (int batch = 0; batch < batchSize; batch++)
            {
                __m256 e256 = _mm256_loadu_ps(&e[(size_t)batch * size + i]);

                __m256 err_sq = _mm256_mul_ps(e256, e256);
                __m256 p_err_sq_minus_1 = _mm256_fmadd_ps(p256, err_sq, neg_one);
                grad = _mm256_fmadd_ps(half, p_err_sq_minus_1, grad);
            }

            logp256 = _mm256_fnmadd_ps(pr_inv_bs256, grad, logp256);
            logp256 = _mm256_max_ps(min_val, _mm256_min_ps(logp256, max_val));
            p256 = Sleef_expf8_u10avx2(logp256);

            _mm256_store_ps(&p[i], p256);
            _mm256_store_ps(&log_p[i], logp256);
        }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
        __m128 neg_one = _mm_set1_ps(-1.0f);
        __m128 half = _mm_set1_ps(0.5f);
        float pr_inv_bs = pr / static_cast<float>(batchSize);
        __m128 pr_inv_bs128 = _mm_set1_ps(pr_inv_bs);
        __m128 min_val = _mm_set1_ps(-5.0f);
        __m128 max_val = _mm_set1_ps(5.0f);

        size_t r = size % 4;
        simd_end = size - r;
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < simd_end; i += 4)
        {
            __m128 grad = _mm_setzero_ps();
            __m128 p128 = _mm_load_ps(&p[i]);
            __m128 logp128 = _mm_load_ps(&log_p[i]);

            for (int batch = 0; batch < batchSize; batch++)
            {
                __m128 e128 = _mm_loadu_ps(&e[(size_t)batch * size + i]);

                __m128 err_sq = _mm_mul_ps(e128, e128);
#ifdef __FMA__
                __m128 p_err_sq_minus_1 = _mm_fmadd_ps(p128, err_sq, neg_one);
                grad = _mm_fmadd_ps(half, p_err_sq_minus_1, grad);
#else
                __m128 p_err_sq_minus_1 = _mm_add_ps(_mm_mul_ps(p128, err_sq), neg_one);
                grad = _mm_add_ps(_mm_mul_ps(half, p_err_sq_minus_1), grad);
#endif
            }

#ifdef __FMA__
            logp128 = _mm_fnmadd_ps(pr_inv_bs128, grad, logp128);
#else
            logp128 = _mm_sub_ps(logp128, _mm_mul_ps(pr_inv_bs128, grad));
#endif
            logp128 = _mm_max_ps(min_val, _mm_min_ps(logp128, max_val));
            p128 = Sleef_expf4_u10(logp128);

            _mm_store_ps(&p[i], p128);
            _mm_store_ps(&log_p[i], logp128);
        }
#endif

        for (size_t i = simd_end; i < size; i++)
        {
            float grad = 0.0f;
            for (int batch = 0; batch < batchSize; ++batch)
            {
                float err = e[(size_t)batch * size + i];
                grad += 0.5f * (p[i] * err * err - 1.0f);
            }

            grad /= batchSize;
            log_p[i] -= pr * grad;
            log_p[i] = std::max(-5.0f, std::min(log_p[i], 5.0f));
            p[i] = Sleef_expf_u10(log_p[i]);
        }
    }

    void DiscriminativePCLayer::ResetState() noexcept
    {
        size_t N = (size_t)batchSize * size;
        std::memset(z, 0, N * sizeof(float));
    }

    void DiscriminativePCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t copySize = std::min(inputData.size(), (size_t)(batchSize * size)) * sizeof(float);
        memcpy(z, inputData.data(), copySize);
        isClamped = true;
        muCacheValid = false;
    }

    void DiscriminativePCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    void DiscriminativePCLayer::ResyncLogPrecision() noexcept
    {
        for (size_t i = 0; i < size; ++i)
            log_p[i] = std::log(std::max(p[i], 1e-8f));
    }

    size_t DiscriminativePCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n) { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t own_state_size = (size_t)batchSize * size;

        total += pad16(own_state_size) * 3; // z, e, dz_dt
        total += pad16(own_state_size) * 3; // zF, zFDeriv, feedbackScratch 
        total += pad16(size) * 2;           // p, log_p

        if (nextSize > 0)
        {
            size_t out_state_size = (size_t)batchSize * nextSize;
            size_t w_size = (size_t)size * nextSize;
            
            total += pad16(w_size);                  // W
            total += pad16(nextSize);                // b
            total += pad16(out_state_size) * 3;      // mu, bottom_up, cachedMu 

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(w_size) * 3;
                total += pad16(nextSize) * 3;
            }
        }
        return total;
    }

    void DiscriminativePCLayer::BindMemory(MemoryArena &arena)
    {
        size_t own_state_size = (size_t)batchSize * size;
        size_t out_state_size = (size_t)batchSize * nextSize;

        z = arena.AllocateFloats(own_state_size);
        e = arena.AllocateFloats(own_state_size);
        dz_dt = arena.AllocateFloats(own_state_size);

        zF = arena.AllocateFloats(own_state_size);
        zFDeriv = arena.AllocateFloats(own_state_size);
        feedbackScratch = arena.AllocateFloats(own_state_size);

        p = arena.AllocateFloats(size);
        log_p = arena.AllocateFloats(size);

        std::memset(z, 0, own_state_size * sizeof(float));
        std::memset(e, 0, own_state_size * sizeof(float));
        std::memset(dz_dt, 0, own_state_size * sizeof(float));

        std::memset(zF, 0, own_state_size * sizeof(float));
        std::memset(zFDeriv, 0, own_state_size * sizeof(float));
        std::memset(feedbackScratch, 0, own_state_size * sizeof(float));

        std::fill_n(p, size, 1.0f);
        std::fill_n(log_p, size, 0.0f);

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;
            W = arena.AllocateFloats(w_size);
            b = arena.AllocateFloats(nextSize);
            mu = arena.AllocateFloats(out_state_size);
            bottom_up = arena.AllocateFloats(out_state_size);
            cachedMu = arena.AllocateFloats(out_state_size);

            std::memset(b, 0, nextSize * sizeof(float));
            std::memset(mu, 0, out_state_size * sizeof(float));
            std::memset(bottom_up, 0, out_state_size * sizeof(float));
            std::memset(cachedMu, 0, out_state_size * sizeof(float));

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                grad_W = arena.AllocateFloats(w_size);
                m_W = arena.AllocateFloats(w_size);
                v_W = arena.AllocateFloats(w_size);

                grad_b = arena.AllocateFloats(nextSize);
                m_b = arena.AllocateFloats(nextSize);
                v_b = arena.AllocateFloats(nextSize);

                std::memset(m_W, 0, w_size * sizeof(float));
                std::memset(v_W, 0, w_size * sizeof(float));
                std::memset(m_b, 0, nextSize * sizeof(float));
                std::memset(v_b, 0, nextSize * sizeof(float));

                std::memset(grad_W, 0, w_size * sizeof(float));
                std::memset(grad_b, 0, nextSize * sizeof(float));
            }
        }

        if (localArena && localArena.get() != &arena)
        {
            localArena.reset();
        }
    }
}
