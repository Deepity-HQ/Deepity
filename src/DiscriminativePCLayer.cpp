#include "DiscriminativePCLayer.h"
#include "Optimize.h"
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cblas.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <sleef.h>
#include <cstring>

#define ALIGN64(n) (((n) + 63) & ~63)

namespace Deep
{
    DiscriminativePCLayer::DiscriminativePCLayer(int size, int nextSize, int batchSize,
                                                 float learningRate, float inferenceRate, float lmbda,
                                                 void (*act)(float *, size_t),
                                                 void (*dAct)(float *, size_t, bool))
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct)
    {
        this->size = size;
        this->nextSize = nextSize;

        size_t allocOwn = (size_t)batchSize * size * sizeof(float);     // z, e, dz_dt
        size_t allocOut = (size_t)batchSize * nextSize * sizeof(float); // mu, scratch
        size_t allocBias = nextSize * sizeof(float);                    // b
        size_t allocPrecision = size * sizeof(float);                   // p

        z.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocOwn))));
        e.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocOwn))));
        p.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocPrecision))));
        log_p.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocPrecision))));
        dz_dt.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocOwn))));
        std::memset(z.get(), 0, ALIGN64(allocOwn));
        std::memset(e.get(), 0, ALIGN64(allocOwn));
        std::fill_n(p.get(), size, 1.0f);
        std::fill_n(log_p.get(), size, 0.0f);
        std::memset(dz_dt.get(), 0, ALIGN64(allocOwn));

        if (nextSize > 0)
        {
            W.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64((size_t)nextSize * size * sizeof(float)))));
            b.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocBias))));
            mu.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocOut))));
            bottom_up.reset(static_cast<float *>(std::aligned_alloc(64, ALIGN64(allocOut))));

            std::memset(b.get(), 0, ALIGN64(allocBias));
            std::memset(mu.get(), 0, ALIGN64(allocOut));
            std::memset(bottom_up.get(), 0, ALIGN64(allocOut));
            // W left uninitialized -- RandomizeWeights() must run first.
        }
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
            for (size_t i = 0; i < Wsz; ++i)
                W.get()[i] = dist(rng);
        }

        // Biases are kept at zero initially, so we don't randomize them here.
    }

    float DiscriminativePCLayer::CalculateState() noexcept
    {
        const size_t N = (size_t)batchSize * size;

        if (layerBelow == nullptr)
        {
            std::memset(e.get(), 0, N * sizeof(float));
        }
        else
        {
            cblas_scopy(N, z.get(), 1, e.get(), 1);
            cblas_saxpy(N, -1.0f, layerBelow->mu.get(), 1, e.get(), 1);
        }

        float totalEnergy = 0.0f;

        for (int batch = 0; batch < batchSize; ++batch)
        {
            const size_t offset = (size_t)batch * size;

            size_t i = 0;

#if defined(__AVX512F__)
            __m512 small = _mm512_set1_ps(1e-8f);
            __m512 half = _mm512_set1_ps(0.5f);
            __m512 energy = _mm512_setzero_ps();

            size_t r = size % 16;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 16)
            {
                __m512 p512 = _mm512_load_ps(&p[i]);
                __m512 e512 = _mm512_loadu_ps(&e[offset + i]);
                __m512 precision = _mm512_max_ps(p512, small);

                __m512 m1 = _mm512_mul_ps(
                    half,
                    _mm512_mul_ps(precision,
                                  _mm512_mul_ps(e512, e512)));
                energy = _mm512_fnmadd_ps(half, Sleef_logf16_u10avx512f(precision), energy);

                energy = _mm512_add_ps(energy, m1);
            }
            totalEnergy += _mm512_reduce_add_ps(energy);

#elif defined(__AVX2__) || defined(__AVX__)
            __m256 small = _mm256_set1_ps(1e-8f);
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 energy = _mm256_setzero_ps();

            size_t r = size % 8;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 8)
            {
                __m256 p256 = _mm256_load_ps(&p[i]);
                __m256 e256 = _mm256_loadu_ps(&e[offset + i]);
                __m256 precision = _mm256_max_ps(p256, small);

                __m256 m1 = _mm256_mul_ps(
                    half,
                    _mm256_mul_ps(precision,
                                  _mm256_mul_ps(e256, e256)));
                energy = _mm256_fnmadd_ps(half, Sleef_logf8_u10avx2(precision), energy);

                energy = _mm256_add_ps(energy, m1);
            }
            totalEnergy += hsum256_ps(energy);

#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
            __m128 small = _mm_set1_ps(1e-8f);
            __m128 half = _mm_set1_ps(0.5f);
            __m128 energy = _mm_setzero_ps();

            size_t r = size % 4;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 4)
            {
                __m128 p256 = _mm_load_ps(&p[i]);
                __m128 e256 = _mm_loadu_ps(&e[offset + i]);
                __m128 precision = _mm_max_ps(p256, small);

                __m128 m1 = _mm_mul_ps(
                    half,
                    _mm_mul_ps(precision,
                               _mm_mul_ps(e256, e256)));

#ifdef __FMA__

                energy = _mm_fnmadd_ps(half, Sleef_logf4_u10avx2128(precision), energy);
#else
                __m128 m2 = _mm_mul_ps(
                    half,
                    Sleef_logf4_u10avx2128(precision));
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
                totalEnergy -= 0.5f * std::log(precision);
            }
        }

        if (nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;

            cblas_sgemm(
                CblasRowMajor,
                CblasNoTrans,
                CblasTrans,
                batchSize,
                nextSize,
                size,
                1.0f,
                z.get(),
                size,
                W.get(),
                size,
                0.0f,
                mu.get(),
                nextSize);

            for (int batch = 0; batch < batchSize; ++batch)
                cblas_saxpy(nextSize, 1.0f,
                            b.get(), 1,
                            mu.get() + batch * nextSize, 1);

            activation(mu.get(), Nout);
        }

        return totalEnergy;
    }

    void DiscriminativePCLayer::UpdateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

        if (nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;
            activationDerivative(mu.get(), Nout, true); // raw activation -> f'(), in place
        }

        if (isClamped)
            return;

        for (int batch = 0; batch < batchSize; ++batch)
        {
            size_t offset = (size_t)batch * size;
            size_t i = 0;
#if defined(__AVX512F__)
            __m512 neg_one = _mm512_set1_ps(-1.0f);
            size_t r = size % 16;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 16)
            {
                __m512 p512 = _mm512_load_ps(&p[i]);
                __m512 e512 = _mm512_loadu_ps(&e[offset + i]);

                __m512 res = _mm512_mul_ps(neg_one, _mm512_mul_ps(p512, e512));
                _mm512_storeu_ps(&dz_dt[offset + i], res);
            }
#elif defined(__AVX2__) || defined(__AVX__)
            __m256 neg_one = _mm256_set1_ps(-1.0f);
            size_t r = size % 8;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 8)
            {
                __m256 p256 = _mm256_load_ps(&p[i]);
                __m256 e256 = _mm256_loadu_ps(&e[offset + i]);
                __m256 res = _mm256_mul_ps(neg_one, _mm256_mul_ps(p256, e256));
                _mm256_storeu_ps(&dz_dt[offset + i], res);
            }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
            __m128 neg_one = _mm_set1_ps(-1.0f);
            size_t r = size % 4;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 4)
            {
                __m128 p128 = _mm_load_ps(&p[i]);
                __m128 e128 = _mm_loadu_ps(&e[offset + i]);
                __m128 res = _mm_mul_ps(neg_one, _mm_mul_ps(p128, e128));
                _mm_storeu_ps(&dz_dt[offset + i], res);
            }
#endif
            for (; i < size; ++i)
                dz_dt[offset + i] = -p[i] * e[offset + i];
        }

        if (layerAbove != nullptr && nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;
            const float *e_above = layerAbove->GetErrors();

            size_t i = 0;
#if defined(__AVX512F__)
            size_t r = Nout % 16;
            size_t simd_end = Nout - r;
            for (; i < simd_end; i += 16)
            {
                __m512 e_above512 = _mm512_load_ps(e_above + i);
                __m512 mu512 = _mm512_load_ps(&mu[i]);
                _mm512_store_ps(
                    &bottom_up[i],
                    _mm512_mul_ps(e_above512, mu512));
            }
#elif defined(__AVX2__) || defined(__AVX__)
            size_t r = Nout % 8;
            size_t simd_end = Nout - r;
            for (; i < simd_end; i += 8)
            {
                __m256 e_above256 = _mm256_load_ps(e_above + i);
                __m256 mu256 = _mm256_load_ps(&mu[i]);
                _mm256_store_ps(
                    &bottom_up[i],
                    _mm256_mul_ps(e_above256, mu256));
            }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
            size_t r = Nout % 4;
            size_t simd_end = Nout - r;
            for (; i < simd_end; i += 4)
            {
                __m128 e_above128 = _mm_load_ps(e_above + i);
                __m128 mu128 = _mm_load_ps(&mu[i]);
                _mm_store_ps(
                    &bottom_up[i],
                    _mm_mul_ps(e_above128, mu128));
            }
#endif
            for (; i < Nout; ++i)
                bottom_up.get()[i] = e_above[i] * mu.get()[i];

            // dz_dt(batch,size) += local_grad(batch,nextSize) @ W(nextSize,size)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, nextSize,
                1.0f,
                bottom_up.get(), nextSize,
                W.get(), size,
                1.0f, dz_dt.get(), size); // beta=1: accumulate directly
        }

        cblas_saxpy(N, ir, dz_dt.get(), 1, z.get(), 1);
    }

    void DiscriminativePCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || nextSize == 0)
            return;

        size_t Nout = (size_t)batchSize * nextSize;
        const float *e_above = layerAbove->GetErrors();
        float *local_grad = bottom_up.get(); // scratch, same buffer as UpdateState's feedback term

#pragma omp simd
        for (size_t i = 0; i < Nout; i++)
            local_grad[i] = e_above[i] * mu.get()[i];

        // L2 weight decay: W *= (1 - lmbda)
        if (lmbda > 0.0f)
            cblas_sscal((size_t)nextSize * size, 1.0f - lmbda, W.get(), 1);

        // W(nextSize,size) += (lr/batch) * local_grad^T(nextSize,batch) @ z(batch,size)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            nextSize, size, batchSize,
            lr / batchSize,
            local_grad, nextSize,
            z.get(), size,
            1.0f, W.get(), size);

        float lr_batch = lr / batchSize;
        for (int batch = 0; batch < batchSize; batch++)
        {
            cblas_saxpy(nextSize, lr_batch, local_grad + batch * nextSize, 1, b.get(), 1);
        }
    }

    void DiscriminativePCLayer::UpdatePrecision() noexcept
    {
        for (int i = 0; i < size; ++i)
        {
            float grad = 0.0f;

            for (int batch = 0; batch < batchSize; ++batch)
            {
                float err = e[(size_t)batch * size + i];
                grad += 0.5f * (p[i] * err * err - 1.0f);
            }

            grad /= batchSize;

            log_p[i] -= pr * grad;
            p[i] = std::exp(log_p[i]);
        }
    }

    void DiscriminativePCLayer::ResetState() noexcept
    {
        size_t N = (size_t)batchSize * size;
        std::memset(z.get(), 0, N * sizeof(float));
    }

    void DiscriminativePCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        // Safely copy the entire batched array directly into memory in one shot
        size_t copySize = std::min(inputData.size(), (size_t)(batchSize * size)) * sizeof(float);
        memcpy(z.get(), inputData.data(), copySize);
        isClamped = true;
    }

    void DiscriminativePCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }
}