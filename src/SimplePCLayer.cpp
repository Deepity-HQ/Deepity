#include <deepity/layers/SimplePCLayer.h>
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

namespace Deep
{
    SimplePCLayer::SimplePCLayer(size_t size, size_t nextSize, size_t batchSize,
                                 float learningRate, float inferenceRate, float lmbda,
                                 void (*act)(float *, size_t),
                                 void (*dAct)(float *, size_t, bool))
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct), activationType(To_AType(act)), opt(OptimizerType::SGD)
    {
        this->size = size;
        this->nextSize = nextSize;
        activationDerivativeInto = To_dFn2(To_AType(dAct));
        DynamicThread(batchSize);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    SimplePCLayer::SimplePCLayer(size_t size, size_t nextSize, size_t batchSize,
                                 float learningRate, float inferenceRate, float lmbda,
                                 ActivationType aType, ActivationType dType)
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activationType(aType)
    {
        this->activation = To_Fn(aType);
        this->activationDerivative = To_dFn(dType);
        this->activationDerivativeInto = To_dFn2(dType);
        this->size = size;
        this->nextSize = nextSize;
        DynamicThread(batchSize);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    void SimplePCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = size * nextSize;
        float limit = std::sqrt(2.0f / (size + nextSize));

        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(seedGenerator);

#pragma omp parallel if (!omp_in_parallel())
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
            std::normal_distribution<float> dist(0.0f, limit);

#pragma omp for
            for (ptrdiff_t i = 0; i < (ptrdiff_t)Wsz; ++i)
                W[i] = dist(rng);
        }
    }

    float SimplePCLayer::CalculateState() noexcept
    {
        const size_t N = batchSize * size;

        if (layerBelow == nullptr)
        {
            std::memset(e, 0, N * sizeof(float));
            if (nextSize > 0)
            {
                ComputeMuOnly();
            }
            return 0.0f;
        }

        cblas_scopy(N, z, 1, e, 1);
        cblas_saxpy(N, -1.0f, layerBelow->mu, 1, e, 1);

        float totalEnergy = 0.0f;

#pragma omp parallel for schedule(static) reduction(+ : totalEnergy) if (batchSize > 4 && !omp_in_parallel())
        for (int batch = 0; batch < batchSize; ++batch)
        {
            const size_t offset = (size_t)batch * size;
            size_t i = 0;

#if defined(__AVX512F__)
            __m512 half = _mm512_set1_ps(0.5f);
            __m512 energy = _mm512_setzero_ps();
            size_t r = size % 16;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 16)
            {
                __m512 e512 = _mm512_loadu_ps(&e[offset + i]);
                energy = _mm512_fmadd_ps(half, _mm512_mul_ps(e512, e512), energy);
            }
            totalEnergy += _mm512_reduce_add_ps(energy);
#elif defined(__AVX2__) || defined(__AVX__)
            __m256 half = _mm256_set1_ps(0.5f);
            __m256 energy = _mm256_setzero_ps();
            size_t r = size % 8;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 8)
            {
                __m256 e256 = _mm256_loadu_ps(&e[offset + i]);
#ifdef __FMA__
                energy = _mm256_fmadd_ps(half, _mm256_mul_ps(e256, e256), energy);
#else
                energy = _mm256_add_ps(energy, _mm256_mul_ps(half, _mm256_mul_ps(e256, e256)));
#endif
            }
            totalEnergy += hsum256_ps(energy);
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
            __m128 half = _mm_set1_ps(0.5f);
            __m128 energy = _mm_setzero_ps();
            size_t r = size % 4;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 4)
            {
                __m128 e128 = _mm_loadu_ps(&e[offset + i]);
#ifdef __FMA__
                energy = _mm_fmadd_ps(half, _mm_mul_ps(e128, e128), energy);
#else
                energy = _mm_add_ps(energy, _mm_mul_ps(half, _mm_mul_ps(e128, e128)));
#endif
            }
            totalEnergy += hsum128_ps(energy);
#endif
            for (; i < size; ++i)
            {
                float err = e[offset + i];
                totalEnergy += 0.5f * err * err;
            }
        }
        if (nextSize > 0)
        {
            ComputeMuOnly();
        }

        return totalEnergy;
    }

    void SimplePCLayer::ComputeMuOnly() noexcept
    {
        if (nextSize == 0)
            return;

        size_t Nout = batchSize * nextSize;
        size_t N = batchSize * size;

        if (isClamped && muCacheValid)
        {
            cblas_scopy((int)Nout, cachedMu, 1, mu, 1);
            return;
        }

        cblas_scopy((int)N, z, 1, zF, 1);
        switch (activationType)
        {
        case ActivationType::RELU:
            Deep::relu(zF, N);
            break;
        case ActivationType::SIGMOID:
            Deep::sigmoid(zF, N);
            break;
        case ActivationType::eSIGMOID:
            Deep::e_sigmoid(zF, N);
            break;
        case ActivationType::TANH:
            Deep::tanh(zF, N);
            break;
        case ActivationType::LINEAR:
            Deep::linear(zF, N);
            break;
        default:
            activation(zF, N);
            break; // custom/unrecognized function pointer
        }

        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            batchSize, nextSize, size,
            1.0f, zF, size, W, size, 0.0f, mu, nextSize);

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
        for (int batch = 0; batch < batchSize; ++batch)
        {
            cblas_saxpy(nextSize, 1.0f, b, 1, mu + batch * nextSize, 1);
        }

        if (isClamped)
        {
            cblas_scopy((int)Nout, mu, 1, cachedMu, 1);
            muCacheValid = true;
        }
    }

    void SimplePCLayer::UpdateState() noexcept
    {
        size_t N = batchSize * size;

        if (isClamped)
            return;

        if (layerAbove != nullptr && nextSize > 0)
        {
            const float *e_above = layerAbove->GetErrors();

            switch (activationType)
            {
            case ActivationType::RELU:
                Deep::dReluInto(zFDeriv, z, N);
                break;
            case ActivationType::SIGMOID:
                Deep::dSigmoidInto(zFDeriv, z, N);
                break;
            case ActivationType::TANH:
                Deep::dTanhInto(zFDeriv, z, N);
                break;
            case ActivationType::LINEAR:
                Deep::dLinearInto(zFDeriv, z, N);
                break;
            default:
                activationDerivativeInto(zFDeriv, z, N);
                break; // custom/unrecognized, or eSIGMOID (no dedicated derivative variant exists)
            }

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, nextSize,
                1.0f, e_above, nextSize, W, size,
                0.0f, feedbackScratch, size);

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
            for (int batch = 0; batch < batchSize; ++batch)
            {
                float *RESTRICT zPtr = z;
                const float *RESTRICT feedbackPtr = feedbackScratch;
                const float *RESTRICT zFDerivPtr = zFDeriv;
                const float *RESTRICT ePtr = e;
                size_t offset = (size_t)batch * size;
                for (size_t i = 0; i < size; ++i)
                {
                    size_t idx = offset + i;
                    // Fused dz_dt directly into the z update to save memory writes
                    zPtr[idx] += ir * ((feedbackPtr[idx] * zFDerivPtr[idx]) - ePtr[idx]);
                }
            }
        }
        else // Output Layer
        {
#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
            for (int batch = 0; batch < batchSize; ++batch)
            {
                float *RESTRICT zPtr = z;
                const float *RESTRICT ePtr = e;
                size_t offset = (size_t)batch * size;
                for (size_t i = 0; i < size; ++i)
                {
                    size_t idx = offset + i;
                    zPtr[idx] += ir * -ePtr[idx];
                }
            }
        }
    }

    void SimplePCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || nextSize == 0)
            return;

        const float *local_grad = layerAbove->GetErrors();

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

    void SimplePCLayer::ResetState() noexcept
    {
        size_t N = (size_t)batchSize * size;
        std::memset(z, 0, N * sizeof(float));
    }

    void SimplePCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t copySize = (std::min)(inputData.size(), (size_t)(batchSize * size)) * sizeof(float);
        memcpy(z, inputData.data(), copySize);
        isClamped = true;
        muCacheValid = false;
    }

    void SimplePCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    size_t SimplePCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t own_state_size = (size_t)batchSize * size;

        // dz_dt removed: state size drops from 3 arrays to 2
        total += pad16(own_state_size) * 2;

        if (nextSize > 0)
        {
            size_t out_state_size = (size_t)batchSize * nextSize;
            size_t w_size = (size_t)size * nextSize;

            total += pad16(w_size);
            total += pad16(nextSize);
            total += pad16(out_state_size) * 2;
            // E, prevZ, and bottom_up removed
            total += pad16(own_state_size) * 3; // zF, zFDeriv, feedbackScratch

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(w_size) * 3;
                total += pad16(nextSize) * 3;
            }
        }

        return total;
    }

    void SimplePCLayer::BindMemory(MemoryArena &arena)
    {
        size_t own_state_size = (size_t)batchSize * size;
        size_t out_state_size = (size_t)batchSize * nextSize;

        z = arena.AllocateFloats(own_state_size);
        e = arena.AllocateFloats(own_state_size);

        std::memset(z, 0, own_state_size * sizeof(float));
        std::memset(e, 0, own_state_size * sizeof(float));

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;

            W = arena.AllocateFloats(w_size);
            b = arena.AllocateFloats(nextSize);
            mu = arena.AllocateFloats(out_state_size);
            cachedMu = arena.AllocateFloats(out_state_size);
            zF = arena.AllocateFloats(own_state_size);
            zFDeriv = arena.AllocateFloats(own_state_size);
            feedbackScratch = arena.AllocateFloats(own_state_size);

            std::memset(b, 0, nextSize * sizeof(float));
            std::memset(mu, 0, out_state_size * sizeof(float));
            std::memset(cachedMu, 0, out_state_size * sizeof(float));
            std::memset(zF, 0, own_state_size * sizeof(float));
            std::memset(zFDeriv, 0, own_state_size * sizeof(float));
            std::memset(feedbackScratch, 0, own_state_size * sizeof(float));

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
