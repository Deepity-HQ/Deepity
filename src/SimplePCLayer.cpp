#include <deepity/layers/SimplePCLayer.h>
#include <deepity/Optimize.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cblas.h>
#include <omp.h>
#include <immintrin.h>
#include <algorithm>
#include <sleef.h>
#include <cstring>

namespace Deep
{
    SimplePCLayer::SimplePCLayer(int size, int nextSize, int batchSize,
                                 float learningRate, float inferenceRate, float lmbda,
                                 void (*act)(float *, size_t),
                                 void (*dAct)(float *, size_t, bool))
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct), activationType(ActivationType::RELU), opt(OptimizerType::SGD)
    {
        this->size = size;
        this->nextSize = nextSize;
        DynamicThread(batchSize);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    SimplePCLayer::SimplePCLayer(int size, int nextSize, int batchSize,
                                 float learningRate, float inferenceRate, float lmbda,
                                 ActivationType aType, ActivationType dType)
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activationType(aType)
    {
        this->activation = To_Fn(aType);
        this->activationDerivative = To_dFn(dType);
        this->size = size;
        this->nextSize = nextSize;
        DynamicThread(batchSize);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    void SimplePCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
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

    float SimplePCLayer::CalculateState() noexcept
    {
        const size_t N = (size_t)batchSize * size;

        if (layerBelow == nullptr)
        {
            std::memset(e, 0, N * sizeof(float));
        }
        else
        {
            cblas_scopy(N, z, 1, e, 1);
            cblas_saxpy(N, -1.0f, layerBelow->mu, 1, e, 1);
        }

        // No precision weighting: E = 0.5 * sum(e^2), plain SSE. No p
        // multiply, no -0.5*log(p) term -- both were exactly inert at
        // pr=0.0, which was the only value ever actually used.
        float totalEnergy = 0.0f;

#pragma omp parallel for schedule(static) reduction(+ : totalEnergy)
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
            size_t Nout = (size_t)batchSize * nextSize;
            size_t N = (size_t)batchSize * size;

            // Mu-caching: recompute only when z has moved enough SINCE THE
            // LAST RECOMPUTE (not just the immediately preceding step --
            // comparing against prevZ, updated only on real recomputes,
            // catches cumulative drift across several skipped steps that a
            // step-to-step comparison would miss).
            //
            // threshold=0 (or a CLAMPED layer, whose z never changes at
            // all) makes the ratio always exactly 0 -- reproducing the
            // original, EXACT, validated behavior. threshold>0 extends
            // this to UNCLAMPED layers too, as a genuine APPROXIMATION --
            // correctness there means "close enough for real training,"
            // not "bit-identical," and needs accuracy-impact validation,
            // not just a single-batch trajectory diff.
            //
            // mu itself is NOT a stable buffer between steps -- UpdateState()
            // mutates it in place (converts it to its own derivative, for
            // the feedback GEMM) -- so a skip must copy a PRESERVED value
            // back into mu, not just skip writing to mu.
            bool shouldRecompute = true;
            if (muCacheThreshold >= 0.0f && muCacheValid)
            {
                float zNorm = cblas_snrm2((int)N, prevZ, 1);
                // Reuse dz_dt as scratch -- safe: UpdateState() overwrites
                // it fresh every step and nothing reads it between here
                // and that call, avoiding a per-call heap allocation on
                // this hot path.
                cblas_scopy((int)N, z, 1, dz_dt, 1);
                cblas_saxpy((int)N, -1.0f, prevZ, 1, dz_dt, 1);
                float deltaNorm = cblas_snrm2((int)N, dz_dt, 1);
                float ratio = deltaNorm / (zNorm + 1e-8f);
                shouldRecompute = ratio > muCacheThreshold;
            }

            if (shouldRecompute)
            {
                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasTrans,
                    batchSize, nextSize, size,
                    1.0f, z, size, W, size, 0.0f, mu, nextSize);

                for (int batch = 0; batch < batchSize; ++batch)
                    cblas_saxpy(nextSize, 1.0f, b, 1, mu + batch * nextSize, 1);

                activation(mu, Nout);

                if (muCacheThreshold >= 0.0f)
                {
                    cblas_scopy((int)Nout, mu, 1, cachedMu, 1);
                    cblas_scopy((int)N, z, 1, prevZ, 1);
                    muCacheValid = true;
                }
            }
            else
            {
                // Restore the preserved, correctly-activated prediction --
                // mu was left holding last step's DERIVATIVE by
                // UpdateState(), not the prediction itself.
                cblas_scopy((int)Nout, cachedMu, 1, mu, 1);
            }
        }

        return totalEnergy;
    }

    void SimplePCLayer::UpdateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

        if (nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;
            activationDerivative(mu, Nout, true);
        }

        if (isClamped)
            return;

        // Own term: dz_dt = -e (was -p*e; p=1 always made this a no-op
        // multiply, so the multiplication itself is simply removed).
#pragma omp parallel for schedule(static)
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
                __m512 e512 = _mm512_loadu_ps(&e[offset + i]);
                _mm512_storeu_ps(&dz_dt[offset + i], _mm512_mul_ps(neg_one, e512));
            }
#elif defined(__AVX2__) || defined(__AVX__)
            __m256 neg_one = _mm256_set1_ps(-1.0f);
            size_t r = size % 8;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 8)
            {
                __m256 e256 = _mm256_loadu_ps(&e[offset + i]);
                _mm256_storeu_ps(&dz_dt[offset + i], _mm256_mul_ps(neg_one, e256));
            }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
            __m128 neg_one = _mm_set1_ps(-1.0f);
            size_t r = size % 4;
            size_t simd_end = size - r;
            for (; i < simd_end; i += 4)
            {
                __m128 e128 = _mm_loadu_ps(&e[offset + i]);
                _mm_storeu_ps(&dz_dt[offset + i], _mm_mul_ps(neg_one, e128));
            }
#endif
            for (; i < size; ++i)
                dz_dt[offset + i] = -e[offset + i];
        }

        if (layerAbove != nullptr && nextSize > 0)
        {
            const float *e_above = layerAbove->GetErrors();

            // bottom_up = e_above * mu(f') -- was e_above * p_above * mu(f');
            // p_above=1 always made that multiply a no-op, removed.
#pragma omp parallel for schedule(static)
            for (int batch = 0; batch < batchSize; ++batch)
            {
                size_t offset = (size_t)batch * nextSize;
                size_t f = 0;
#if defined(__AVX512F__)
                size_t r = nextSize % 16;
                size_t simd_end = nextSize - r;
                for (; f < simd_end; f += 16)
                {
                    __m512 e_above512 = _mm512_loadu_ps(e_above + offset + f);
                    __m512 mu512 = _mm512_loadu_ps(mu + offset + f);
                    _mm512_storeu_ps(bottom_up + offset + f, _mm512_mul_ps(e_above512, mu512));
                }
#elif defined(__AVX2__) || defined(__AVX__)
                size_t r = nextSize % 8;
                size_t simd_end = nextSize - r;
                for (; f < simd_end; f += 8)
                {
                    __m256 e_above256 = _mm256_loadu_ps(e_above + offset + f);
                    __m256 mu256 = _mm256_loadu_ps(mu + offset + f);
                    _mm256_storeu_ps(bottom_up + offset + f, _mm256_mul_ps(e_above256, mu256));
                }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
                size_t r = nextSize % 4;
                size_t simd_end = nextSize - r;
                for (; f < simd_end; f += 4)
                {
                    __m128 e_above128 = _mm_loadu_ps(e_above + offset + f);
                    __m128 mu128 = _mm_loadu_ps(mu + offset + f);
                    _mm_storeu_ps(bottom_up + offset + f, _mm_mul_ps(e_above128, mu128));
                }
#endif
                for (; f < nextSize; ++f)
                    bottom_up[offset + f] = e_above[offset + f] * mu[offset + f];
            }

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, nextSize,
                1.0f, bottom_up, nextSize, W, size,
                1.0f, dz_dt, size);
        }

        cblas_saxpy(N, ir, dz_dt, 1, z, 1);
    }

    void SimplePCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || nextSize == 0)
            return;

        const float *e_above = layerAbove->GetErrors();
        float *local_grad = bottom_up;

        // 1. Compute the local gradient delta (shared across all optimizers)
#pragma omp parallel for schedule(static)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            size_t offset = (size_t)batch * nextSize;
            size_t f = 0;
#if defined(__AVX512F__)
            size_t r = nextSize % 16;
            size_t simd_end = nextSize - r;
            for (; f < simd_end; f += 16)
            {
                __m512 e512 = _mm512_loadu_ps(e_above + offset + f);
                __m512 mu512 = _mm512_loadu_ps(mu + offset + f);
                _mm512_storeu_ps(local_grad + offset + f, _mm512_mul_ps(e512, mu512));
            }
#elif defined(__AVX2__) || defined(__AVX__)
            size_t r = nextSize % 8;
            size_t simd_end = nextSize - r;
            for (; f < simd_end; f += 8)
            {
                __m256 e256 = _mm256_loadu_ps(e_above + offset + f);
                __m256 mu256 = _mm256_loadu_ps(mu + offset + f);
                _mm256_storeu_ps(local_grad + offset + f, _mm256_mul_ps(e256, mu256));
            }
#elif defined(__SSE__) || defined(_M_AMD64) || defined(_M_X64)
            size_t r = nextSize % 4;
            size_t simd_end = nextSize - r;
            for (; f < simd_end; f += 4)
            {
                __m128 e128 = _mm_loadu_ps(e_above + offset + f);
                __m128 mu128 = _mm_loadu_ps(mu + offset + f);
                _mm_storeu_ps(local_grad + offset + f, _mm_mul_ps(e128, mu128));
            }
#endif
            for (; f < nextSize; ++f)
                local_grad[offset + f] = e_above[offset + f] * mu[offset + f];
        }

        // 2. Dispatch to the selected optimizer
        switch (opt)
        {
        case OptimizerType::SGD:
        {
            if (lmbda > 0.0f)
                cblas_sscal((size_t)nextSize * size, 1.0f - lmbda, W, 1);

            // Writes scaled updates directly into W via beta=1.0f
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                nextSize, size, batchSize,
                lr / batchSize, local_grad, nextSize, z, size,
                1.0f, W, size);

            float lr_batch = lr / batchSize;
            for (int batch = 0; batch < batchSize; batch++)
                cblas_saxpy(nextSize, lr_batch, local_grad + batch * nextSize, 1, b, 1);

            break;
        }
        case OptimizerType::ADAM:
        case OptimizerType::ADAMW:
        {
            t++; // Increment layer-wide timestep ONCE

            size_t num_weights = (size_t)nextSize * size;

            // NOTE: Using 1.0f / batchSize instead of 1.0f so the gradient
            // magnitude doesn't drastically shift Adam's variance estimates
            // if you change your batch size.
            float grad_scale = -1.0f / batchSize;

            // Compute raw gradients for W into grad_W
            // beta=0.0f overwrites grad_W cleanly, no memset needed
            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                nextSize, size, batchSize,
                grad_scale, local_grad, nextSize, z, size,
                0.0f, grad_W, size);

            // Compute raw gradients for biases into grad_b
            std::memset(grad_b, 0, nextSize * sizeof(float));
            for (int batch = 0; batch < batchSize; batch++)
                cblas_saxpy(nextSize, grad_scale, local_grad + batch * nextSize, 1, grad_b, 1);

            // Apply specific Adam flavor to Weights
            if (opt == OptimizerType::ADAMW)
            {
                Deep::AdamWUpdate(W, grad_W, m_W, v_W, num_weights, t, lr, lmbda);
            }
            else
            {
                Deep::AdamUpdate(W, grad_W, m_W, v_W, num_weights, t, lr);
            }

            // Apply plain Adam to Biases (biases almost never use weight decay)
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
        size_t copySize = std::min(inputData.size(), (size_t)(batchSize * size)) * sizeof(float);
        memcpy(z, inputData.data(), copySize);
        isClamped = true;
        muCacheValid = false; // fresh data this batch -- must recompute at least once
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

        // z, e, dz_dt -- NO p/log_p buffers at all
        total += pad16(own_state_size) * 3;

        if (nextSize > 0)
        {
            size_t out_state_size = (size_t)batchSize * nextSize;
            size_t w_size = (size_t)size * nextSize;

            total += pad16(w_size);             // W
            total += pad16(nextSize);           // b
            total += pad16(out_state_size) * 3; // mu, cachedMu, bottom_up
            total += pad16(own_state_size);     // prevZ (own_state_size, matches z)

            // Conditionally allocate Adam variables to save space for SGD users
            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(w_size) * 3;   // grad_W, m_W, v_W
                total += pad16(nextSize) * 3; // grad_b, m_b, v_b
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
        dz_dt = arena.AllocateFloats(own_state_size);

        std::memset(z, 0, own_state_size * sizeof(float));
        std::memset(e, 0, own_state_size * sizeof(float));
        std::memset(dz_dt, 0, own_state_size * sizeof(float));

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;

            W = arena.AllocateFloats(w_size);
            b = arena.AllocateFloats(nextSize);
            mu = arena.AllocateFloats(out_state_size);
            cachedMu = arena.AllocateFloats(out_state_size);
            bottom_up = arena.AllocateFloats(out_state_size);
            prevZ = arena.AllocateFloats(own_state_size);

            std::memset(b, 0, nextSize * sizeof(float));
            std::memset(mu, 0, out_state_size * sizeof(float));
            std::memset(cachedMu, 0, out_state_size * sizeof(float));
            std::memset(bottom_up, 0, out_state_size * sizeof(float));
            std::memset(prevZ, 0, own_state_size * sizeof(float));

            // Conditionally bind Adam variables
            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                grad_W = arena.AllocateFloats(w_size);
                m_W = arena.AllocateFloats(w_size);
                v_W = arena.AllocateFloats(w_size);

                grad_b = arena.AllocateFloats(nextSize);
                m_b = arena.AllocateFloats(nextSize);
                v_b = arena.AllocateFloats(nextSize);

                // Adam moments must begin at zero
                std::memset(m_W, 0, w_size * sizeof(float));
                std::memset(v_W, 0, w_size * sizeof(float));
                std::memset(m_b, 0, nextSize * sizeof(float));
                std::memset(v_b, 0, nextSize * sizeof(float));

                // It's good practice to zero the gradients as well
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