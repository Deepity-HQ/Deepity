#include <deepity/layers/GaussSeidelPCLayer.h>
#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <cstring>
#include <immintrin.h>
#include <omp.h>
#include <algorithm>

namespace Deep
{
    GaussSeidelPCLayer::GaussSeidelPCLayer(int size, int nextSize, int batchSize,
                                            float learningRate, float inferenceRate, float lmbda,
                                            void (*act)(float *, size_t),
                                            void (*dAct)(float *, size_t, bool))
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct),
          activationType(ActivationType::NONE)
    {
        this->size = size;
        this->nextSize = nextSize;
        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    GaussSeidelPCLayer::GaussSeidelPCLayer(int size, int nextSize, int batchSize,
                                            float learningRate, float inferenceRate, float lmbda,
                                            ActivationType aType, ActivationType dType)
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda),
          layerAbove(nullptr), layerBelow(nullptr),
          activation(To_Fn(aType)), activationDerivative(To_dFn(dType)), activationType(aType)
    {
        this->size = size;
        this->nextSize = nextSize;
        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    void GaussSeidelPCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t copySize = std::min(inputData.size(), (size_t)(batchSize * size)) * sizeof(float);
        memcpy(z, inputData.data(), copySize);
        isClamped = true;
    }

    void GaussSeidelPCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    void GaussSeidelPCLayer::ResetState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * size;
        std::memset(z, 0, ownStateSize * sizeof(float));
        std::memset(e, 0, ownStateSize * sizeof(float));
        std::memset(dz_dt, 0, ownStateSize * sizeof(float));
        if (nextSize > 0)
        {
            size_t outStateSize = (size_t)batchSize * nextSize;
            std::memset(mu, 0, outStateSize * sizeof(float));
        }
        isClamped = false;
    }

	void GaussSeidelPCLayer::UpdateState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * size;

        if (isClamped) return;

        // 1. Heavy lifting: BLAS Matrix Multiplication
        if (layerAbove != nullptr && nextSize > 0)
        {
            const float *e_above = layerAbove->GetErrors();
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, nextSize,
                1.0f, e_above, nextSize, W, size,
                0.0f, dz_dt, size);
        }
        else
        {
            std::memset(dz_dt, 0, ownStateSize * sizeof(float));
        }

        // 2. FUSED E-M STEP (L1 Cache Blocked)
        // We chunk the operations so z_deriv never leaves the L1 cache.
        constexpr int CHUNK_SIZE = 2048;

        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)ownStateSize; i += CHUNK_SIZE)
        {
            int chunk = std::min(CHUNK_SIZE, (int)ownStateSize - i);

            // Step A: Copy z into z_deriv for this chunk only
            for (int j = 0; j < chunk; ++j) {
                z_deriv[i + j] = z[i + j];
            }

            // Step B: Compute derivative in-place (data is hot in L1)
            activationDerivative(z_deriv + i, chunk, false);

            // Step C: Apply E-M feedback and Euler step immediately
            for (int j = 0; j < chunk; ++j) {
                float feedback = dz_dt[i + j];
                float dz = -e[i + j] + (feedback * z_deriv[i + j]);
                z[i + j] += ir * dz;
            }
        }
    }

	void GaussSeidelPCLayer::ComputePrediction() noexcept
    {
        if (nextSize == 0) return;

        size_t ownStateSize = (size_t)batchSize * size;
        size_t Nout = (size_t)batchSize * nextSize;
        
        // FUSED: Chunked copy + activation
        constexpr int CHUNK_SIZE = 2048;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)ownStateSize; i += CHUNK_SIZE)
        {
            int chunk = std::min(CHUNK_SIZE, (int)ownStateSize - i);
            for (int j = 0; j < chunk; ++j) dz_dt[i + j] = z[i + j];
            activation(dz_dt + i, chunk);
        }

        // mu = phi(z) * W^T
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            batchSize, nextSize, size,
            1.0f, dz_dt, size, W, size, 0.0f, mu, nextSize);

        // FUSED: Bias addition
        #pragma omp parallel for schedule(static) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch) {
            for (size_t out = 0; out < nextSize; ++out) {
                mu[batch * nextSize + out] += b[out];
            }
        }
    }
    
    // ------------------------------------------------------------------
    // Step 3: fresh error, using this layer's own (just-updated) z as
    // the target and layerBelow's FRESH mu (from its ComputePrediction()
    // call, earlier in this same timestep) as the prediction.
    // ------------------------------------------------------------------
float GaussSeidelPCLayer::ComputeError() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * size;

        if (layerBelow == nullptr)
        {
            std::memset(e, 0, ownStateSize * sizeof(float));
            return 0.0f;
        }

        const float* mu_below = layerBelow->GetMu();
        float totalEnergy = 0.0f;

        // FUSED: Subtract prediction and accumulate energy in one single pass
        #pragma omp parallel for schedule(static) reduction(+:totalEnergy)
        for (int i = 0; i < (int)ownStateSize; ++i)
        {
            float err = z[i] - mu_below[i];
            e[i] = err;
            totalEnergy += 0.5f * err * err;
        }

        return totalEnergy;
    }

void GaussSeidelPCLayer::UpdateWeights() noexcept
    {
	if (layerAbove == nullptr || nextSize == 0) return;

        const float *local_grad = layerAbove->GetErrors();
        size_t ownStateSize = (size_t)batchSize * size;
        
        // FUSED: Chunked copy + activation
        constexpr int CHUNK_SIZE = 2048;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < (int)ownStateSize; i += CHUNK_SIZE)
        {
            int chunk = std::min(CHUNK_SIZE, (int)ownStateSize - i);
            for (int j = 0; j < chunk; ++j) dz_dt[i + j] = z[i + j];
            activation(dz_dt + i, chunk);
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
                lr / batchSize, local_grad, nextSize, dz_dt, size,
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
            float grad_scale = -1.0f / batchSize;

            cblas_sgemm(
                CblasRowMajor, CblasTrans, CblasNoTrans,
                nextSize, size, batchSize,
                grad_scale, local_grad, nextSize, dz_dt, size,
                0.0f, grad_W, size);

            std::memset(grad_b, 0, nextSize * sizeof(float));
            for (int batch = 0; batch < batchSize; ++batch)
                cblas_saxpy(nextSize, grad_scale, local_grad + batch * nextSize, 1, grad_b, 1);

            if (opt == OptimizerType::ADAMW)
                Deep::AdamWUpdate(W, grad_W, m_W, v_W, (size_t)nextSize * size, t, lr, lmbda);
            else
                Deep::AdamUpdate(W, grad_W, m_W, v_W, (size_t)nextSize * size, t, lr);

            Deep::AdamUpdate(b, grad_b, m_b, v_b, nextSize, t, lr);
            break;
        }
        }
    }

       void GaussSeidelPCLayer::RandomizeWeights(std::mt19937 &twister) noexcept
    {
        if (nextSize == 0)
            return;

        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = (size_t)size * nextSize;
        float limit = std::sqrt(2.0f / (size + nextSize));

        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(twister);

#pragma omp parallel
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
            std::normal_distribution<float> dist(0.0f, limit);

#pragma omp for
            for (ptrdiff_t i = 0; i < (ptrdiff_t)Wsz; ++i)
                W[i] = dist(rng);
        }

        // E: feedback-alignment matrix -- INDEPENDENT random draw (fresh
        // seeds, matching ngc-learn's use of a separate subkey for E vs
        // W), uniform +-0.3 (matching ngc-learn's ACTUAL StaticSynapse
        // init convention exactly -- not the Gaussian/Xavier-style limit
        // W uses above). Never touched again after this -- no evolve()
        // call exists for it, matching "Static" in StaticSynapse.
        std::vector<uint32_t> eSeeds(omp_get_max_threads());
        for (auto &s : eSeeds)
            s = seedDist(twister);

#pragma omp parallel
        {
            std::mt19937 rng(eSeeds[omp_get_thread_num()]);
            std::uniform_real_distribution<float> eDist(-0.3f, 0.3f);

#pragma omp for
            for (ptrdiff_t i = 0; i < (ptrdiff_t)Wsz; ++i)
                E[i] = eDist(rng);
        }
    }

    size_t GaussSeidelPCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t own_state_size = (size_t)batchSize * size;

        total += pad16(own_state_size) * 4; // z, e, dz_dt, z_deriv

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;
            size_t out_state_size = (size_t)batchSize * nextSize;

            total += pad16(w_size);              // W
            total += pad16(nextSize);            // b
            total += pad16(out_state_size) * 3;  // mu, muDeriv, bottom_up
            total += pad16(w_size);              // E (feedback alignment, same shape as W)

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(w_size) * 3;   // grad_W, m_W, v_W
                total += pad16(nextSize) * 3; // grad_b, m_b, v_b
            }
        }

        return total;
    }

    void GaussSeidelPCLayer::BindMemory(MemoryArena &arena)
    {
        size_t own_state_size = (size_t)batchSize * size;

        z = arena.AllocateFloats(own_state_size);
        e = arena.AllocateFloats(own_state_size);
        dz_dt = arena.AllocateFloats(own_state_size);
	z_deriv = arena.AllocateFloats(own_state_size);

        std::memset(z, 0, own_state_size * sizeof(float));
        std::memset(e, 0, own_state_size * sizeof(float));
        std::memset(dz_dt, 0, own_state_size * sizeof(float));
	std::memset(z_deriv, 0, own_state_size * sizeof(float));

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;
            size_t out_state_size = (size_t)batchSize * nextSize;

            W = arena.AllocateFloats(w_size);
            b = arena.AllocateFloats(nextSize);
            mu = arena.AllocateFloats(out_state_size);
            muDeriv = arena.AllocateFloats(out_state_size);
            bottom_up = arena.AllocateFloats(out_state_size);
            E = arena.AllocateFloats(w_size);

            std::memset(b, 0, nextSize * sizeof(float));
            std::memset(mu, 0, out_state_size * sizeof(float));
            std::memset(muDeriv, 0, out_state_size * sizeof(float));
            std::memset(bottom_up, 0, out_state_size * sizeof(float));
            std::memset(E, 0, w_size * sizeof(float));

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                grad_W = arena.AllocateFloats(w_size);
                m_W = arena.AllocateFloats(w_size);
                v_W = arena.AllocateFloats(w_size);
                grad_b = arena.AllocateFloats(nextSize);
                m_b = arena.AllocateFloats(nextSize);
                v_b = arena.AllocateFloats(nextSize);

                std::memset(grad_W, 0, w_size * sizeof(float));
                std::memset(m_W, 0, w_size * sizeof(float));
                std::memset(v_W, 0, w_size * sizeof(float));
                std::memset(grad_b, 0, nextSize * sizeof(float));
                std::memset(m_b, 0, nextSize * sizeof(float));
                std::memset(v_b, 0, nextSize * sizeof(float));
            }
        }
        else
        {
            W = nullptr;
            b = nullptr;
            mu = nullptr;
            muDeriv = nullptr;
            E = nullptr;
            bottom_up = nullptr;
        }
    }
}
