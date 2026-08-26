#include <deepity/layers/GaussSeidelPCLayer.h>
#include <cblas.h>
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

    // ------------------------------------------------------------------
    // Step 1: z-update, using mu/e_above HELD OVER from the end of the
    // previous timestep. Structurally the SAME feedback math as
    // SimplePCLayer's UpdateState() -- the only real difference is mu's
    // derivative is computed into a SEPARATE scratch buffer (muDeriv),
    // never mutating mu itself, since the layer above's ComputeError()
    // needs mu in its clean, activated form later in THIS SAME timestep.
    // ------------------------------------------------------------------
    void GaussSeidelPCLayer::UpdateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

        if (isClamped)
            return;

        std::memset(dz_dt, 0, N * sizeof(float));

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
        for (int batch = 0; batch < batchSize; ++batch)
        {
            size_t offset = (size_t)batch * size;
            for (size_t i = 0; i < size; ++i)
                dz_dt[offset + i] = -e[offset + i];
        }

        if (layerAbove != nullptr && nextSize > 0)
        {
            const float *e_above = layerAbove->GetErrors();

            // Derivative computed into a SEPARATE buffer -- mu itself is
            // NOT touched, and stays valid for the layer above to read
            // later in this same timestep (via ComputeError(), called
            // AFTER this timestep's own ComputePrediction()).
            cblas_scopy((int)((size_t)batchSize * nextSize), mu, 1, muDeriv, 1);
            activationDerivative(muDeriv, (size_t)batchSize * nextSize, true);

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
            for (int batch = 0; batch < batchSize; ++batch)
            {
                size_t offset = (size_t)batch * nextSize;
                for (size_t f = 0; f < nextSize; ++f)
                    bottom_up[offset + f] = e_above[offset + f] * muDeriv[offset + f];
            }

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, nextSize,
                1.0f, bottom_up, nextSize, W, size,
                1.0f, dz_dt, size);
        }

        cblas_saxpy((int)N, ir, dz_dt, 1, z, 1);
    }

    // ------------------------------------------------------------------
    // Step 2: fresh prediction, using the z JUST updated by UpdateState()
    // in THIS same timestep. No error/energy computed here at all --
    // that's ComputeError()'s job, called after every layer's
    // ComputePrediction() has run.
    // ------------------------------------------------------------------
    void GaussSeidelPCLayer::ComputePrediction() noexcept
    {
        if (nextSize == 0)
            return;
        size_t Nout = (size_t)batchSize * nextSize;
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            batchSize, nextSize, size,
            1.0f, z, size, W, size, 0.0f, mu, nextSize);
        for (int batch = 0; batch < batchSize; ++batch)
            cblas_saxpy(nextSize, 1.0f, b, 1, mu + batch * nextSize, 1);
        activation(mu, Nout);
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

        cblas_scopy((int)ownStateSize, z, 1, e, 1);
        cblas_saxpy((int)ownStateSize, -1.0f, layerBelow->GetMu(), 1, e, 1);

        float totalEnergy = 0.0f;
#pragma omp parallel for schedule(static) reduction(+ : totalEnergy) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            for (size_t i = 0; i < size; ++i)
            {
                float err = e[(size_t)batch * size + i];
                totalEnergy += 0.5f * err * err;
            }
        }

        return totalEnergy;
    }

    // ------------------------------------------------------------------
    // Called ONCE after the full settling loop completes, matching
    // ngc-learn's evolve_process. Recomputes mu's derivative fresh at
    // this point (into muDeriv) rather than relying on a value computed
    // earlier in this timestep's UpdateState() -- that earlier value
    // would be one timestep stale relative to the CURRENT, final mu.
    // ------------------------------------------------------------------
    void GaussSeidelPCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || nextSize == 0)
            return;

        const float *e_above = layerAbove->GetErrors();
        float *local_grad = bottom_up;

        cblas_scopy((int)((size_t)batchSize * nextSize), mu, 1, muDeriv, 1);
        activationDerivative(muDeriv, (size_t)batchSize * nextSize, true);

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
        for (int batch = 0; batch < batchSize; ++batch)
        {
            size_t offset = (size_t)batch * nextSize;
            for (size_t f = 0; f < nextSize; ++f)
                local_grad[offset + f] = e_above[offset + f] * muDeriv[offset + f];
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
                lr / batchSize, local_grad, nextSize, z, size,
                1.0f, W, size);

            float lr_batch = lr / batchSize;
#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
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
                grad_scale, local_grad, nextSize, z, size,
                0.0f, grad_W, size);

            std::memset(grad_b, 0, nextSize * sizeof(float));
#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
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
    }

    size_t GaussSeidelPCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t own_state_size = (size_t)batchSize * size;

        total += pad16(own_state_size) * 3; // z, e, dz_dt

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;
            size_t out_state_size = (size_t)batchSize * nextSize;

            total += pad16(w_size);             // W
            total += pad16(nextSize);           // b
            total += pad16(out_state_size) * 3; // mu, muDeriv, bottom_up

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

        std::memset(z, 0, own_state_size * sizeof(float));
        std::memset(e, 0, own_state_size * sizeof(float));
        std::memset(dz_dt, 0, own_state_size * sizeof(float));

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;
            size_t out_state_size = (size_t)batchSize * nextSize;

            W = arena.AllocateFloats(w_size);
            b = arena.AllocateFloats(nextSize);
            mu = arena.AllocateFloats(out_state_size);
            muDeriv = arena.AllocateFloats(out_state_size);
            bottom_up = arena.AllocateFloats(out_state_size);

            std::memset(b, 0, nextSize * sizeof(float));
            std::memset(mu, 0, out_state_size * sizeof(float));
            std::memset(muDeriv, 0, out_state_size * sizeof(float));
            std::memset(bottom_up, 0, out_state_size * sizeof(float));

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
            bottom_up = nullptr;
        }
    }
}