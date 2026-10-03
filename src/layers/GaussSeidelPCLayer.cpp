#include <deepity/layers/GaussSeidelPCLayer.h>
#include <deepity/backend/CPUBackend.h>
#include <algorithm>
#include <type_traits>

namespace Deep
{
    namespace
    {
        void DeleteBackend(IComputeBackend *p) { delete p; }
        void NoOpDeleter(IComputeBackend *) {}
    }

    GaussSeidelPCLayer::GaussSeidelPCLayer(int size, int nextSize, int batchSize,
                                           float learningRate, float inferenceRate, float lmbda,
                                           ActivationType aType, ActivationType dType,
                                           IComputeBackend *backend)
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda),
          activationType(aType), derivativeType(dType),
          backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
    {
        this->size = size;
        this->nextSize = nextSize;

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    void GaussSeidelPCLayer::SetLearningRate(float learningRate) noexcept
    {
        lr = learningRate;
        if (lr_device)
            backend->CopyFromHost(lr_device, &lr, 1);
    }

    void GaussSeidelPCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t copyFloats = (std::min)(inputData.size(), (size_t)batchSize * size);
        backend->CopyFromHost(z, inputData.data(), copyFloats);
        isClamped = true;
    }

    void GaussSeidelPCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    void GaussSeidelPCLayer::ResetState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * size;
        backend->Zero(z, ownStateSize);
        backend->Zero(e, ownStateSize);
        backend->Zero(dz_dt, ownStateSize);
        if (nextSize > 0)
            backend->Zero(mu, (size_t)batchSize * nextSize);
        isClamped = false;
    }

    void GaussSeidelPCLayer::UpdateState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * size;

        if (isClamped)
            return;

        // Feedback goes through E (the independently-initialized,
        // never-updated feedback-alignment matrix), not W.
        if (layerAbove != nullptr && nextSize > 0)
        {
            const float *e_above = layerAbove->GetErrors();
            backend->MatMul(
                /*transA=*/false, /*transB=*/false,
                batchSize, (int)size, (int)nextSize,
                1.0f, e_above, (int)nextSize, E, (int)size,
                0.0f, dz_dt, (int)size);
        }
        else
        {
            backend->Zero(dz_dt, ownStateSize);
        }

        // z += ir * ((dz_dt * f'(z)) - e); deriv computed inline from z.
        backend->FusedStateUpdate(z, dz_dt, derivativeType, e, ownStateSize, ir);
    }

    void GaussSeidelPCLayer::ComputePrediction() noexcept
    {
        if (nextSize == 0)
            return;

        size_t N = (size_t)batchSize * size;

        // dz_dt := phi(z), reused as the activated-z scratch buffer.
        // UpdateState() (called earlier in the same Step()) is the only
        // other writer, and it's done with dz_dt by the time this runs.
        backend->ActivationInto(activationType, dz_dt, z, N);

        backend->MatMul(
            /*transA=*/false, /*transB=*/true,
            batchSize, (int)nextSize, (int)size,
            1.0f, dz_dt, (int)size, W, (int)size,
            0.0f, mu, (int)nextSize);

        backend->AddBiasBroadcast(mu, b, batchSize, nextSize);
    }

    float GaussSeidelPCLayer::ComputeError(bool needEnergy) noexcept
    {
        size_t ownStateSize = (size_t)batchSize * size;

        if (layerBelow == nullptr)
        {
            backend->Zero(e, ownStateSize);
            return 0.0f;
        }

        if (needEnergy)
            return backend->ComputeErrorAndEnergy(e, z, layerBelow->GetMu(), ownStateSize);

        backend->ComputeError(e, z, layerBelow->GetMu(), ownStateSize);
        return 0.0f;
    }

    void GaussSeidelPCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || nextSize == 0)
            return;

        const float *local_grad = layerAbove->GetErrors();
        size_t N = (size_t)batchSize * size;

        // dz_dt := phi(z), same reuse as ComputePrediction().
        backend->ActivationInto(activationType, dz_dt, z, N);

        switch (opt)
        {
        case OptimizerType::SGD:
        {
            if (lmbda > 0.0f)
                backend->Scale(W, (size_t)nextSize * size, 1.0f - lmbda);

            float lr_batch = lr / batchSize;

            backend->MatMul(
                /*transA=*/true, /*transB=*/false,
                (int)nextSize, (int)size, batchSize,
                lr_batch, local_grad, (int)nextSize, dz_dt, (int)size,
                1.0f, W, (int)size);

            backend->SumRows(biasGradScratch, local_grad, batchSize, nextSize);
            backend->AxpyInto(b, biasGradScratch, nextSize, lr_batch);
            break;
        }
        case OptimizerType::ADAM:
        case OptimizerType::ADAMW:
        {
            backend->IncrementCounter(t_device);

            float grad_scale = -1.0f / batchSize;

            backend->MatMul(
                /*transA=*/true, /*transB=*/false,
                (int)nextSize, (int)size, batchSize,
                grad_scale, local_grad, (int)nextSize, dz_dt, (int)size,
                0.0f, grad_W, (int)size);

            backend->SumRows(grad_b, local_grad, batchSize, nextSize);
            backend->Scale(grad_b, nextSize, grad_scale);

            if (opt == OptimizerType::ADAMW)
                backend->AdamWStep(W, grad_W, m_W, v_W, (size_t)nextSize * size, t_device, lr_device, lmbda);
            else
                backend->AdamStep(W, grad_W, m_W, v_W, (size_t)nextSize * size, t_device, lr_device);

            backend->AdamStep(b, grad_b, m_b, v_b, nextSize, t_device, lr_device);
            break;
        }
        }
    }

    void GaussSeidelPCLayer::RandomizeWeights(std::mt19937 &twister) noexcept
    {
        if (nextSize == 0)
            return;

        size_t Wsz = (size_t)size * nextSize;
        float limit = std::sqrt(2.0f / (size + nextSize));

        std::uniform_int_distribution<uint32_t> seedDist;
        uint32_t seedW = seedDist(twister);
        backend->RandomizeNormal(W, Wsz, 0.0f, limit, seedW);

        // E: feedback-alignment matrix, an independent random draw with
        // its own seed, uniform +-0.3, not the Gaussian/Xavier-style
        // limit W uses above. Never touched again after this.
        uint32_t seedE = seedDist(twister);
        backend->RandomizeUniform(E, Wsz, -0.3f, 0.3f, seedE);
    }

    size_t GaussSeidelPCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t own_state_size = (size_t)batchSize * size;

        total += pad16(own_state_size) * 2; // z, e
        total += pad16(own_state_size);     // dz_dt

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;
            size_t out_state_size = (size_t)batchSize * nextSize;

            total += pad16(w_size);           // W
            total += pad16(nextSize);         // b
            total += pad16(out_state_size);   // mu
            total += pad16(w_size);           // E (feedback alignment, same shape as W)
            total += pad16(nextSize);         // biasGradScratch

            // Always allocated, regardless of optimizer, so switching to Adam/AdamW after Compile() stays safe.
            total += pad16(w_size) * 3;   // grad_W, m_W, v_W
            total += pad16(nextSize) * 3; // grad_b, m_b, v_b
            total += pad16(1) * 2;        // t_device, lr_device
        }

        return total;
    }

    template <typename ArenaT>
    void GaussSeidelPCLayer::BindMemory(ArenaT &arena)
    {
        size_t own_state_size = (size_t)batchSize * size;

        z = arena.AllocateFloats(own_state_size);
        e = arena.AllocateFloats(own_state_size);
        dz_dt = arena.AllocateFloats(own_state_size);

        backend->Zero(z, own_state_size);
        backend->Zero(e, own_state_size);
        backend->Zero(dz_dt, own_state_size);

        if (nextSize > 0)
        {
            size_t w_size = (size_t)size * nextSize;
            size_t out_state_size = (size_t)batchSize * nextSize;

            W = arena.AllocateFloats(w_size);
            b = arena.AllocateFloats(nextSize);
            mu = arena.AllocateFloats(out_state_size);
            E = arena.AllocateFloats(w_size);
            biasGradScratch = arena.AllocateFloats(nextSize);

            backend->Zero(b, nextSize);
            backend->Zero(mu, out_state_size);
            backend->Zero(E, w_size);
            backend->Zero(biasGradScratch, nextSize);

            // Always allocated; see GetRequiredFloats().
            grad_W = arena.AllocateFloats(w_size);
            m_W = arena.AllocateFloats(w_size);
            v_W = arena.AllocateFloats(w_size);
            grad_b = arena.AllocateFloats(nextSize);
            m_b = arena.AllocateFloats(nextSize);
            v_b = arena.AllocateFloats(nextSize);

            backend->Zero(grad_W, w_size);
            backend->Zero(m_W, w_size);
            backend->Zero(v_W, w_size);
            backend->Zero(grad_b, nextSize);
            backend->Zero(m_b, nextSize);
            backend->Zero(v_b, nextSize);

            t_device = reinterpret_cast<int *>(arena.AllocateFloats(1));
            lr_device = arena.AllocateFloats(1);
            int zero = 0;
            backend->CopyFromHost(reinterpret_cast<float *>(t_device), reinterpret_cast<float *>(&zero), 1);
            backend->CopyFromHost(lr_device, &lr, 1);
        }
        else
        {
            W = nullptr;
            b = nullptr;
            mu = nullptr;
            E = nullptr;
            biasGradScratch = nullptr;
        }

        if constexpr (std::is_same_v<ArenaT, MemoryArena>)
        {
            if (localArena && localArena.get() != &arena)
                localArena.reset();
        }
        else
        {
            localArena.reset();
        }
    }

    std::map<std::string, TensorDescriptor> GaussSeidelPCLayer::GetStateDict() const
    {
        return {
            {"W", {W, {(size_t)nextSize, (size_t)size}}},
            {"b", {b, {(size_t)nextSize}}},
            {"E", {E, {(size_t)nextSize, (size_t)size}}}};
    }

    template void GaussSeidelPCLayer::BindMemory<MemoryArena>(MemoryArena &arena);
#if defined(DEEPITY_USE_CUDA)
    template void GaussSeidelPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena &arena);
#endif
}
