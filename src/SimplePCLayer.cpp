#include <deepity/layers/SimplePCLayer.h>
#include <deepity/backend/CPUBackend.h>
#include <deepity/utils/Optimize.h>
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <omp.h>
#include <algorithm>
#include <cstring>

namespace Deep
{
    namespace
    {
        ActivationType ToDerivativeType(ActivationType fwd)
        {
            switch (fwd)
            {
            case ActivationType::RELU:
                return ActivationType::dRELU;
            case ActivationType::SIGMOID:
                return ActivationType::dSIGMOID;
            case ActivationType::eSIGMOID:
                return ActivationType::d_eSIGMOID;
            case ActivationType::TANH:
                return ActivationType::dTANH;
            case ActivationType::LINEAR:
                return ActivationType::dLINEAR;
            default:
                return ActivationType::NONE;
            }
        }

        void DeleteBackend(IComputeBackend *p) { delete p; }
        void NoOpDeleter(IComputeBackend *) {}
    }

    SimplePCLayer::SimplePCLayer(size_t size, size_t nextSize, size_t batchSize,
                                 float learningRate, float inferenceRate, float lmbda,
                                 void (*act)(float *, size_t),
                                 void (*dAct)(float *, size_t, bool),
                                 IComputeBackend *backend)
        : backend(backend ? backend : new CPUBackend(),
                  backend ? NoOpDeleter : DeleteBackend),
          batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
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
                                 ActivationType aType, ActivationType dType,
                                 IComputeBackend *backend)
        : backend(backend ? backend : new CPUBackend(),
                  backend ? NoOpDeleter : DeleteBackend),
          batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
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

    void SimplePCLayer::RandomizeWeights(std::mt19937 &seedGenerator, const char *distribution) noexcept
    {
        char name[16] = {0};
        float a = 0.0f, b = 1.0f;

        if (sscanf(distribution, "%15[^(](%f,%f)", name, &a, &b) != 3)
        {
            // Malformed string -- fall back to this class's original,
            // validated default (He/Xavier-style normal init) rather than
            // silently doing nothing.
            name[0] = '\0';
            strcpy(name, "normal");
            a = 0.0f;
            b = std::sqrt(2.0f / (size + nextSize));
        }

        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = size * nextSize;
        uint32_t seed = seedDist(seedGenerator);

        if (strcmp(name, "normal") == 0)
            backend->RandomizeNormal(W, Wsz, a, b, seed);
        else if (strcmp(name, "uniform") == 0)
            backend->RandomizeUniform(W, Wsz, a, b, seed);
        else
            backend->RandomizeNormal(W, Wsz, 0.0f, std::sqrt(2.0f / (size + nextSize)), seed);
    }

    float SimplePCLayer::CalculateState() noexcept
    {
        const size_t N = batchSize * size;

        if (layerBelow == nullptr)
        {
            backend->Zero(e, N);
            if (nextSize > 0)
            {
                ComputeMuOnly();
            }
            return 0.0f;
        }

        float totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->mu, N);

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
            backend->Copy(mu, cachedMu, Nout);
            return;
        }

        backend->ActivationInto(activationType, zF, z, N);

        backend->MatMul(
            false, true,
            (int)batchSize, (int)nextSize, (int)size,
            1.0f, zF, (int)size, W, (int)size, 0.0f, mu, (int)nextSize);

        bool parallelOk = backend->GetDeviceType() == DeviceType::DEVICE_CPU;
#pragma omp parallel for schedule(static) if (batchSize > 4 && parallelOk && !omp_in_parallel())
        for (int batch = 0; batch < batchSize; ++batch)
        {
            backend->AxpyInto(mu + batch * nextSize, b, nextSize, 1.0f);
        }

        if (isClamped)
        {
            backend->Copy(cachedMu, mu, Nout);
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

            backend->ActivationDerivativeInto(ToDerivativeType(activationType), zFDeriv, z, N);

            backend->MatMul(
                /*transA=*/false, /*transB=*/false,
                (int)batchSize, (int)size, (int)nextSize,
                1.0f, e_above, (int)nextSize, W, (int)size,
                0.0f, feedbackScratch, (int)size);

            backend->FusedStateUpdate(z, feedbackScratch, zFDeriv, e, N, ir);
        }
        else // Output Layer
        {
            // z[i] += ir * (-e[i])  ==  z += (-ir) * e, i.e. AxpyInto
            // with alpha = -ir.
            backend->AxpyInto(z, e, N, -ir);
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
                backend->Scale(W, (size_t)nextSize * size, 1.0f - lmbda);

            backend->MatMul(
                true, false,
                (int)nextSize, (int)size, (int)batchSize,
                lr / batchSize, local_grad, (int)nextSize, zF, (int)size,
                1.0f, W, (int)size);

            float lr_batch = lr / batchSize;
            for (int batch = 0; batch < batchSize; batch++)
                backend->AxpyInto(b, local_grad + batch * nextSize, nextSize, lr_batch);

            break;
        }
        case OptimizerType::ADAM:
        case OptimizerType::ADAMW:
        {
            t++;

            size_t num_weights = (size_t)nextSize * size;
            float grad_scale = -1.0f;

            backend->MatMul(
                true, false,
                (int)nextSize, (int)size, (int)batchSize,
                grad_scale, local_grad, (int)nextSize, zF, (int)size,
                0.0f, grad_W, (int)size);

            backend->Zero(grad_b, nextSize);
            for (int batch = 0; batch < batchSize; batch++)
                backend->AxpyInto(grad_b, local_grad + batch * nextSize, nextSize, grad_scale);

            if (opt == OptimizerType::ADAMW)
            {
                backend->AdamWStep(W, grad_W, m_W, v_W, num_weights, t, lr, lmbda);
            }
            else
            {
                backend->AdamStep(W, grad_W, m_W, v_W, num_weights, t, lr);
            }

            backend->AdamStep(b, grad_b, m_b, v_b, nextSize, t, lr);
            break;
        }
        }
    }

    void SimplePCLayer::ResetState() noexcept
    {
        size_t N = (size_t)batchSize * size;
        backend->Zero(z, N);
    }

    void SimplePCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t copyFloats = (std::min)(inputData.size(), (size_t)(batchSize * size));
        backend->CopyFromHost(z, inputData.data(), copyFloats);
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

        total += pad16(own_state_size) * 2;

        if (nextSize > 0)
        {
            size_t out_state_size = (size_t)batchSize * nextSize;
            size_t w_size = (size_t)size * nextSize;

            total += pad16(w_size);
            total += pad16(nextSize);
            total += pad16(out_state_size) * 2;
            total += pad16(own_state_size) * 3;

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(w_size) * 3;
                total += pad16(nextSize) * 3;
            }
        }

        return total;
    }

    template <typename ArenaT>
    void SimplePCLayer::BindMemory(ArenaT &arena)
    {
        size_t own_state_size = (size_t)batchSize * size;
        size_t out_state_size = (size_t)batchSize * nextSize;

        z = arena.AllocateFloats(own_state_size);
        e = arena.AllocateFloats(own_state_size);

        backend->Zero(z, own_state_size);
        backend->Zero(e, own_state_size);

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

            backend->Zero(b, nextSize);
            backend->Zero(mu, out_state_size);
            backend->Zero(cachedMu, out_state_size);
            backend->Zero(zF, own_state_size);
            backend->Zero(zFDeriv, own_state_size);
            backend->Zero(feedbackScratch, own_state_size);

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                grad_W = arena.AllocateFloats(w_size);
                m_W = arena.AllocateFloats(w_size);
                v_W = arena.AllocateFloats(w_size);

                grad_b = arena.AllocateFloats(nextSize);
                m_b = arena.AllocateFloats(nextSize);
                v_b = arena.AllocateFloats(nextSize);

                backend->Zero(m_W, w_size);
                backend->Zero(v_W, w_size);
                backend->Zero(m_b, nextSize);
                backend->Zero(v_b, nextSize);

                backend->Zero(grad_W, w_size);
                backend->Zero(grad_b, nextSize);
            }
        }

        if (localArena && localArena.get() != &arena)
        {
            localArena.reset();
        }
    }

    template void SimplePCLayer::BindMemory<MemoryArena>(MemoryArena &arena);
#if defined(DEEPITY_ENABLE_CUDA)
    template void SimplePCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena &arena);
#endif
}