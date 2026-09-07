#include "deepity/layers/DirectKPPCLayer.h"
#include "deepity/backend/CPUBackend.h"
#include "deepity/utils/Optimize.h"
#include <cstring>
#include <type_traits>

namespace Deep
{
    namespace
    {
        // Same mapping as SimplePCLayer.cpp -- see its own comment for
        // why this is needed (activationType stores the FORWARD type,
        // but ActivationDerivativeInto expects the DERIVATIVE type).
        // Note this also closes a real gap the old code had: the
        // original UpdateState()'s switch fell back to a raw function
        // pointer for eSIGMOID (no dedicated derivative variant existed
        // in that dispatch), whereas backend->ActivationDerivativeInto
        // DOES have a real d_eSIGMOID case.
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

    DirectKPPCLayer::DirectKPPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
                                     float learningRate, float inferenceRate, float feedback, float lmbda,
                                     ActivationType aType, ActivationType dType,
                                     IComputeBackend *backend)
        : size(size),
          nextSize(nextSize),
          terminalSize(terminalSize),
          batchSize(batchSize),
          lr(learningRate),
          ir(inferenceRate),
          fl(feedback),
          lmbda(lmbda),
          activationType(aType),
          backend(backend ? backend : new CPUBackend(),
                  backend ? NoOpDeleter : DeleteBackend)
    {
        this->activation = To_Fn(aType);
        this->activationDerivative = To_dFn(dType);
        this->activationDerivativeInto = To_dFn2(dType);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    void DirectKPPCLayer::SetLearningRate(float learningRate) noexcept
    {
        lr = learningRate;
        if (lr_device)
            backend->CopyFromHost(lr_device, &lr, 1);
    }

    void DirectKPPCLayer::SetFeedbackRate(float feedbackRate) noexcept
    {
        fl = feedbackRate;
        if (fl_device)
            backend->CopyFromHost(fl_device, &fl, 1);
    }

    // --- Setup ---

    template <typename ArenaT>
    void DirectKPPCLayer::BindMemory(ArenaT &arena)
    {
        size_t own_state_size = batchSize * size;
        size_t out_state_size = batchSize * nextSize;
        size_t direct_size = size * terminalSize;

        z = arena.AllocateFloats(own_state_size);
        e = arena.AllocateFloats(own_state_size);

        backend->Zero(z, own_state_size);
        backend->Zero(e, own_state_size);

        if (nextSize > 0)
        {
            size_t w_size = size * nextSize;

            W = arena.AllocateFloats(w_size);
            b = arena.AllocateFloats(nextSize);
            mu = arena.AllocateFloats(out_state_size);
            cachedMu = arena.AllocateFloats(out_state_size);
            proj = arena.AllocateFloats(out_state_size);
            Psi = arena.AllocateFloats(direct_size);
            zF = arena.AllocateFloats(own_state_size);
            zFDeriv = arena.AllocateFloats(own_state_size);
            feedbackScratch = arena.AllocateFloats(own_state_size);

            backend->Zero(b, nextSize);
            backend->Zero(mu, out_state_size);
            backend->Zero(cachedMu, out_state_size);
            backend->Zero(proj, out_state_size);
            backend->Zero(Psi, direct_size);
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

                // Device-resident t/lr for W's optimizer -- see header's
                // file-level note for why these can't just be host
                // values once graph capture is in the picture.
                t_device = reinterpret_cast<int *>(arena.AllocateFloats(1));
                lr_device = arena.AllocateFloats(1);
                int zero = 0;
                backend->CopyFromHost(reinterpret_cast<float *>(t_device), reinterpret_cast<float *>(&zero), 1);
                backend->CopyFromHost(lr_device, &lr, 1);
            }

            if (optPsi == OptimizerType::ADAM || optPsi == OptimizerType::ADAMW)
            {
                grad_Psi = arena.AllocateFloats(direct_size);
                m_Psi = arena.AllocateFloats(direct_size);
                v_Psi = arena.AllocateFloats(direct_size);

                backend->Zero(m_Psi, direct_size);
                backend->Zero(v_Psi, direct_size);
                backend->Zero(grad_Psi, direct_size);

                // Same as above, but for Psi's SEPARATE optimizer state.
                tPsi_device = reinterpret_cast<int *>(arena.AllocateFloats(1));
                fl_device = arena.AllocateFloats(1);
                int zero = 0;
                backend->CopyFromHost(reinterpret_cast<float *>(tPsi_device), reinterpret_cast<float *>(&zero), 1);
                backend->CopyFromHost(fl_device, &fl, 1);
            }
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

    size_t DirectKPPCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t own_state_size = batchSize * size;
        size_t direct_size = terminalSize * size;

        total += pad16(own_state_size) * 2;

        if (nextSize > 0)
        {
            size_t out_state_size = (size_t)batchSize * nextSize;
            size_t w_size = (size_t)size * nextSize;

            total += pad16(w_size);
            total += pad16(nextSize);
            total += pad16(out_state_size) * 3;
            total += pad16(own_state_size) * 3; // zF, zFDeriv, feedbackScratch
            total += pad16(direct_size);        // Psi

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(w_size) * 3;
                total += pad16(nextSize) * 3;
                total += pad16(1) * 2; // t_device, lr_device
            }

            if (optPsi == OptimizerType::ADAM || optPsi == OptimizerType::ADAMW)
            {
                total += pad16(direct_size) * 3;
                total += pad16(1) * 2; // tPsi_device, fl_device
            }
        }

        return total;
    }

    void DirectKPPCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        if (nextSize == 0)
            return;

        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = size * nextSize;
        size_t Psisz = terminalSize * size;
        float limit = std::sqrt(2.0f / (size + nextSize));
        float limPsi = std::sqrt(2.0f / (size + terminalSize));

        uint32_t seedW = seedDist(seedGenerator);
        uint32_t seedPsi = seedDist(seedGenerator);

        backend->RandomizeNormal(W, Wsz, 0.0f, limit, seedW);
        backend->RandomizeNormal(Psi, Psisz, 0.0f, limPsi, seedPsi);
    }

    // --- Core DKP-PC Mechanics ---

    float DirectKPPCLayer::CalculateState(bool needEnergy) noexcept
    {
        const size_t N = batchSize * size;

        if (layerBelow == nullptr)
        {
            backend->Zero(e, N);
            if (nextSize > 0)
                ComputeMuOnly();
            return 0.0f;
        }

        float totalEnergy = 0.0f;
        if (needEnergy)
            totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->mu, N);
        else
            backend->ComputeError(e, z, layerBelow->mu, N);

        if (nextSize > 0)
            ComputeMuOnly();

        return totalEnergy;
    }

    void DirectKPPCLayer::ComputeMuOnly() noexcept
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
            /*transA=*/false, /*transB=*/true,
            (int)batchSize, (int)nextSize, (int)size,
            1.0f, zF, (int)size, W, (int)size, 0.0f, mu, (int)nextSize);

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
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

    void DirectKPPCLayer::UpdateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

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
            backend->AxpyInto(z, e, N, -ir);
        }
    }

    void DirectKPPCLayer::UpdateWeights() noexcept
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
                /*transA=*/true, /*transB=*/false,
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
            backend->IncrementCounter(t_device);

            size_t num_weights = (size_t)nextSize * size;
            float adam_scale = -1.0f;

            backend->MatMul(
                /*transA=*/true, /*transB=*/false,
                (int)nextSize, (int)size, (int)batchSize,
                adam_scale, local_grad, (int)nextSize, zF, (int)size,
                0.0f, grad_W, (int)size);

            backend->Zero(grad_b, nextSize);
            for (int batch = 0; batch < batchSize; batch++)
                backend->AxpyInto(grad_b, local_grad + batch * nextSize, nextSize, adam_scale);

            if (opt == OptimizerType::ADAMW)
                backend->AdamWStep(W, grad_W, m_W, v_W, num_weights, t_device, lr_device, lmbda);
            else
                backend->AdamStep(W, grad_W, m_W, v_W, num_weights, t_device, lr_device);

            backend->AdamStep(b, grad_b, m_b, v_b, nextSize, t_device, lr_device);
            break;
        }
        }

        switch (optPsi)
        {
        case OptimizerType::SGD:
        {
            backend->MatMul(
                /*transA=*/true, /*transB=*/false,
                (int)size, (int)terminalSize, (int)batchSize,
                fl / batchSize, z, (int)size, terminalLayer->GetErrors(), (int)terminalSize,
                1.0f, Psi, (int)terminalSize);
            break;
        }
        case OptimizerType::ADAMW:
        case OptimizerType::ADAM:
        {
            backend->IncrementCounter(tPsi_device);

            size_t num_weights_psi = size * terminalSize;
            float adam_scale = -1.0f;

            backend->MatMul(
                /*transA=*/true, /*transB=*/false,
                (int)size, (int)terminalSize, (int)batchSize,
                adam_scale, z, (int)size, terminalLayer->GetErrors(), (int)terminalSize,
                0.0f, grad_Psi, (int)terminalSize);

            if (optPsi == OptimizerType::ADAMW)
                backend->AdamWStep(Psi, grad_Psi, m_Psi, v_Psi, num_weights_psi, tPsi_device, fl_device, lmbda);
            else
                backend->AdamStep(Psi, grad_Psi, m_Psi, v_Psi, num_weights_psi, tPsi_device, fl_device);

            break;
        }
        }
    }

    void DirectKPPCLayer::DirectFeedbackUpdate() noexcept
    {
        // If the layer above has no Psi weights (i.e. it is the terminal layer), skip DFA
        if (layerAbove == nullptr || layerAbove->GetDirectFeedbackWeights() == nullptr)
            return;

        // proj = terminalLayer->GetErrors() @ layerAbove->GetDirectFeedbackWeights()^T
        backend->MatMul(
            /*transA=*/false, /*transB=*/true,
            (int)batchSize, (int)nextSize, (int)terminalSize,
            1.0f, terminalLayer->GetErrors(), (int)terminalSize,
            layerAbove->GetDirectFeedbackWeights(), (int)terminalSize,
            0.0f, proj, (int)nextSize);

        // W += fl * proj^T @ zF
        backend->MatMul(
            /*transA=*/true, /*transB=*/true,
            (int)nextSize, (int)size, (int)batchSize,
            fl / batchSize, proj, (int)nextSize,
            zF, (int)size,
            1.0f, W, (int)size);
    }

    // --- Getters / Setters ---

    void DirectKPPCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t copyFloats = (std::min)(inputData.size(), (size_t)(batchSize * size));
        backend->CopyFromHost(z, inputData.data(), copyFloats);
        isClamped = true;
        muCacheValid = false;
    }

    void DirectKPPCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    void DirectKPPCLayer::ResetState() noexcept
    {
        size_t N = (size_t)batchSize * size;
        backend->Zero(z, N);
    }

    template void DirectKPPCLayer::BindMemory<MemoryArena>(MemoryArena &arena);
#if defined(DEEPITY_USE_CUDA)
    template void DirectKPPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena &arena);
#endif
}