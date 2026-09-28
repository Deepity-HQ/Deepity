#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <deepity/backend/Backend.h>

namespace Deep
{
    GaussSeidelPCNetwork::GaussSeidelPCNetwork(int batchSize, DeviceType device) noexcept
        : device(device), batchSize(batchSize)
    {
        backend = CreateBackend(device);
    }

    void GaussSeidelPCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                                        void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool))
    {
        AddLayer(size, nextSize, lr, ir, lmbda, To_AType(act), To_AType(dAct));
    }

    void GaussSeidelPCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                                        ActivationType aType, ActivationType dType)
    {
        auto l = std::make_unique<GaussSeidelPCLayer>(
            size, nextSize, batchSize, lr, ir, lmbda, aType, dType, backend.get());

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }
        layers.push_back(std::move(l));
    }

    void GaussSeidelPCNetwork::RandomizeWeights(std::mt19937 &rng)
    {
        for (auto &l : layers)
            l->RandomizeWeights(rng);
    }

    void GaussSeidelPCNetwork::ResetState() noexcept
    {
        for (auto &l : layers)
            l->ResetState();
    }

    void GaussSeidelPCNetwork::Clamp(const std::vector<float> &input)
    {
        layers.front()->ClampState(input);
    }

    float GaussSeidelPCNetwork::Step() noexcept
    {
        // Sweep 1: EVERY layer's z updates, using mu/e_above held over
        // from the end of the previous step.
        for (auto &l : layers)
            l->UpdateState();

        // Sweep 2: EVERY layer's mu recomputes, using the z JUST updated
        // in sweep 1. Order among layers doesn't matter here.
        for (auto &l : layers)
            l->ComputePrediction();

        // Sweep 3: EVERY layer's error recomputes, using this step's
        // fresh z and layerBelow's fresh mu from sweep 2. Order among
        // layers doesn't matter here either. Energy is only meaningful
        // starting from this point.
        float totalEnergy = 0.0f;
        for (auto &l : layers)
            totalEnergy += l->ComputeError();

        return totalEnergy;
    }

    void GaussSeidelPCNetwork::UpdateWeights() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdateWeights();
    }

    void GaussSeidelPCNetwork::ProjectForward() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
        {
            layers[i]->ComputePrediction();

            // Don't overwrite an already-clamped layer's target with a
            // forward-projected guess.
            if (layers[i + 1]->IsClamped())
                continue;

            const float *mu = layers[i]->GetMu();
            float *nextZ = layers[i + 1]->GetBeliefs();
            size_t n = (size_t)layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

            backend->Copy(nextZ, mu, n);
        }
    }

    float GaussSeidelPCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        GetTerminalLayer()->ClampState(y);

        float finalEnergy = 0.0f;
        for (int t = 0; t < inferenceSteps; ++t)
            finalEnergy = Step();

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    float GaussSeidelPCNetwork::TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        ProjectForward();
        GetTerminalLayer()->ClampState(y);

        // Only the terminal layer's error is computed immediately (e3 =
        // z3 - mu3, target vs. projected prediction); hidden layers stay
        // at zero.
        GetTerminalLayer()->ComputeError();

        float finalEnergy = 0.0f;
        for (int t = 0; t < inferenceSteps; ++t)
            finalEnergy = Step();

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> GaussSeidelPCNetwork::Predict(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);

        // Seed the hidden states with the forward pass. Without this,
        // inference starts from z=0 and fails to reach the target.
        ProjectForward();

        for (int t = 0; t < inferenceSteps; ++t)
            Step();

        GaussSeidelPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = (size_t)terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }

    void GaussSeidelPCNetwork::Compile()
    {
        size_t total = 0;
        for (auto &layer : layers)
            total += layer->GetRequiredFloats();

        if (device == DeviceType::DEVICE_CPU)
        {
            cpuArena = std::make_unique<MemoryArena>(total);
            for (auto &layer : layers)
                layer->BindMemory(*cpuArena);
        }
#if defined(DEEPITY_USE_CUDA)
        else
        {
            backend->PrepareForBatchSize(batchSize);
            gpuArena = std::make_unique<DeviceMemoryArena>(backend.get(), total);
            for (auto &layer : layers)
                layer->BindMemory(*gpuArena);
        }
#endif
    }
}
