#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/backend/Backend.h>

namespace Deep
{
    ConvPCNetwork::ConvPCNetwork(int batchSize, DeviceType device) noexcept
        : device(device), batchSize(batchSize)
    {
        backend = CreateBackend(device);
    }

    void ConvPCNetwork::AddLayer(int inChannels, int outChannels,
                                 int inHeight, int inWidth,
                                 int kernelH, int kernelW,
                                 int strideH, int strideW,
                                 int padH, int padW,
                                 float lr, float ir, float pr, float lmbda,
                                 ActivationType aType, ActivationType dType)
    {
        auto l = std::make_unique<ConvPCLayer>(
            inChannels, outChannels, inHeight, inWidth,
            kernelH, kernelW, strideH, strideW, padH, padW,
            batchSize, lr, ir, pr, lmbda, aType, dType, backend.get());

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }
        layers.push_back(std::move(l));
    }

    void ConvPCNetwork::SetOptimizer(OptimizerType opt) noexcept
    {
        // Applied to every layer inside Compile() instead of immediately,
        // so it works even when called before any AddLayer().
        pendingOpt = opt;
    }

    void ConvPCNetwork::Compile()
    {
        for (auto &l : layers)
            l->SetOptimizer(pendingOpt);

        size_t total = 0;
        for (auto &l : layers)
            total += l->GetRequiredFloats();

        if (device == DeviceType::DEVICE_CPU)
        {
            cpuArena = std::make_unique<MemoryArena>(total);
            for (auto &l : layers)
                l->BindMemory(*cpuArena);
        }
#if defined(DEEPITY_USE_CUDA)
        else
        {
            backend->PrepareForBatchSize(batchSize);
            gpuArena = std::make_unique<DeviceMemoryArena>(backend.get(), total);
            for (auto &l : layers)
                l->BindMemory(*gpuArena);
        }
#endif
    }

    void ConvPCNetwork::RandomizeWeights(std::mt19937 &rng) noexcept
    {
        for (auto &l : layers)
            l->RandomizeWeights(rng);
    }

    void ConvPCNetwork::ResetState() noexcept
    {
        for (auto &l : layers)
            l->ResetState();
    }

    void ConvPCNetwork::Clamp(const std::vector<float> &input) noexcept
    {
        layers.front()->ClampState(input);
    }

    float ConvPCNetwork::CalculateState() noexcept
    {
        float e = 0.0f;
        for (auto &l : layers)
            e += l->CalculateState();
        return e;
    }

    void ConvPCNetwork::UpdateState() noexcept
    {
        for (auto &l : layers)
            l->UpdateState();
    }

    void ConvPCNetwork::UpdateWeights() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdateWeights();
    }

    void ConvPCNetwork::UpdatePrecision() noexcept
    {
        // NOTE: matches UpdateWeights()'s loop bound, but UpdatePrecision()
        // gates on layerBelow (not layerAbove), so this skips the
        // terminal layer's own precision update even though it has a
        // layerBelow and would otherwise update.
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdatePrecision();
    }

    float ConvPCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        GetTerminalLayer()->ClampState(y);

        float finalEnergy = 0.0f;
        for (int t = 0; t < inferenceSteps; ++t)
        {
            finalEnergy = CalculateState();
            UpdateState();
        }

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> ConvPCNetwork::Predict(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);

        for (int t = 0; t < inferenceSteps; ++t)
        {
            CalculateState();
            UpdateState();
        }

        ConvPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }

    void ConvPCNetwork::ProjectForward() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
        {
            layers[i]->ComputeMuOnly();

            // Don't overwrite an already-clamped layer's target with a
            // forward-projected guess.
            if (layers[i + 1]->IsClamped())
                continue;

            const float *mu = layers[i]->GetMu();
            float *nextZ = layers[i + 1]->GetBeliefs();
            size_t n = layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

            backend->Copy(nextZ, mu, n);
        }
    }

    float ConvPCNetwork::TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        ProjectForward();
        GetTerminalLayer()->ClampState(y);

        float finalEnergy = 0.0f;
        for (int t = 0; t < inferenceSteps; ++t)
        {
            finalEnergy = CalculateState();
            UpdateState();
        }

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> ConvPCNetwork::PredictWithProjection(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        ProjectForward();

        for (int t = 0; t < inferenceSteps; ++t)
        {
            CalculateState();
            UpdateState();
        }

        ConvPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }
}
