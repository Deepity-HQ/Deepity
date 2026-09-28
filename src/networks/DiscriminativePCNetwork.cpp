#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/backend/Backend.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/ModelIO.h>
#include <xmmintrin.h>
#include <pmmintrin.h>

namespace Deep
{
    /// @brief Flushes denormal floats to zero on both input and output of
    /// SSE FP ops, avoiding the severe slowdown x86 FPUs hit computing
    /// with denormals, settling can drive values arbitrarily close to
    /// zero over many steps.
    static inline void ProtectFPU()
    {
        _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
        _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }

    DiscriminativePCNetwork::DiscriminativePCNetwork(DeviceType device)
        : batchSize(0), autoSize(true), device(device)
    {
        backend = CreateBackend(device);
    }

    DiscriminativePCNetwork::DiscriminativePCNetwork(int batchSize, DeviceType device)
        : batchSize(batchSize), autoSize(false), device(device)
    {
        backend = CreateBackend(device);
    }

    void DiscriminativePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float pr, float lmbda,
                                           void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool))
    {
        AddLayer(size, nextSize, lr, ir, pr, lmbda, To_AType(act), To_AType(dAct));
    }

    void DiscriminativePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float pr, float lmbda,
                                           Deep::ActivationType aType, Deep::ActivationType dType)
    {
        if (autoSize && layers.empty())
        {
            batchSize = (int)Deep::AutoBatchSize(size, nextSize);
            autoSize = false;
            DynamicThread(batchSize);
        }

        auto l = std::make_unique<DiscriminativePCLayer>(
            size, nextSize, batchSize, lr, ir, pr, lmbda, aType, dType, backend.get());

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }
        layers.push_back(std::move(l));
    }

    void DiscriminativePCNetwork::RandomizeWeights(std::mt19937 &rng)
    {
        for (auto &l : layers)
            l->RandomizeWeights(rng);
    }

    void DiscriminativePCNetwork::ResetState() noexcept
    {
        for (auto &l : layers)
            l->ResetState();
    }

    void DiscriminativePCNetwork::Clamp(const std::vector<float> &input)
    {
        layers.front()->ClampState(input);
    }

    float DiscriminativePCNetwork::CalculateState()
    {
        float e = 0.0f;
        for (auto &l : layers)
            e += l->CalculateState();
        return e;
    }

    void DiscriminativePCNetwork::UpdateState()
    {
        for (auto &l : layers)
            l->UpdateState();
    }

    void DiscriminativePCNetwork::UpdateWeights()
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdateWeights();
    }

    void DiscriminativePCNetwork::UpdatePrecision()
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdatePrecision();
    }

    void DiscriminativePCNetwork::ProjectForward() noexcept
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

    float DiscriminativePCNetwork::TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ProtectFPU();
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

        UpdatePrecision();
        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> DiscriminativePCNetwork::PredictWithProjection(const std::vector<float> &x, int inferenceSteps)
    {
        ProtectFPU();
        ResetState();
        Clamp(x);
        ProjectForward();

        for (int t = 0; t < inferenceSteps; t++)
        {
            CalculateState();
            UpdateState();
        }

        DiscriminativePCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }

    float DiscriminativePCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        GetTerminalLayer()->ClampState(y);

        float finalEnergy = 0.0f;
        for (int t = 0; t < inferenceSteps; t++)
        {
            finalEnergy = CalculateState();
            UpdateState();
        }

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> DiscriminativePCNetwork::Predict(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);

        for (int t = 0; t < inferenceSteps; t++)
        {
            CalculateState();
            UpdateState();
        }

        DiscriminativePCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();

        // The terminal layer's 'size' is its output dimension. Total
        // elements = batchSize * size.
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }

    void DiscriminativePCNetwork::Compile()
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

    bool DiscriminativePCNetwork::Save(const std::string &filename) const noexcept
    {
        return ModelIO::Save(*this, filename);
    }

    bool DiscriminativePCNetwork::Load(const std::string &filename) noexcept
    {
        return ModelIO::Load(*this, filename);
    }
}
