#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/backend/Backend.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/ModelIO.h>
#include <iostream>
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

    float DiscriminativePCNetwork::CalculateState(bool needEnergy)
    {
        float e = 0.0f;
        for (auto &l : layers)
            e += l->CalculateState(needEnergy);
        return needEnergy ? e : 0.0f;
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
        // Clamped BEFORE ProjectForward() (reordered from the original
        // ProjectForward()-then-clamp): required so ProjectForward()'s own
        // IsClamped() guard protects the terminal layer, and so
        // ProjectForward() can safely move inside the captured region
        // below -- see ConvPCNetwork::TrainStepWithProjection()'s identical
        // comment for why.
        GetTerminalLayer()->ClampState(y);

        if (device == DeviceType::DEVICE_GPU)
        {
            if (!graphCapturedWithProjection || capturedInferenceStepsWithProjection != inferenceSteps)
            {
                backend->BeginGraphCapture();
                ProjectForward();
                for (int t = 0; t < inferenceSteps; ++t)
                {
                    CalculateState(false);
                    UpdateState();
                }
                UpdatePrecision();
                UpdateWeights();
                bool captureOk = backend->EndGraphCapture();

                if (captureOk)
                {
                    graphCapturedWithProjection = true;
                    capturedInferenceStepsWithProjection = inferenceSteps;

                    // The recording pass above just ran this settling loop
                    // as real C++, leaving muCacheValid=true behind for any
                    // clamped layer it touched last; nothing resets that
                    // before the fresh energy read below, which would
                    // otherwise wrongly reuse mu from before this call's
                    // own weight update (see ConvPCNetwork::TrainStep()'s
                    // identical bug).
                    for (auto &l : layers)
                        l->InvalidateMuCache();
                }
                else
                {
                    std::cerr << "Graph capture failed, falling back to non-graph execution for this call.\n";
                }
            }

            if (graphCapturedWithProjection)
            {
                backend->ReplayGraph();
                backend->Synchronize();
            }
            else
            {
                ProjectForward();
                for (int t = 0; t < inferenceSteps; ++t)
                {
                    CalculateState(false);
                    UpdateState();
                }
                UpdatePrecision();
                UpdateWeights();
            }
        }
        else
        {
            ProjectForward();
            for (int t = 0; t < inferenceSteps; ++t)
            {
                CalculateState(false);
                UpdateState();
            }
            UpdatePrecision();
            UpdateWeights();
        }

        float finalEnergy = CalculateState();
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

        if (device == DeviceType::DEVICE_GPU)
        {
            if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
            {
                backend->BeginGraphCapture();
                for (int t = 0; t < inferenceSteps; t++)
                {
                    CalculateState(false);
                    UpdateState();
                }
                UpdateWeights();
                bool captureOk = backend->EndGraphCapture();

                if (captureOk)
                {
                    graphCaptured = true;
                    capturedInferenceSteps = inferenceSteps;

                    // The recording pass above just ran this settling loop
                    // as real C++, leaving muCacheValid=true behind for any
                    // clamped layer it touched last; nothing resets that
                    // before the fresh energy read below, which would
                    // otherwise wrongly reuse mu from before this call's
                    // own weight update (see ConvPCNetwork::TrainStep()'s
                    // identical bug).
                    for (auto &l : layers)
                        l->InvalidateMuCache();
                }
                else
                {
                    std::cerr << "Graph capture failed, falling back to non-graph execution for this call.\n";
                }
            }

            if (graphCaptured)
            {
                backend->ReplayGraph();
                backend->Synchronize();
            }
            else
            {
                for (int t = 0; t < inferenceSteps; t++)
                {
                    CalculateState(false);
                    UpdateState();
                }
                UpdateWeights();
            }
        }
        else
        {
            for (int t = 0; t < inferenceSteps; t++)
            {
                CalculateState(false);
                UpdateState();
            }
            UpdateWeights();
        }

        float finalEnergy = CalculateState();
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
