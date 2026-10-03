#include <deepity/networks/ConvPCNetwork.h>
#include <deepity/backend/Backend.h>
#include <iostream>

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

    float ConvPCNetwork::CalculateState(bool needEnergy) noexcept
    {
        float e = 0.0f;
        for (auto &l : layers)
            e += l->CalculateState(needEnergy);
        return needEnergy ? e : 0.0f;
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

        if (device == DeviceType::DEVICE_GPU)
        {
            if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
            {
                backend->BeginGraphCapture();
                for (int t = 0; t < inferenceSteps; ++t)
                {
                    // needEnergy=false: ComputePrecisionWeightedErrorAndEnergy's
                    // synchronous GPU readback is illegal while this stream is
                    // being captured. The real energy read happens in the
                    // fresh, uncaptured call at the end of this function.
                    CalculateState(false);
                    UpdateState();
                }
                UpdateWeights();
                bool captureOk = backend->EndGraphCapture();

                if (captureOk)
                {
                    graphCaptured = true;
                    capturedInferenceSteps = inferenceSteps;

                    // The settling loop just recorded above ran as real C++,
                    // leaving muCacheValid=true whenever it last touched a
                    // clamped layer. Nothing will reset that before the fresh
                    // CalculateState() read below, so it would wrongly reuse
                    // cachedMu (computed from this call's PRE-update weights,
                    // the last time the loop's recording touched it) instead
                    // of recomputing from the weights ReplayGraph() is about
                    // to produce. See ConvPCLayer::InvalidateMuCache().
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
                // cudaGraphLaunch() only enqueues the replay; the fresh
                // energy readout below needs it to have actually finished.
                // Same-stream ordering should guarantee this without an
                // explicit wait, but this costs nothing extra (the
                // ...AndEnergy() call right after does its own blocking
                // sync regardless) and removes any doubt.
                backend->Synchronize();
            }
            else
            {
                for (int t = 0; t < inferenceSteps; ++t)
                {
                    CalculateState(false);
                    UpdateState();
                }
                UpdateWeights();
            }
        }
        else
        {
            for (int t = 0; t < inferenceSteps; ++t)
            {
                CalculateState(false);
                UpdateState();
            }
            UpdateWeights();
        }

        // A fresh, uncaptured call: a value returned by a call made
        // INSIDE the captured region above would be stale on every
        // ReplayGraph() after the first (only the GPU kernel launches
        // are replayed, not the host-side C++ call that returned this
        // float), so the real energy readout always happens out here,
        // identically for both the CPU and GPU paths.
        float finalEnergy = CalculateState();
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
        // Clamped BEFORE ProjectForward() (reordered from the original
        // ProjectForward()-then-clamp), required for two reasons: (1) so
        // ProjectForward()'s own IsClamped() guard actually protects the
        // terminal layer from being overwritten by a forward-projected
        // guess, and (2) so ProjectForward() can safely move inside the
        // captured region below -- ClampState()'s CopyFromHost(z, y.data(),
        // ...) call captures a SPECIFIC host pointer; replaying that
        // inside a captured graph with a different batch's `y` (a
        // different vector, different address, every call) would copy
        // stale data. Keeping it out here means it genuinely re-executes
        // with fresh data on every call, same pattern DirectKPPCNetwork
        // already uses.
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
                UpdateWeights();
                bool captureOk = backend->EndGraphCapture();

                if (captureOk)
                {
                    graphCapturedWithProjection = true;
                    capturedInferenceStepsWithProjection = inferenceSteps;

                    // See the identical fix in TrainStep() above: the
                    // recording pass above just ran this settling loop as
                    // real C++, leaving muCacheValid=true behind for any
                    // clamped layer it touched last.
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
            UpdateWeights();
        }

        float finalEnergy = CalculateState();
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
