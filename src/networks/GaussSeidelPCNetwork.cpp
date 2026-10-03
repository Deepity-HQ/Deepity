#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <deepity/backend/Backend.h>
#include <iostream>

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

    float GaussSeidelPCNetwork::Step(bool needEnergy) noexcept
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
        // starting from this point. needEnergy=false still correctly
        // updates every layer's `e` (required for the next step's
        // dynamics), just skips the energy value itself -- on
        // CUDABackend that also means no blocking host sync, required
        // when this runs inside a captured CUDA graph region.
        float totalEnergy = 0.0f;
        for (auto &l : layers)
            totalEnergy += l->ComputeError(needEnergy);

        return needEnergy ? totalEnergy : 0.0f;
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

        if (device == DeviceType::DEVICE_GPU)
        {
            if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
            {
                backend->BeginGraphCapture();
                for (int t = 0; t < inferenceSteps; ++t)
                    Step(false);
                UpdateWeights();
                bool captureOk = backend->EndGraphCapture();

                if (captureOk)
                {
                    graphCaptured = true;
                    capturedInferenceSteps = inferenceSteps;
                }
                else
                {
                    std::cerr << "Graph capture failed, falling back to non-graph execution for this call.\n";
                }
            }

            if (graphCaptured)
            {
                backend->ReplayGraph();
            }
            else
            {
                for (int t = 0; t < inferenceSteps; ++t)
                    Step(false);
                UpdateWeights();
            }
        }
        else
        {
            for (int t = 0; t < inferenceSteps; ++t)
                Step(false);
            UpdateWeights();
        }

        // Fresh, uncaptured energy readout: sweeps 2+3 of Step() (recompute
        // predictions/errors from the CURRENT z), deliberately without
        // sweep 1's UpdateState() -- calling the full Step() again here
        // would apply an extra, unintended settling step beyond
        // inferenceSteps, on top of the same stale-return-value-under-
        // graph-capture concern every other ported network has.
        float finalEnergy = 0.0f;
        for (auto &l : layers)
            l->ComputePrediction();
        for (auto &l : layers)
            finalEnergy += l->ComputeError();

        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    float GaussSeidelPCNetwork::TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        // Clamped BEFORE ProjectForward() (reordered from the original
        // ProjectForward()-then-clamp): required so ProjectForward()'s own
        // IsClamped() guard protects the terminal layer, and so
        // ProjectForward() can safely move inside the captured region
        // below -- see ConvPCNetwork::TrainStepWithProjection()'s identical
        // comment for why. The end result for z3/mu3 (and therefore the
        // immediate ComputeError() below) is unchanged either way:
        // ComputePrediction() for the second-to-last layer still runs
        // unconditionally, only the backend->Copy into an ALREADY-clamped
        // terminal's z is what gets skipped.
        GetTerminalLayer()->ClampState(y);

        if (device == DeviceType::DEVICE_GPU)
        {
            if (!graphCapturedWithProjection || capturedInferenceStepsWithProjection != inferenceSteps)
            {
                backend->BeginGraphCapture();
                ProjectForward();
                // Only the terminal layer's error is computed immediately
                // (e3 = z3 - mu3, target vs. projected prediction); hidden
                // layers stay at zero. needEnergy=false: this return value
                // is discarded either way, and the call must not
                // synchronously read back from the GPU while this stream
                // is being captured.
                GetTerminalLayer()->ComputeError(false);
                for (int t = 0; t < inferenceSteps; ++t)
                    Step(false);
                UpdateWeights();
                bool captureOk = backend->EndGraphCapture();

                if (captureOk)
                {
                    graphCapturedWithProjection = true;
                    capturedInferenceStepsWithProjection = inferenceSteps;
                }
                else
                {
                    std::cerr << "Graph capture failed, falling back to non-graph execution for this call.\n";
                }
            }

            if (graphCapturedWithProjection)
            {
                backend->ReplayGraph();
            }
            else
            {
                ProjectForward();
                GetTerminalLayer()->ComputeError(false);
                for (int t = 0; t < inferenceSteps; ++t)
                    Step(false);
                UpdateWeights();
            }
        }
        else
        {
            ProjectForward();
            GetTerminalLayer()->ComputeError(false);
            for (int t = 0; t < inferenceSteps; ++t)
                Step(false);
            UpdateWeights();
        }

        float finalEnergy = 0.0f;
        for (auto &l : layers)
            l->ComputePrediction();
        for (auto &l : layers)
            finalEnergy += l->ComputeError();

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
