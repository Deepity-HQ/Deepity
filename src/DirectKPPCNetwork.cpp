#include <deepity/networks/DirectKPPCNetwork.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <iostream>

namespace Deep
{
    DirectKPPCNetwork::DirectKPPCNetwork(int batchSize, DeviceType device) noexcept
        : device(device), batchSize(batchSize)
    {
        backend = CreateBackend(device);
    }

    void DirectKPPCNetwork::AddLayer(size_t size, size_t nextSize, size_t terminalSize,
                                     float lr, float ir, float fl, float lmbda,
                                     ActivationType aType, ActivationType dType)
    {
        std::unique_ptr<DirectKPPCLayer> l = std::make_unique<DirectKPPCLayer>(
            size, nextSize, terminalSize, batchSize, lr, ir, fl, lmbda, aType, dType, backend.get());

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }

        layers.push_back(std::move(l));
    }

    void DirectKPPCNetwork::RandomizeWeights(std::mt19937 &rng)
    {
        for (auto &l : layers)
            l->RandomizeWeights(rng);
    }

    void DirectKPPCNetwork::ResetState() noexcept
    {
        for (auto &l : layers)
            l->ResetState();
    }

    void DirectKPPCNetwork::Clamp(const std::vector<float> &input)
    {
        layers.front()->ClampState(input);
    }

    void DirectKPPCNetwork::ProjectForward() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
        {
            layers[i]->ComputeMuOnly();

            // Skip a layer that's already clamped (the terminal layer,
            // once ClampState(y) has run) -- overwriting its real target
            // with a forward-projected guess is exactly the bug
            // SimplePCNetwork::ProjectForward() had, fixed here before
            // it gets exercised for the first time by moving this call
            // inside graph capture, which requires ClampState(y) to run
            // BEFORE this, not after, for the capture ordering to work.
            if (layers[i + 1]->IsClamped())
                continue;

            const float *mu = layers[i]->GetMu();
            float *nextZ = layers[i + 1]->GetBeliefs();
            size_t n = layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

            backend->Copy(nextZ, mu, n);
        }
    }

    void DirectKPPCNetwork::DirectFeedbackUpdate() noexcept
    {
        for (size_t i = 0; i < layers.size() - 1; ++i)
            layers[i]->DirectFeedbackUpdate();
    }

    float DirectKPPCNetwork::CalculateTerminalError() noexcept
    {
        return GetTerminalLayer()->CalculateState(false);
    }

    float DirectKPPCNetwork::Step(bool needEnergy) noexcept
    {
        float e = 0.0f;
        for (auto &l : layers)
            e += l->CalculateState(needEnergy);
        for (auto &l : layers)
            l->UpdateState();
        return needEnergy ? e : 0.0f;
    }

    void DirectKPPCNetwork::UpdateWeights() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); i++)
            layers[i]->UpdateWeights();
    }

    float DirectKPPCNetwork::TrainStep(const std::vector<float> &x,
                                       const std::vector<float> &y,
                                       int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        // Moved BEFORE ProjectForward() -- required for ProjectForward's
        // new IsClamped() guard to actually protect the terminal layer,
        // and required so ProjectForward() can safely move inside the
        // captured region below.
        GetTerminalLayer()->ClampState(y);

        if (device == DeviceType::DEVICE_GPU)
        {
            if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
            {
                backend->BeginGraphCapture();
                ProjectForward(); // now inside capture -- see note above
                CalculateTerminalError();
                DirectFeedbackUpdate();
                for (int t = 0; t < inferenceSteps; t++)
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
                    std::cerr << "Graph capture failed -- falling back to non-graph execution for this call.\n";
                }
            }

            if (graphCaptured)
            {
                backend->ReplayGraph();
            }
            else
            {
                ProjectForward();
                CalculateTerminalError();
                DirectFeedbackUpdate();
                for (int t = 0; t < inferenceSteps; t++)
                    Step(false);
                UpdateWeights();
            }
        }
        else
        {
            ProjectForward();
            CalculateTerminalError();
            DirectFeedbackUpdate();
            for (int t = 0; t < inferenceSteps; t++)
                Step(false);
            UpdateWeights();
        }

        float finalEnergy = 0.0f;
        for (auto &l : layers)
            finalEnergy += l->CalculateState(true);

        GetTerminalLayer()->UnclampState();
        return finalEnergy;
    }

    std::vector<float> DirectKPPCNetwork::Predict(const std::vector<float> &x,
                                                  int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        ProjectForward();

        for (int t = 0; t < inferenceSteps; t++)
        {
            Step();
        }

        DirectKPPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }

    void DirectKPPCNetwork::Compile()
    {
#pragma omp parallel
        {
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        }
        size_t total_floats_needed = 0;
        for (auto &layer : layers)
            total_floats_needed += layer->GetRequiredFloats();

        if (device == DeviceType::DEVICE_CPU)
        {
            cpuArena = std::make_unique<MemoryArena>(total_floats_needed, false);
            for (auto &layer : layers)
            {
                layer->BindMemory(*cpuArena);
                layer->SetTerminalLayer(layers.back().get());
            }
        }
#if defined(DEEPITY_USE_CUDA)
        else
        {
            backend->PrepareForBatchSize(batchSize);
            gpuArena = std::make_unique<DeviceMemoryArena>(backend.get(), total_floats_needed);
            for (auto &layer : layers)
            {
                layer->BindMemory(*gpuArena);
                layer->SetTerminalLayer(layers.back().get());
            }
        }
#endif
    }
}