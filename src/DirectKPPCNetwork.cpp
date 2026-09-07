#include <deepity/networks/DirectKPPCNetwork.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <iostream>
#include <cstdio>

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

            const float *mu = layers[i]->GetMu();
            float *nextZ = layers[i + 1]->GetBeliefs();
            size_t n = layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

            // Was: std::memcpy(nextZ, mu, n * sizeof(float)) -- wrong if
            // mu/nextZ are device pointers (DEVICE_GPU). backend->Copy()
            // is device-to-device, matching SimplePCNetwork's own fix
            // for the identical bug.
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
        return GetTerminalLayer()->CalculateState(false); // only the error buffer matters here
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
        printf("TrainStep: device=%s, graphCaptured=%d\n",
               (device == DeviceType::DEVICE_GPU ? "GPU" : "CPU"), (int)graphCaptured);
        fflush(stdout);
        ResetState();
        Clamp(x);
        ProjectForward();
        GetTerminalLayer()->ClampState(y);

        if (device == DeviceType::DEVICE_GPU)
        {
            if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
            {
                backend->BeginGraphCapture();
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
                CalculateTerminalError();
                DirectFeedbackUpdate();
                for (int t = 0; t < inferenceSteps; t++)
                    Step(false);
                UpdateWeights();
            }
        }
        else
        {
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

        // Was: std::vector<float>(beliefs, beliefs + count) -- the
        // iterator-range constructor dereferences every element
        // directly, wrong if beliefs is a device pointer. Allocate the
        // host-side result first, then copy it out through the backend.
        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }

    void DirectKPPCNetwork::Compile()
    {
#pragma omp parallel
        { // Broadcast FTZ/DAZ hardware flags to ALL OpenMP worker threads
            _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
            _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
        }
        size_t total_floats_needed = 0;
        for (auto &layer : layers)
            total_floats_needed += layer->GetRequiredFloats();

        if (device == DeviceType::DEVICE_CPU)
        {
            cpuArena = std::make_unique<MemoryArena>(total_floats_needed, false); // TODO: Consider adding huge pages as a param
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