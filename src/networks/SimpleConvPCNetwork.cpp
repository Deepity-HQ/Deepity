#include <deepity/networks/SimpleConvPCNetwork.h>
#include <deepity/backend/Backend.h>
#include <iostream>


namespace Deep
{
    SimpleConvPCNetwork::SimpleConvPCNetwork(int batchSize, DeviceType device) noexcept
        : device(device), batchSize(batchSize)
    {
        backend = CreateBackend(device);
    }

    void SimpleConvPCNetwork::AddLayer(int inChannels, int outChannels,
                                       int inHeight, int inWidth,
                                       int kernelH, int kernelW,
                                       int strideH, int strideW,
                                       int padH, int padW,
                                       float lr, float ir, float lmbda,
                                       ActivationType aType, ActivationType dType)
    {
        auto l = std::make_unique<SimpleConvPCLayer>(
            inChannels, outChannels, inHeight, inWidth,
            kernelH, kernelW, strideH, strideW, padH, padW,
            batchSize, lr, ir, lmbda, aType, dType, backend.get());

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }
        layers.push_back(std::move(l));
    }

    void SimpleConvPCNetwork::SetOptimizer(OptimizerType opt) noexcept
    {
        // Applied to every layer inside Compile() instead of immediately,
        // so it works even when called before any AddLayer().
        pendingOpt = opt;
    }

    void SimpleConvPCNetwork::Compile()
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

    void SimpleConvPCNetwork::RandomizeWeights(std::mt19937 &rng) noexcept
    {
        for (auto &l : layers)
            l->RandomizeWeights(rng);
    }

    void SimpleConvPCNetwork::ResetState() noexcept
    {
        for (auto &l : layers)
            l->ResetState();
    }

    void SimpleConvPCNetwork::Clamp(const std::vector<float> &input) noexcept
    {
        layers.front()->ClampState(input);
    }

    float SimpleConvPCNetwork::CalculateState() noexcept
    {
        float e = 0.0f;
        for (auto &l : layers)
            e += l->CalculateState();
        return e;
    }

    void SimpleConvPCNetwork::UpdateState() noexcept
    {
        for (auto &l : layers)
            l->UpdateState();
    }

    void SimpleConvPCNetwork::UpdateWeights() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdateWeights();
    }

    float SimpleConvPCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
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
                    CalculateState();
                    UpdateState();
                }
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
                {
                    CalculateState();
                    UpdateState();
                }
                UpdateWeights();
            }
        }
        else
        {
            for (int t = 0; t < inferenceSteps; ++t)
            {
                CalculateState();
                UpdateState();
            }
            UpdateWeights();
        }

        float finalEnergy = CalculateState();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> SimpleConvPCNetwork::Predict(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);

        for (int t = 0; t < inferenceSteps; ++t)
        {
            CalculateState();
            UpdateState();
        }

        SimpleConvPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }

    void SimpleConvPCNetwork::ProjectForward() noexcept
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

    float SimpleConvPCNetwork::TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
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
                    CalculateState();
                    UpdateState();
                }
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
                for (int t = 0; t < inferenceSteps; ++t)
                {
                    CalculateState();
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
                CalculateState();
                UpdateState();
            }
            UpdateWeights();
        }

        float finalEnergy = CalculateState();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> SimpleConvPCNetwork::PredictWithProjection(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        ProjectForward();

        for (int t = 0; t < inferenceSteps; ++t)
        {
            CalculateState();
            UpdateState();
        }

        SimpleConvPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        std::vector<float> result(count);
        backend->CopyToHost(result.data(), beliefs, count);
        return result;
    }
}