#include <deepity/networks/DirectKPPCNetwork.h>
#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace Deep
{
    DirectKPPCNetwork::DirectKPPCNetwork(int batchSize) noexcept
        : batchSize(batchSize) {}

    void DirectKPPCNetwork::AddLayer(size_t size, size_t nextSize, size_t terminalSize,
                                     float lr, float ir, float fl, float lmbda,
                                     ActivationType aType, ActivationType dType)
    {
        std::unique_ptr<DirectKPPCLayer> l = std::make_unique<DirectKPPCLayer>(size, nextSize, terminalSize, batchSize, lr, ir, fl, lmbda, aType, dType);

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

            std::memcpy(nextZ, mu, n * sizeof(float));
        }
    }

    float DirectKPPCNetwork::CalculateTerminalError() noexcept
    {
        return GetTerminalLayer()->CalculateState();
    }

    void DirectKPPCNetwork::DirectFeedbackUpdate() noexcept
    {
        for (size_t i = 0; i < layers.size() - 1; ++i)
            layers[i]->DirectFeedbackUpdate();
    }

    float DirectKPPCNetwork::Step() noexcept
    {
        float e = 0.0f;
        for (auto &l : layers)
            e += l->CalculateState();
        for (auto &l : layers)
            l->UpdateState();
        return e;
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
        ProjectForward();
        GetTerminalLayer()->ClampState(y);

        for (int t = 0; t < inferenceSteps; t++)
            Step();

        // Sync e/mu/zF to the TRUE final z before UpdateWeights() reads them.
        // Step()'s own CalculateState() each iteration reflects z BEFORE that
        // iteration's UpdateState() moves it, so after the loop exits, e/mu/zF
        // are one iteration stale relative to the final z -- exactly the bug
        // the test's added TotalEnergy() call worked around. This also gives a
        // more accurate finalEnergy for free, computed at the true final state.
        float finalEnergy = 0.0f;
        for (auto &l : layers)
            finalEnergy += l->CalculateState();

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    std::vector<float> DirectKPPCNetwork::Predict(const std::vector<float> &x,
                                                  int inferenceSteps)
    {
        ResetState();
        Clamp(x);

        for (int t = 0; t < inferenceSteps; t++)
        {
            Step();
        }

        DirectKPPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        return std::vector<float>(beliefs, beliefs + count);
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

        arena = std::make_unique<MemoryArena>(total_floats_needed, false); // TODO: Consider adding huge pages as a param

        for (auto &layer : layers)
        {
            layer->BindMemory(*arena);
            layer->SetTerminalLayer(layers.back().get());
        }
    }
}