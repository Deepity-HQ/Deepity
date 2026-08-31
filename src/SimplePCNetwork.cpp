#include <deepity/networks/SimplePCNetwork.h>
#include <pmmintrin.h>
#include <xmmintrin.h>
#include <omp.h>

namespace Deep
{
    SimplePCNetwork::SimplePCNetwork(int batchSize) noexcept : batchSize(batchSize) {}

    void SimplePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                                   void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool))
    {
        std::unique_ptr<SimplePCLayer> l = std::make_unique<SimplePCLayer>(size, nextSize, batchSize, lr, ir, lmbda, act, dAct);

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }
        layers.push_back(std::move(l));
    }

    void SimplePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                                   ActivationType aType, ActivationType dType)
    {
        std::unique_ptr<SimplePCLayer> l = std::make_unique<SimplePCLayer>(size, nextSize, batchSize, lr, ir, lmbda, aType, dType);

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }
        layers.push_back(std::move(l));
    }

    void SimplePCNetwork::RandomizeWeights(std::mt19937 &rng)
    {
        for (auto &l : layers)
            l->RandomizeWeights(rng);
    }

    void SimplePCNetwork::ResetState() noexcept
    {
        for (auto &l : layers)
            l->ResetState();
    }

    void SimplePCNetwork::Clamp(const std::vector<float> &input)
    {
        layers.front()->ClampState(input);
    }

    float SimplePCNetwork::CalculateState()
    {
        float e = 0.0f;
        for (size_t i = 0; i < layers.size(); i++)
            e += layers[i]->CalculateState();
        return e;
    }

    void SimplePCNetwork::UpdateState()
    {
        for (auto &l : layers)
            l->UpdateState();
    }

    void SimplePCNetwork::UpdateWeights()
    {
        for (size_t i = 0; i + 1 < layers.size(); i++)
            layers[i]->UpdateWeights();
    }

    float SimplePCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
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

    std::vector<float> SimplePCNetwork::Predict(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);

        for (int t = 0; t < inferenceSteps; t++)
        {
            CalculateState();
            UpdateState();
        }

        SimplePCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        return std::vector<float>(beliefs, beliefs + count);
    }

	void SimplePCNetwork::ProjectForward() noexcept
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

    float SimplePCNetwork::TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
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

    std::vector<float> SimplePCNetwork::PredictWithProjection(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        ProjectForward(); // Add your forward projection initialization here

        for (int t = 0; t < inferenceSteps; t++)
        {
            CalculateState();
            UpdateState();
        }

        SimplePCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        return std::vector<float>(beliefs, beliefs + count);
    }

    void SimplePCNetwork::SetMuCacheThreshold(float threshold) noexcept
    {
        for (auto &l : layers)
            l->SetMuCacheThreshold(threshold);
    }

    void SimplePCNetwork::Compile()
    {
#pragma omp parallel 
	{ // Broadcast FTZ/DAZ hardware flags to ALL OpenMP worker threads
		_MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
		_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
	}
        size_t total_floats_needed = 0;
        for (auto &layer : layers)
            total_floats_needed += layer->GetRequiredFloats();

        arena = std::make_unique<MemoryArena>(total_floats_needed, true);
        for (auto &layer : layers)
            layer->BindMemory(*arena);
    }
}
