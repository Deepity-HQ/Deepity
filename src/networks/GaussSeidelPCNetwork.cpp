#include <deepity/networks/GaussSeidelPCNetwork.h>
#include <cstring>

namespace Deep
{
    GaussSeidelPCNetwork::GaussSeidelPCNetwork(int batchSize) noexcept : batchSize(batchSize) {}

    void GaussSeidelPCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                                         void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool))
    {
        std::unique_ptr<GaussSeidelPCLayer> l = std::make_unique<GaussSeidelPCLayer>(size, nextSize, batchSize, lr, ir, lmbda, act, dAct);

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l.get());
            l->SetLayerBelow(layers.back().get());
        }
        layers.push_back(std::move(l));
    }

    void GaussSeidelPCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                                         ActivationType aType, ActivationType dType)
    {
        std::unique_ptr<GaussSeidelPCLayer> l = std::make_unique<GaussSeidelPCLayer>(size, nextSize, batchSize, lr, ir, lmbda, aType, dType);

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

    float GaussSeidelPCNetwork::Step() noexcept
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
        // starting from this point.
        float totalEnergy = 0.0f;
        for (auto &l : layers)
        {
            float e = l->ComputeError();
            totalEnergy += e;
        }

        return totalEnergy;
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

            const float *mu = layers[i]->GetMu();
            float *nextZ = layers[i + 1]->GetBeliefs();
            size_t n = (size_t)layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

            std::memcpy(nextZ, mu, n * sizeof(float));
        }
    }

    float GaussSeidelPCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        GetTerminalLayer()->ClampState(y);

        float finalEnergy = 0.0f;
        for (int t = 0; t < inferenceSteps; ++t)
            finalEnergy = Step();

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

    float GaussSeidelPCNetwork::TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
    {
        ResetState();
        Clamp(x);
        ProjectForward();
        GetTerminalLayer()->ClampState(y);

        // ONLY the terminal layer's error is computed immediately (e3 =
        // z3 - mu3, using the clamped target against the projected
        // prediction) -- confirmed directly from ngc-learn's own official
        // documentation: "error values... at initialization, are e1=0,
        // e2=0, and e3=z3-mu3". Hidden layers correctly STAY at zero,
        // matching ResetState()'s default -- an earlier version of this
        // fix incorrectly computed ALL layers' errors immediately, which
        // contradicts the official spec.
        GetTerminalLayer()->ComputeError();

        float finalEnergy = 0.0f;
        for (int t = 0; t < inferenceSteps; ++t)
            finalEnergy = Step();

        UpdateWeights();
        GetTerminalLayer()->UnclampState();

        return finalEnergy;
    }

std::vector<float> GaussSeidelPCNetwork::Predict(const std::vector<float> &x, int inferenceSteps)
    {
        ResetState();
        Clamp(x);

        // FIXED: Seed the hidden states with the forward pass!
        // Without this, inference starts from z=0 and fails to reach the target.
        ProjectForward();

        // Note: ngc-learn evaluates accuracy on the projection BEFORE settling.
        // If you want to match their 95.09% exactly, you can even pass steps=0 
        // at test time in Python. But settling from the projection works great too.
        for (int t = 0; t < inferenceSteps; ++t)
            Step();

        GaussSeidelPCLayer *terminal = GetTerminalLayer();
        const float *beliefs = terminal->GetBeliefs();
        size_t count = (size_t)terminal->GetBatchSize() * terminal->GetInputSize();

        return std::vector<float>(beliefs, beliefs + count);
    }

    void GaussSeidelPCNetwork::Compile()
    {
        size_t total_floats_needed = 0;
        for (auto &layer : layers)
            total_floats_needed += layer->GetRequiredFloats();

        arena = std::make_unique<MemoryArena>(total_floats_needed);
        for (auto &layer : layers)
            layer->BindMemory(*arena);
    }
}
