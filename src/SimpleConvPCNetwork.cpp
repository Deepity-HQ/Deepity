#include <deepity/networks/SimpleConvPCNetwork.h>

namespace Deep
{
    SimpleConvPCNetwork::SimpleConvPCNetwork(int batchSize) noexcept : batchSize(batchSize) {}

    SimpleConvPCNetwork::~SimpleConvPCNetwork()
    {
        for (auto *l : layers)
            delete l;
    }

    void SimpleConvPCNetwork::AddLayer(int inChannels, int outChannels,
                                       int inHeight, int inWidth,
                                       int kernelH, int kernelW,
                                       int strideH, int strideW,
                                       int padH, int padW,
                                       float lr, float ir, float lmbda,
                                       ActivationType aType, ActivationType dType)
    {
        SimpleConvPCLayer *l = new SimpleConvPCLayer(
            inChannels, outChannels, inHeight, inWidth,
            kernelH, kernelW, strideH, strideW, padH, padW,
            batchSize, lr, ir, lmbda, aType, dType);

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l);
            l->SetLayerBelow(layers.back());
        }
        layers.push_back(l);
    }

    void SimpleConvPCNetwork::SetOptimizer(OptimizerType opt) noexcept
    {
        // Deferred: just records the choice. Actually applied to every
        // layer inside Compile(), BEFORE each layer's GetRequiredFloats()
        // is summed -- GetRequiredFloats() itself depends on opt (Adam
        // buffers only counted if opt is ADAM/ADAMW), so this must happen
        // before the arena is sized, not after.
        pendingOpt = opt;
    }

    void SimpleConvPCNetwork::Compile()
    {
        for (auto *l : layers)
            l->SetOptimizer(pendingOpt);

        size_t total = 0;
        for (auto *l : layers)
            total += l->GetRequiredFloats();

        arena = std::make_unique<MemoryArena>(total);
        for (auto *l : layers)
            l->BindMemory(*arena);
    }

    void SimpleConvPCNetwork::RandomizeWeights(std::mt19937 &rng) noexcept
    {
        for (auto *l : layers)
            l->RandomizeWeights(rng);
    }

    void SimpleConvPCNetwork::ResetState() noexcept
    {
        for (auto *l : layers)
            l->ResetState();
    }

    void SimpleConvPCNetwork::Clamp(const std::vector<float> &input) noexcept
    {
        layers.front()->ClampState(input);
    }

    float SimpleConvPCNetwork::CalculateState() noexcept
    {
        float e = 0.0f;
        for (auto *l : layers)
            e += l->CalculateState();
        return e;
    }

    void SimpleConvPCNetwork::UpdateState() noexcept
    {
        for (auto *l : layers)
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
        float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        return std::vector<float>(beliefs, beliefs + count);
    }
}