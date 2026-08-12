#include "ConvPCNetwork.h"

namespace Deep
{
    ConvPCNetwork::ConvPCNetwork(int batchSize) noexcept : batchSize(batchSize) {}

    ConvPCNetwork::~ConvPCNetwork()
    {
        for (auto *l : layers)
            delete l;
    }

    void ConvPCNetwork::AddLayer(int inChannels, int outChannels,
                                 int inHeight, int inWidth,
                                 int kernelH, int kernelW,
                                 int strideH, int strideW,
                                 int padH, int padW,
                                 float lr, float ir, float pr, float lmbda,
                                 ActivationType aType, ActivationType dType)
    {
        ConvPCLayer *l = new ConvPCLayer(
            inChannels, outChannels, inHeight, inWidth,
            kernelH, kernelW, strideH, strideW, padH, padW,
            batchSize, lr, ir, pr, lmbda, aType, dType);

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l);
            l->SetLayerBelow(layers.back());
        }
        layers.push_back(l);
    }

    void ConvPCNetwork::Compile()
    {
        size_t total = 0;
        for (auto *l : layers)
            total += l->GetRequiredFloats();

        arena = std::make_unique<MemoryArena>(total);
        for (auto *l : layers)
            l->BindMemory(*arena);
    }

    void ConvPCNetwork::RandomizeWeights(std::mt19937 &rng) noexcept
    {
        for (auto *l : layers)
            l->RandomizeWeights(rng);
    }

    void ConvPCNetwork::ResetState() noexcept
    {
        for (auto *l : layers)
            l->ResetState();
    }

    void ConvPCNetwork::Clamp(const std::vector<float> &input) noexcept
    {
        layers.front()->ClampState(input);
    }

    float ConvPCNetwork::CalculateState() noexcept
    {
        float e = 0.0f;
        for (auto *l : layers)
            e += l->CalculateState();
        return e;
    }

    void ConvPCNetwork::UpdateState() noexcept
    {
        for (auto *l : layers)
            l->UpdateState();
    }

    void ConvPCNetwork::UpdateWeights() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdateWeights();
    }

    void ConvPCNetwork::UpdatePrecision() noexcept
    {
        for (size_t i = 0; i + 1 < layers.size(); ++i)
            layers[i]->UpdatePrecision();
    }

    float ConvPCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)
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
        float *beliefs = terminal->GetBeliefs();
        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        return std::vector<float>(beliefs, beliefs + count);
    }
}
