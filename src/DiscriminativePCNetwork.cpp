#include "DiscriminativePCNetwork.h"

namespace Deep
{
    DiscriminativePCNetwork::~DiscriminativePCNetwork()
    {
        for (auto l : layers)
            delete l;
    }

    void DiscriminativePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                                           void (*act)(float *, size_t), void (*dAct)(float *, size_t))
    {
        if (autoSize && layers.empty())
        {
            batchSize = (int)Deep::AutoBatchSize(size, nextSize);
            autoSize = false;
        }
        DiscriminativePCLayer *l = new DiscriminativePCLayer(size, nextSize, batchSize, lr, ir, lmbda, act, dAct);
        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l);
            l->SetLayerBelow(layers.back());
        }
        layers.push_back(l);
    }

    void DiscriminativePCNetwork::RandomizeWeights(std::mt19937 &rng)
    {
        for (auto l : layers)
            l->RandomizeWeights(rng);
    }

    void DiscriminativePCNetwork::ResetState() noexcept
    {
        for (auto l : layers)
            l->ResetState();
    }

    void DiscriminativePCNetwork::Clamp(const std::vector<float> &input)
    {
        layers.front()->ClampState(input);
    }

    float DiscriminativePCNetwork::CalculateState()
    {
        float e = 0.0f;
        for (size_t i = 0; i < layers.size(); i++)
            e += layers[i]->CalculateState();
        return e;
    }

    void DiscriminativePCNetwork::UpdateState()
    {
        for (auto l : layers)
            l->UpdateState();
    }

    void DiscriminativePCNetwork::UpdateWeights()
    {
        for (size_t i = 0; i + 1 < layers.size(); i++)
            layers[i]->UpdateWeights();
    }
}