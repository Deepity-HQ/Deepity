#include "PCNetwork.h"

namespace Deep
{
    PCNetwork::~PCNetwork()
    {
        for (auto l : layers)
            delete l;
    }

    void PCNetwork::AddLayer(int size, int nextSize, float lr,
                             void (*act)(float *, size_t), void (*dAct)(float *, size_t))
    {
        if (autoSize && layers.empty())
        {
            batchSize = (int)Deep::AutoBatchSize(size, nextSize);
            autoSize = false;
        }
        PCLayer *l = new PCLayer(size, nextSize, batchSize, lr, act, dAct);
        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l);
            l->SetLayerBelow(layers.back());
        }
        layers.push_back(l);
    }

    void PCNetwork::RandomizeWeights(std::mt19937 &rng)
    {
        for (auto l : layers)
            l->RandomizeWeights(rng);
    }

    void PCNetwork::Clamp(const std::vector<float> &input)
    {
        layers.front()->ClampState(input);
    }

    float PCNetwork::CalculateState()
    {
        float e = 0.0f;
        for (auto l : layers)
            e += l->CalculateState();
        return e;
    }

    void PCNetwork::UpdateState()
    {
        for (auto l : layers)
            l->UpdateState();
    }

    void PCNetwork::UpdateWeights()
    {
        for (size_t i = 0; i + 1 < layers.size(); i++)
            layers[i]->UpdateWeights();
    }
}