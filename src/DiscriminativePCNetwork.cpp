#include <deepity/networks/DiscriminativePCNetwork.h>
#include <deepity/Optimize.h>
#include <deepity/ModelIO.h>
#include <cstring>
#include <cmath>
#include <limits>

namespace Deep
{
    DiscriminativePCNetwork::~DiscriminativePCNetwork()
    {
        for (auto l : layers)
            delete l;

        layers.clear();
    }

    void DiscriminativePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float pr, float lmbda,
                                           void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool))
    {
        if (autoSize && layers.empty())
        {
            batchSize = (int)Deep::AutoBatchSize(size, nextSize);
            autoSize = false;
            DynamicThread(batchSize);
        }

        // Pass raw pointers directly to Constructor 2
        DiscriminativePCLayer *l = new DiscriminativePCLayer(size, nextSize, batchSize, lr, ir, pr, lmbda, act, dAct);

        if (!layers.empty())
        {
            layers.back()->SetLayerAbove(l);
            l->SetLayerBelow(layers.back());
        }
        layers.push_back(l);
    }

    void DiscriminativePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float pr, float lmbda,
                                           Deep::ActivationType aType, Deep::ActivationType dType)
    {
        if (autoSize && layers.empty())
        {
            batchSize = (int)Deep::AutoBatchSize(size, nextSize);
            autoSize = false;
            DynamicThread(batchSize);
        }

        // Pass enums directly to Constructor 1
        DiscriminativePCLayer *l = new DiscriminativePCLayer(size, nextSize, batchSize, lr, ir, pr, lmbda, aType, dType);

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

    void DiscriminativePCNetwork::UpdatePrecision()
    {
        for (size_t i = 0; i + 1 < layers.size(); i++)
            layers[i]->UpdatePrecision();
    }

    float DiscriminativePCNetwork::TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps)

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

    std::vector<float> DiscriminativePCNetwork::Predict(const std::vector<float> &x, int inferenceSteps)
    {

        ResetState();

        Clamp(x);

        for (int t = 0; t < inferenceSteps; t++)

        {

            CalculateState();

            UpdateState();
        }

        DiscriminativePCLayer *terminal = GetTerminalLayer();

        const float *beliefs = terminal->GetBeliefs();

        // The terminal layer's 'size' is its output dimension. Total elements = batchSize * size.

        size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

        return std::vector<float>(beliefs, beliefs + count);
    }

    void DiscriminativePCNetwork::Compile()
    {
        size_t total_floats_needed = 0;

        // 1. Calculate the exact footprint of the entire network
        for (auto *layer : layers)
        {
            total_floats_needed += layer->GetRequiredFloats();
        }

        // 2. Allocate the single contiguous block of memory
        arena = std::make_unique<MemoryArena>(total_floats_needed);

        // 3. Bind every layer sequentially into the arena
        for (auto *layer : layers)
        {
            layer->BindMemory(*arena);
        }
    }

    bool DiscriminativePCNetwork::Save(const std::string &filename) const noexcept
    {
        return ModelIO::Save(*this, filename);
    }

    bool DiscriminativePCNetwork::Load(const std::string &filename) noexcept
    {
        return ModelIO::Load(*this, filename);
    }
}
