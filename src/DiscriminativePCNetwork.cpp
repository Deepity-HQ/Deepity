#include "DiscriminativePCNetwork.h"
#include "Optimize.h"
#include <cstring>

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
        DiscriminativePCLayer *l = new DiscriminativePCLayer(size, nextSize, batchSize, lr, ir, pr, lmbda, act, dAct);
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

    bool DiscriminativePCNetwork::Save(const std::string &filename) const noexcept
    {
        std::ofstream of(filename, std::ios::binary);
        if (!of)
            return false;

        const char magic[8] = "D33P1TY";
        of.write(magic, sizeof(magic));

        uint32_t version = 1;
        of.write(reinterpret_cast<const char *>(&version), sizeof(version));

        uint32_t layerCount = static_cast<uint32_t>(layers.size());
        of.write(reinterpret_cast<const char *>(&layerCount), sizeof(layerCount));

        for (const DiscriminativePCLayer *layer : layers)
        {
            uint32_t inputSize = static_cast<uint32_t>(layer->GetInputSize());
            uint32_t outputSize = static_cast<uint32_t>(layer->GetOutputSize());
            uint32_t batchSize = static_cast<uint32_t>(layer->GetBatchSize());

            float lr = layer->GetLearningRate();
            float ir = layer->GetInferenceRate();
            float pr = layer->GetPrecisionRate();
            float lambda = layer->GetLambda();

            ActivationType activation = layer->GetActivationType();

            of.write(reinterpret_cast<const char *>(&inputSize), sizeof(inputSize));
            of.write(reinterpret_cast<const char *>(&outputSize), sizeof(outputSize));
            of.write(reinterpret_cast<const char *>(&batchSize), sizeof(batchSize));

            of.write(reinterpret_cast<const char *>(&lr), sizeof(lr));
            of.write(reinterpret_cast<const char *>(&ir), sizeof(ir));
            of.write(reinterpret_cast<const char *>(&pr), sizeof(pr));
            of.write(reinterpret_cast<const char *>(&lambda), sizeof(lambda));

            of.write(reinterpret_cast<const char *>(&activation), sizeof(activation));

            uint64_t weightCount =
                static_cast<uint64_t>(inputSize) *
                static_cast<uint64_t>(outputSize);

            of.write(reinterpret_cast<const char *>(&weightCount), sizeof(weightCount));
            of.write(reinterpret_cast<const char *>(layer->GetWeights()),
                     sizeof(float) * weightCount);

            uint64_t biasCount = outputSize;

            of.write(reinterpret_cast<const char *>(&biasCount), sizeof(biasCount));
            of.write(reinterpret_cast<const char *>(layer->GetBiases()),
                     sizeof(float) * biasCount);
        }

        return of.good();
    }

    bool DiscriminativePCNetwork::Load(const std::string &filename) noexcept
    {
        std::ifstream in(filename, std::ios::binary);
        if (!in)
            return false;

        char magic[8];
        in.read(magic, sizeof(magic));

        if (std::memcmp(magic, "D33P1TY", sizeof(magic)) != 0)
            return false;

        uint32_t version;
        in.read(reinterpret_cast<char *>(&version), sizeof(version));

        if (version != 1)
            return false;

        for (auto *l : layers)
            delete l;
        layers.clear();

        uint32_t layerCount;
        in.read(reinterpret_cast<char *>(&layerCount), sizeof(layerCount));

        for (uint32_t i = 0; i < layerCount; ++i)
        {
            uint32_t inputSize;
            uint32_t outputSize;
            uint32_t batchSize;

            float lr;
            float ir;
            float pr;
            float lambda;

            ActivationType activation;

            in.read(reinterpret_cast<char *>(&inputSize), sizeof(inputSize));
            in.read(reinterpret_cast<char *>(&outputSize), sizeof(outputSize));
            in.read(reinterpret_cast<char *>(&batchSize), sizeof(batchSize));

            in.read(reinterpret_cast<char *>(&lr), sizeof(lr));
            in.read(reinterpret_cast<char *>(&ir), sizeof(ir));
            in.read(reinterpret_cast<char *>(&pr), sizeof(pr));
            in.read(reinterpret_cast<char *>(&lambda), sizeof(lambda));

            in.read(reinterpret_cast<char *>(&activation), sizeof(activation));

            AddLayer(
                inputSize,
                outputSize,
                lr,
                ir,
                pr,
                lambda,
                To_Fn(activation),
                To_dFn(activation));

            DiscriminativePCLayer *layer = layers.back();

            uint64_t weightCount;
            in.read(reinterpret_cast<char *>(&weightCount), sizeof(weightCount));

            uint64_t expectedWeightCount =
                static_cast<uint64_t>(inputSize) *
                static_cast<uint64_t>(outputSize);

            if (weightCount != expectedWeightCount)
                return false;

            in.read(reinterpret_cast<char *>(layer->GetWeights()),
                    sizeof(float) * weightCount);

            uint64_t biasCount;
            in.read(reinterpret_cast<char *>(&biasCount), sizeof(biasCount));

            if (biasCount != outputSize)
                return false;

            in.read(reinterpret_cast<char *>(layer->GetBiases()),
                    sizeof(float) * biasCount);
        }

        return in.good();
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
}