#pragma once

#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/SimplePCLayer.h>
#include <deepity/MemoryArena.h>

/**
 * @file SimplePCNetwork.h
 * @brief Network wrapper for SimplePCLayer, mirroring DiscriminativePCNetwork
 * exactly (minus the UpdatePrecision() call, which no longer exists).
 */

namespace Deep
{
    class SimplePCNetwork
    {
    public:
        explicit SimplePCNetwork(int batchSize) noexcept;
        ~SimplePCNetwork();

        SimplePCNetwork(const SimplePCNetwork &) = delete;
        SimplePCNetwork &operator=(const SimplePCNetwork &) = delete;

        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool));

        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      ActivationType aType, ActivationType dType);

        void RandomizeWeights(std::mt19937 &rng);
        void ResetState() noexcept;
        void Clamp(const std::vector<float> &input);

        float CalculateState();
        void UpdateState();
        void UpdateWeights();
        // NOTE: no UpdatePrecision() -- precision doesn't exist in this class.

        SimplePCLayer *GetTerminalLayer() noexcept { return layers.back(); }
        std::vector<SimplePCLayer *> &GetLayers() noexcept { return layers; }
        const std::vector<SimplePCLayer *> &GetLayers() const noexcept { return layers; }
        int GetBatchSize() const noexcept { return batchSize; }

        void SetOptimizer(OptimizerType o) noexcept {
            for (SimplePCLayer *layer : layers)
                layer->SetOptimizer(o);
        }

        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        void Compile();

    private:
        std::vector<SimplePCLayer *> layers;
        std::unique_ptr<MemoryArena> arena;
        int batchSize;
    };
}
