#pragma once
#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/ConvPCLayer.h>
#include <deepity/MemoryArena.h>

namespace Deep
{
    class PCNDiagnostics;

    class ConvPCNetwork
    {
    public:
        explicit ConvPCNetwork(int batchSize) noexcept;
        ~ConvPCNetwork() = default;

        ConvPCNetwork(const ConvPCNetwork &) = delete;
        ConvPCNetwork &operator=(const ConvPCNetwork &) = delete;

        void AddLayer(int inChannels, int outChannels,
                      int inHeight, int inWidth,
                      int kernelH, int kernelW,
                      int strideH = 1, int strideW = 1,
                      int padH = 0, int padW = 0,
                      float lr = 1e-6f, float ir = 0.1f,
                      float pr = 0.01f, float lmbda = 1e-2f,
                      ActivationType aType = ActivationType::RELU,
                      ActivationType dType = ActivationType::dRELU);

        void Compile();
        void RandomizeWeights(std::mt19937 &rng) noexcept;
        void ResetState() noexcept;
        void Clamp(const std::vector<float> &input) noexcept;
        float CalculateState() noexcept;
        void UpdateState() noexcept;
        void UpdateWeights() noexcept;
        void UpdatePrecision() noexcept;

        ConvPCLayer *GetTerminalLayer() noexcept { return layers.back().get(); }
        const auto &GetLayers() const noexcept { return layers; }
        int GetBatchSize() const noexcept { return batchSize; }

        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        void ProjectForward() noexcept;
        float TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);
        std::vector<float> PredictWithProjection(const std::vector<float> &x, int inferenceSteps);

    private:
        std::vector<std::unique_ptr<ConvPCLayer>> layers;
        std::unique_ptr<MemoryArena> arena;
        int batchSize;

        friend class PCNDiagnostics;
    };
} // namespace Deep
