#pragma once

#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/SimpleConvPCLayer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/DeviceType.h>

/**
 * @file SimpleConvPCNetwork.h
 * @brief Convolutional counterpart to SimplePCNetwork, device-aware
 * (DeviceType constructor parameter, backend member, cpuArena/gpuArena
 * split in Compile()).
 */

namespace Deep
{
    class PCNDiagnostics;

    /// @brief Convolutional Predictive Coding Network built from
    /// SimpleConvPCLayer, precision-free, AdamW-capable, device-aware.
    class SimpleConvPCNetwork
    {
    public:
        /// @brief Constructs an empty network; add layers via AddLayer(),
        /// then Compile() before use.
        /// @param batchSize Fixed batch size for every layer.
        /// @param device Which device this network's layers run on.
        explicit SimpleConvPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept;

        SimpleConvPCNetwork(const SimpleConvPCNetwork &) = delete;
        SimpleConvPCNetwork &operator=(const SimpleConvPCNetwork &) = delete;

        ~SimpleConvPCNetwork() = default;

        /// @brief Adds a convolutional layer. Pass outChannels=0 to mark a
        /// terminal layer (no outgoing prediction), matching
        /// SimpleConvPCLayer's nextSize=0 convention.
        void AddLayer(int inChannels, int outChannels,
                      int inHeight, int inWidth,
                      int kernelH, int kernelW,
                      int strideH = 1, int strideW = 1,
                      int padH = 0, int padW = 0,
                      float lr = 1e-6f, float ir = 0.1f, float lmbda = 1e-2f,
                      ActivationType aType = ActivationType::RELU,
                      ActivationType dType = ActivationType::dRELU);

        /// @brief Sets the optimizer for EVERY layer added so far. Safe to
        /// call any time before Compile(), unlike using SimpleConvPCLayer
        /// standalone, memory allocation is deferred to Compile(), not the
        /// constructor, so this doesn't require a manual rebind.
        void SetOptimizer(OptimizerType opt) noexcept;

        /// @brief Sums every layer's required float count into a single
        /// contiguous arena and binds each layer into it. Call after all
        /// AddLayer()/SetOptimizer() calls, before RandomizeWeights().
        void Compile();

        /// @brief Initializes every layer's weights randomly.
        void RandomizeWeights(std::mt19937 &rng) noexcept;
        /// @brief Resets the beliefs (z) on every layer.
        void ResetState() noexcept;

        /// @brief Clamps the flattened, batched input to the first
        /// (input) layer.
        void Clamp(const std::vector<float> &input) noexcept;

        /// @brief Computes and returns the network's total energy at the
        /// current state, without changing it.
        float CalculateState() noexcept;
        /// @brief Runs one settling step on every layer.
        void UpdateState() noexcept;

        /// @brief Calls UpdateWeights() on every layer except the terminal
        /// one (each layer's own UpdateWeights() is also self-guarded
        /// against outChannels==0, so this is belt-and-suspenders).
        void UpdateWeights() noexcept;

        /// @brief Returns the last layer (outChannels==0, the terminal one).
        SimpleConvPCLayer *GetTerminalLayer() noexcept { return layers.back().get(); }
        /// @brief Returns every layer in the network, in the order they were added.
        const auto &GetLayers() const noexcept { return layers; }
        /// @brief Returns the batch size given at construction.
        int GetBatchSize() const noexcept { return batchSize; }

        /// @brief Full train step: clamp input+target, settle for
        /// inferenceSteps, update weights once, return the final energy.
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Clamps input only, settles, and returns the terminal
        /// layer's settled beliefs (flattened, batched).
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        /// @brief Seeds hidden layers from a genuine forward pass through
        /// current weights, instead of zero-init. Call after Clamp(),
        /// before the settling loop.
        void ProjectForward() noexcept;
        /// @brief Full train step with forward-projection init: clamp
        /// input, project forward, clamp target, settle, update weights,
        /// return the final energy.
        float TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);
        /// @brief Inference-only counterpart to TrainStepWithProjection():
        /// clamp input, project forward, settle, read out the terminal
        /// layer's beliefs. No target clamp, no weight update.
        std::vector<float> PredictWithProjection(const std::vector<float> &x, int inferenceSteps);

    private:
        std::vector<std::unique_ptr<SimpleConvPCLayer>> layers;
        std::unique_ptr<IComputeBackend> backend;
        DeviceType device;
        std::unique_ptr<MemoryArena> cpuArena;
#if defined(DEEPITY_USE_CUDA)
        std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif
        int batchSize;
        OptimizerType pendingOpt = OptimizerType::SGD;
        friend class PCNDiagnostics;
    };
} // namespace Deep