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
 * @brief Convolutional counterpart to SimplePCNetwork, now device-aware
 * (matches DirectKPPCNetwork's own port earlier tonight: DeviceType
 * constructor parameter, backend member, cpuArena/gpuArena split).
 *
 * Two real bugs fixed during this port, same class found and fixed in
 * SimplePCNetwork/DirectKPPCNetwork: ProjectForward() used std::memcpy
 * directly (wrong once beliefs/mu are device pointers) and had no
 * isClamped guard (would overwrite an already-clamped terminal layer's
 * z with a stale forward-projected value); Predict()/
 * PredictWithProjection() used the std::vector iterator-range
 * constructor directly on `beliefs` (dereferences immediately, wrong
 * for device memory).
 */

namespace Deep
{
    class PCNDiagnostics;

    class SimpleConvPCNetwork
    {
    public:
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
        /// call any time before Compile() -- unlike using SimpleConvPCLayer
        /// standalone, memory allocation is deferred to Compile(), not the
        /// constructor, so this doesn't require a manual rebind.
        void SetOptimizer(OptimizerType opt) noexcept;

        /// @brief Sums every layer's required float count into a single
        /// contiguous arena and binds each layer into it. Call after all
        /// AddLayer()/SetOptimizer() calls, before RandomizeWeights().
        void Compile();

        void RandomizeWeights(std::mt19937 &rng) noexcept;
        void ResetState() noexcept;

        /// @brief Clamps the flattened, batched input to the first
        /// (input) layer.
        void Clamp(const std::vector<float> &input) noexcept;

        float CalculateState() noexcept;
        void UpdateState() noexcept;

        /// @brief Calls UpdateWeights() on every layer except the terminal
        /// one (each layer's own UpdateWeights() is also self-guarded
        /// against outChannels==0, so this is belt-and-suspenders).
        void UpdateWeights() noexcept;

        SimpleConvPCLayer *GetTerminalLayer() noexcept { return layers.back().get(); }
        const auto &GetLayers() const noexcept { return layers; }
        int GetBatchSize() const noexcept { return batchSize; }

        /// @brief Full train step: clamp input+target, settle for
        /// inferenceSteps, update weights once, return the final energy.
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Clamps input only, settles, and returns the terminal
        /// layer's settled beliefs (flattened, batched).
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        void ProjectForward() noexcept;
        float TrainStepWithProjection(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);
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