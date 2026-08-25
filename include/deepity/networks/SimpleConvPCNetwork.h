#pragma once

#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/SimpleConvPCLayer.h>
#include <deepity/MemoryArena.h>

/**
 * @file SimpleConvPCNetwork.h
 * @brief Convolutional counterpart to SimplePCNetwork -- same public
 * interface shape as ConvPCNetwork (AddLayer/Compile/CalculateState/
 * UpdateState/UpdateWeights/TrainStep/Predict), wired to SimpleConvPCLayer
 * instead, with AdamW/Adam support via SetOptimizer().
 *
 * Unlike using SimpleConvPCLayer standalone (today's verification tests
 * had to manually rebind memory after SetOptimizer(), since the
 * constructor already runs BindMemory() with whatever opt was at
 * construction time), THIS class fixes that gap properly: SetOptimizer()
 * is safe to call any time before Compile(), and Compile() is what
 * actually allocates memory (via a shared arena across all layers),
 * mirroring SimplePCNetwork's established, correct flow.
 *
 * AddLayer's spatial dimensions (inHeight/inWidth) are NOT auto-inferred
 * from the previous layer's output shape -- pass them explicitly for each
 * layer, matching ConvPCNetwork's existing convention.
 */

namespace Deep
{
    class PCNDiagnostics;

    class SimpleConvPCNetwork
    {
    public:
        explicit SimpleConvPCNetwork(int batchSize) noexcept;

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
        /// Mirrors ConvPCNetwork::TrainStep()/DiscriminativePCNetwork's
        /// TrainStep() exactly.
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Clamps input only, settles, and returns the terminal
        /// layer's settled beliefs (flattened, batched).
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

    private:
        std::vector<std::unique_ptr<SimpleConvPCLayer>> layers;
        std::unique_ptr<MemoryArena> arena;
        int batchSize;
        OptimizerType pendingOpt = OptimizerType::SGD;
        friend class PCNDiagnostics;
    };
} // namespace Deep