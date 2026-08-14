#pragma once

#include <vector>
#include <memory>
#include <random>
#include "ConvPCLayer.h"
#include "MemoryArena.h"

/**
 * @file ConvPCNetwork.h
 * @brief Convolutional counterpart to DiscriminativePCNetwork -- same
 * public interface shape (AddLayer/Compile/CalculateState/UpdateState/
 * UpdateWeights/TrainStep/Predict), wired to ConvPCLayer instead.
 *
 * AddLayer's spatial dimensions (inHeight/inWidth) are NOT auto-inferred
 * from the previous layer's output shape -- pass them explicitly for each
 * layer. This mirrors how every ConvPCLayer test this session specified
 * shapes directly rather than relying on inference, and avoids a class of
 * silent-shape-mismatch bug if a stride/padding choice doesn't produce the
 * dimension you expected.
 */

namespace Deep
{
    class PCNDiagnostics;

    class ConvPCNetwork
    {
    public:
        explicit ConvPCNetwork(int batchSize) noexcept;
        ~ConvPCNetwork();

        // No copy (owns raw pointers + a MemoryArena); move not implemented
        // either, matching DiscriminativePCNetwork's conventions.
        ConvPCNetwork(const ConvPCNetwork &) = delete;
        ConvPCNetwork &operator=(const ConvPCNetwork &) = delete;

        /// @brief Adds a convolutional layer. Pass outChannels=0 to mark a
        /// terminal layer (no outgoing prediction), matching
        /// DiscriminativePCLayer's nextSize=0 convention.
        void AddLayer(int inChannels, int outChannels,
                      int inHeight, int inWidth,
                      int kernelH, int kernelW,
                      int strideH = 1, int strideW = 1,
                      int padH = 0, int padW = 0,
                      float lr = 1e-6f, float ir = 0.1f,
                      float pr = 0.01f, float lmbda = 1e-2f,
                      ActivationType aType = ActivationType::RELU,
                      ActivationType dType = ActivationType::dRELU);

        /// @brief Sums every layer's required float count into a single
        /// contiguous arena and binds each layer into it. Call after all
        /// AddLayer() calls, before RandomizeWeights().
        void Compile();

        void RandomizeWeights(std::mt19937 &rng) noexcept;
        void ResetState() noexcept;

        /// @brief Clamps the flattened, batched input to the first
        /// (input) layer.
        void Clamp(const std::vector<float> &input) noexcept;

        float CalculateState() noexcept;
        void UpdateState() noexcept;

        /// @brief Calls UpdateWeights() on every layer except the terminal
        /// one (mirrors DiscriminativePCNetwork; each layer's own
        /// UpdateWeights() is also self-guarded against outChannels==0, so
        /// this is belt-and-suspenders, not load-bearing).
        void UpdateWeights() noexcept;

        void UpdatePrecision() noexcept;

        ConvPCLayer *GetTerminalLayer() noexcept { return layers.back(); }
	const std::vector<ConvPCLayer *> &GetLayers() const noexcept { return layers; }
        int GetBatchSize() const noexcept { return batchSize; }

        /// @brief Full train step: clamp input+target, settle for
        /// inferenceSteps, update weights once, return the final energy.
        /// Mirrors DiscriminativePCNetwork::TrainStep() exactly.
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Clamps input only, settles, and returns the terminal
        /// layer's settled beliefs (flattened, batched).
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

    private:
        std::vector<ConvPCLayer *> layers;
        std::unique_ptr<MemoryArena> arena;
        int batchSize;
        friend class PCNDiagnostics;
    };
} // namespace Deep
