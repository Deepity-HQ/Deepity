#pragma once
#include <vector>
#include <memory>
#include <random>
#include <deepity/layers/ConvPCLayer.h>
#include <deepity/MemoryArena.h>

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

    /**
     * @brief A convolutional Predictive Coding (PC) network: an ordered
     * stack of ConvPCLayer instances sharing a single contiguous
     * MemoryArena, trained via iterative energy-relaxation rather than
     * backpropagation.
     *
     * Mirrors DiscriminativePCNetwork's public interface shape
     * (AddLayer/Compile/CalculateState/UpdateState/UpdateWeights/
     * TrainStep/Predict) so the two can generally be swapped for each
     * other, but wires ConvPCLayer instead of DiscriminativePCLayer,
     * making this the convolutional variant.
     *
     * Typical usage:
     * @code
     * Deep::ConvPCNetwork net(batchSize);
     * net.AddLayer(inChannels, outChannels, inHeight, inWidth, kernelH, kernelW);
     * // ... additional AddLayer() calls ...
     * net.Compile();
     * net.RandomizeWeights(rng);
     * float energy = net.TrainStep(x, y, inferenceSteps);
     * @endcode
     */
    class ConvPCNetwork
    {
    public:
        /**
         * @brief Constructs an empty convolutional PC network for a fixed
         * batch size.
         *
         * No layers exist yet; call AddLayer() for each layer followed by
         * a single Compile() call before use.
         *
         * @param batchSize Number of samples processed together in every
         * Clamp()/CalculateState()/UpdateState()/TrainStep()/Predict()
         * call. Fixed for the lifetime of the network.
         */
        explicit ConvPCNetwork(int batchSize) noexcept;

        /**
         * @brief Destroys the network, freeing every owned ConvPCLayer
         * and releasing the backing MemoryArena.
         */
        ~ConvPCNetwork();

        // No copy (owns raw pointers + a MemoryArena); move not implemented
        // either, matching DiscriminativePCNetwork's conventions.

        /// @brief Deleted: ConvPCNetwork owns raw layer pointers and a
        /// MemoryArena, so copying is not supported.
        ConvPCNetwork(const ConvPCNetwork &) = delete;

        /// @brief Deleted: ConvPCNetwork owns raw layer pointers and a
        /// MemoryArena, so copy-assignment is not supported.
        ConvPCNetwork &operator=(const ConvPCNetwork &) = delete;

        /**
         * @brief Adds a convolutional layer. Pass outChannels=0 to mark a
         * terminal layer (no outgoing prediction), matching
         * DiscriminativePCLayer's nextSize=0 convention.
         *
         * Spatial dimensions are NOT inferred from the previous layer's
         * output shape -- inHeight/inWidth must be passed explicitly for
         * every layer, including layers after the first. See the
         * file-level @ref ConvPCNetwork.h documentation for why.
         *
         * @param inChannels Number of input channels this layer accepts.
         * @param outChannels Number of output channels this layer
         * produces; pass 0 to mark this as the terminal layer.
         * @param inHeight Height of the input feature map for this layer.
         * @param inWidth Width of the input feature map for this layer.
         * @param kernelH Height of the convolution kernel.
         * @param kernelW Width of the convolution kernel.
         * @param strideH Vertical stride of the convolution. Defaults to 1.
         * @param strideW Horizontal stride of the convolution. Defaults
         * to 1.
         * @param padH Vertical (top/bottom) zero-padding applied to the
         * input. Defaults to 0.
         * @param padW Horizontal (left/right) zero-padding applied to the
         * input. Defaults to 0.
         * @param lr Learning rate used for this layer's weight updates.
         * @param ir Inference rate used while settling this layer's
         * beliefs during CalculateState()/UpdateState().
         * @param pr Precision-update rate used by UpdatePrecision().
         * @param lmbda Weight-decay / regularization coefficient applied
         * during UpdateWeights().
         * @param aType Activation function applied to this layer's
         * beliefs. Defaults to ActivationType::RELU.
         * @param dType Derivative of the activation function, used during
         * energy relaxation. Defaults to ActivationType::dRELU. Must
         * correspond to @p aType.
         */
        void AddLayer(int inChannels, int outChannels,
                      int inHeight, int inWidth,
                      int kernelH, int kernelW,
                      int strideH = 1, int strideW = 1,
                      int padH = 0, int padW = 0,
                      float lr = 1e-6f, float ir = 0.1f,
                      float pr = 0.01f, float lmbda = 1e-2f,
                      ActivationType aType = ActivationType::RELU,
                      ActivationType dType = ActivationType::dRELU);

        /**
         * @brief Sums every layer's required float count into a single
         * contiguous arena and binds each layer into it. Call after all
         * AddLayer() calls, before RandomizeWeights().
         *
         * @warning Calling AddLayer() again after Compile() has been
         * called results in undefined/unsupported behavior, since layer
         * offsets into the arena are fixed at compile time.
         */
        void Compile();

        /**
         * @brief Randomizes every layer's weights (and biases, where
         * applicable) in place using the supplied random engine.
         *
         * Must be called after Compile(), since weight storage is only
         * valid once every layer has been bound into the arena.
         *
         * @param rng The classic Mersenne Twister engine used as the
         * source of randomness; passed by reference so the caller retains
         * control of seeding and can reuse the same engine elsewhere.
         */
        void RandomizeWeights(std::mt19937 &rng) noexcept;

        /**
         * @brief Resets every layer's beliefs/errors (and any other
         * per-step state) back to their initial values, without touching
         * learned weights.
         *
         * Typically called between independent Clamp()/settle cycles so
         * that stale beliefs from a previous input don't bias the next
         * one.
         */
        void ResetState() noexcept;

        /**
         * @brief Clamps the flattened, batched input to the first
         * (input) layer.
         *
         * @param input Flattened, batched input data. Expected length and
         * layout must match the first layer's inChannels/inHeight/inWidth
         * and the network's batchSize.
         */
        void Clamp(const std::vector<float> &input) noexcept;

        /**
         * @brief Performs a single energy-relaxation step across every
         * layer given the current clamped/settled state, without
         * committing the resulting update (see UpdateState()).
         *
         * @return The network's total energy at the current state, summed
         * across all layers -- lower values indicate a better fit between
         * predictions and observed/clamped values.
         */
        float CalculateState() noexcept;

        /**
         * @brief Commits the state update computed by the most recent
         * CalculateState() call, advancing every layer's beliefs one
         * relaxation step.
         */
        void UpdateState() noexcept;

        /**
         * @brief Calls UpdateWeights() on every layer except the terminal
         * one (mirrors DiscriminativePCNetwork; each layer's own
         * UpdateWeights() is also self-guarded against outChannels==0, so
         * this is belt-and-suspenders, not load-bearing).
         */
        void UpdateWeights() noexcept;

        /**
         * @brief Updates every layer's precision estimate based on
         * current prediction errors, using each layer's configured
         * precision rate (pr).
         */
        void UpdatePrecision() noexcept;

        /**
         * @brief Returns the network's terminal (final) layer.
         * @return Pointer to the last layer added via AddLayer(); this is
         * the layer with outChannels==0.
         */
        ConvPCLayer *GetTerminalLayer() noexcept { return layers.back(); }

        /**
         * @brief Returns every layer in the network, in the order they
         * were added.
         * @return A const reference to the internal layer list. Valid for
         * the lifetime of this ConvPCNetwork.
         */
        const std::vector<ConvPCLayer *> &GetLayers() const noexcept { return layers; }

        /**
         * @brief Returns the fixed batch size this network was
         * constructed with.
         * @return The number of samples processed together in every
         * per-step call.
         */
        int GetBatchSize() const noexcept { return batchSize; }

        /**
         * @brief Full train step: clamp input+target, settle for
         * inferenceSteps, update weights once, return the final energy.
         * Mirrors DiscriminativePCNetwork::TrainStep() exactly.
         *
         * @param x Flattened, batched input data clamped to the first
         * layer.
         * @param y Flattened, batched target data clamped to the terminal
         * layer.
         * @param inferenceSteps Number of CalculateState()/UpdateState()
         * relaxation iterations to run before updating weights.
         * @return The network's total energy after the final relaxation
         * step, immediately before weights are updated.
         */
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /**
         * @brief Clamps input only, settles, and returns the terminal
         * layer's settled beliefs (flattened, batched).
         *
         * @param x Flattened, batched input data clamped to the first
         * layer.
         * @param inferenceSteps Number of CalculateState()/UpdateState()
         * relaxation iterations to run before reading out predictions.
         * @return The terminal layer's settled beliefs, flattened and
         * batched in the same layout as @p x.
         */
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

    private:
        /// @brief Every layer in the network, in the order added via
        /// AddLayer(); owned raw pointers, freed in the destructor.
        std::vector<ConvPCLayer *> layers;

        /// @brief Single contiguous memory arena backing every layer's
        /// beliefs/errors/weights, bound during Compile().
        std::unique_ptr<MemoryArena> arena;

        /// @brief Fixed batch size supplied at construction; shared by
        /// every layer in this network.
        int batchSize;

        friend class PCNDiagnostics;
    };
} // namespace Deep