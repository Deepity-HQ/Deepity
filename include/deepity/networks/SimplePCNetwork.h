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
 *
 * This header includes implementations of PC layer-to-layer interaction.
 *
 * Usage:
 *  #include <deepity/SimplePCNetwork.h>
 *
 * Example:
 *  Deep::SimplePCNetwork network(1);
 *  network.addLayer({...});
 *  network.Clamp(input);
 *  network.CalculateState();
 *
 * @note All layers are stored in a vector.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    /// @brief An abstracted class for an array of `SimplePCLayer`
    class SimplePCNetwork
    {
    public:
        /// @brief Constructs an empty network with a predetermined batch size.
        /// @param batchSize Batch size
        explicit SimplePCNetwork(int batchSize) noexcept;

        /// @brief Default destructor; deletes each layer.
        ~SimplePCNetwork();

        SimplePCNetwork(const SimplePCNetwork &) = delete;
        SimplePCNetwork &operator=(const SimplePCNetwork &) = delete;

        /// @brief Adds a layer to the network.
        /// @param size input size
        /// @param nextSize output size
        /// @param lr learning rate for beliefs
        /// @param ir learning rate for weights
        /// @param lmbda weight decay (L2 regularization) coefficient
        /// @param act activation function
        /// @param dAct derivative of previous activation function
        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      void (*act)(float *, size_t), void (*dAct)(float *, size_t, bool));

        void AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                      ActivationType aType, ActivationType dType);

        /// @brief Randomizes the weights of each layer
        /// @param rng The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &rng);

        /// @brief Resets each layer's state without touching learned weights.
        void ResetState() noexcept;

        /// @brief Clamps the input to the first layer, necessary for prediction
        /// @param input reference to input vector
        void Clamp(const std::vector<float> &input);

        /// @brief Calculates the state of each layer
        /// @return Returns total energy
        float CalculateState();

        /// @brief Updates each layer's state
        void UpdateState();

        /// @brief Updates each layer's weights
        void UpdateWeights();

        // NOTE: no UpdatePrecision() -- precision doesn't exist in this class.

        /// @brief Returns the network's terminal (final) layer.
        /// @return Pointer to the last layer added via AddLayer().
        SimplePCLayer *GetTerminalLayer() noexcept { return layers.back(); }

        /// @brief Returns every layer in the network, in the order they were added.
        /// @return A reference to the internal layer list.
        std::vector<SimplePCLayer *> &GetLayers() noexcept { return layers; }

        /// @brief Returns every layer in the network, in the order they were added.
        /// @return A const reference to the internal layer list.
        const std::vector<SimplePCLayer *> &GetLayers() const noexcept { return layers; }

        /// @brief Returns the batch size for the network's layers
        /// @return size_t batchSize
        int GetBatchSize() const noexcept { return batchSize; }

        /// @brief Sets the optimizer used for weight updates on every layer.
        /// @param o The optimizer type to apply.
        void SetOptimizer(OptimizerType o) noexcept
        {
            for (SimplePCLayer *layer : layers)
                layer->SetOptimizer(o);
        }

        /// @brief Runs a complete training step (clamp, settle, update, unclamp)
        /// @param x The batched input data
        /// @param y The batched target data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return The final energy state of the network before weight updates
        float TrainStep(const std::vector<float> &x, const std::vector<float> &y, int inferenceSteps);

        /// @brief Runs a forward prediction pass (clamp, settle, read)
        /// @param x The batched input data
        /// @param inferenceSteps The number of relaxation iterations
        /// @return A vector containing the batched predictions
        std::vector<float> Predict(const std::vector<float> &x, int inferenceSteps);

        /// @brief Runs a single, non-iterative forward pass through current
        /// weights, seeding each hidden layer's z from the PREVIOUS layer's
        /// mu -- giving the settling loop a genuine, current-weights-based
        /// starting point instead of zero-init.
        ///
        /// Reuses each layer's EXISTING CalculateState() (already-verified
        /// forward computation, mu = f(W@z+b)) as a side effect -- no new
        /// math, just a new sequence of existing calls. Assumes the input
        /// layer (layers[0]) is ALREADY clamped before this is called.
        ///
        /// @warning Layer 0's error (e) and this call's energy return value
        /// are meaningless here -- CalculateState() computes both mu (what
        /// we want) and e/energy (a side effect we're discarding, since the
        /// NEXT layer's z hasn't been set to a meaningful value yet at the
        /// point each layer's CalculateState() runs). Only mu is used.
        void ProjectForward() noexcept;

        /// @brief Loads all layers into one contiguous block of memory.
        void Compile();

    private:
        std::vector<SimplePCLayer *> layers;
        std::unique_ptr<MemoryArena> arena;
        int batchSize;
    };
}