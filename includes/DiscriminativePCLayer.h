#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include "Activations.h"
#include "Layer.h"

/**
 * @file DiscriminativePCLayer.h
 * @brief Defines the single-layer implementation of a PC model.
 *
 * This header includes implementations of state calculation, state updates, and learning.
 *
 * Usage:
 *  #include <DiscriminativePCLayer.h>
 *
 * Example:
 *  Deep::Layer layer(2, 3, 1, 1e-6);
 *  layer.ClampState({ 0.1f, 0.75f });
 *  layer.CalculateState();
 *
 * @note All members are stored as pointers except for the input itself.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    /**
     * Represents a single layer in a Predictive Coding Network.
     * * To perform inference (prediction):
     * 1. Clamp the input data to the bottom layer's latent state (z).
     * 2. Enter a continuous loop across all layers, calling CalculateState()
     * to compute the local prediction errors (e).
     * 3. Call UpdateState() to adjust the latent states (z) based on those errors.
     * 4. Repeat steps 2 and 3 until the states settle into an equilibrium
     * (the energy is minimized and dz/dt approaches zero).
     * 5. Read the final predictions from the latent states of the desired layers.
     * 6. (For learning): Call UpdateWeights() on all layers simultaneously
     * once the states have fully settled.
     */
    class DiscriminativePCLayer : public Layer
    {
    public:
        /// @brief Constructor for a Deepity Layer
        /// @param size Size of layer
        /// @param nextSize Size of next layer (used for row size of W)
        /// @param batchSize Batch size (default to 1 for simplicity)
        /// @param learningRate Learning rate to update weights
        /// @param inferenceRate Learning rate to update state
        /// @param lmbda Weight decay (L2 regularization) coefficient
        /// @param act Activation function
        /// @param dAct Derivative of Activation function
        DiscriminativePCLayer(int size, int nextSize, int batchSize = 1,
            float learningRate = 1e-6, float inferenceRate = 0.1f, float lmbda = 1e-2f,
              void (*act)(float *, size_t) = relu,
              void (*dAct)(float *, size_t) = dRelu);

        /// @brief Destructor
        ~DiscriminativePCLayer() override;

        /// @brief Copy constructor
        /// @param other Layer to copy from
        DiscriminativePCLayer(const DiscriminativePCLayer &other);
        /// @brief Copy assignment
        /// @param other Layer to copy from
        /// @return Copied layer
        DiscriminativePCLayer &operator=(const DiscriminativePCLayer &other);
        /// @brief Move constructor
        /// @param other Layer to move from
        DiscriminativePCLayer(DiscriminativePCLayer &&other) noexcept;
        /// @brief Move assignment
        /// @param other Layer to move from
        /// @return Moved layer
        DiscriminativePCLayer &operator=(DiscriminativePCLayer &&other);

        /// @brief Calculates the total network energy state.
        ///
        /// \f[
        /// E = \sum_l 1/2 ||z^{(l)} - \mu^{(l)}||^2
        /// \f]
        float CalculateState() noexcept override;

        /// @brief Computes the state derivatives for inference.
        ///
        /// \f[
        /// \frac{dz^{(l)}}{dt} = -e^{(l)} + (W^{(l-1)})^T e^{(l-1)} \odot \sigma'(W^{(l-1)}z^{(l)})
        /// \f]
        void UpdateState() noexcept override;

        /// @brief Computes weight updates via gradient descent, with L2 weight decay.
        ///
        /// \f[
        /// W^{(l)} \leftarrow (1 - \lambda) W^{(l)} - \eta e^{(l)} (z^{(l+1)})^T
        /// \f]
        void UpdateWeights() noexcept override;

        /// @brief Does nothing; exists for class extension.
        void Flush() noexcept override {} // no buffer

        /// @brief Clamps the layer to the input data
        /// @param inputData Input
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Unclamps the layer to the input data
        void UnclampState() noexcept;

        /// @brief Returns beliefs
        /// @return float *z
        float *GetBeliefs() noexcept override { return z; }
        /// @brief Returns errors
        /// @return float *e
        const float *GetErrors() const noexcept override { return e; }
        /// @brief Returns size (input size)
        /// @return size_t size
        size_t GetInputSize() const noexcept override { return size; }
        /// @brief Returns nextSize (output size)
        /// @return size_t nextSize
        size_t GetOutputSize() const noexcept override { return nextSize; }
        /// @brief Returns batchSize
        /// @return size_t batchSize
        size_t GetBatchSize() const noexcept override { return batchSize; }

        /// @brief Returns a read-only version of the stored weights
        /// @return const float *W
        const float *GetWeights() const noexcept { return W; }

        /// @brief Ties this layer to one above it
        /// @param above DiscriminativePCLayer*
        void SetLayerAbove(DiscriminativePCLayer *above) noexcept { layerAbove = above; }
        /// @brief Ties this layer to one below it
        /// @param below DiscriminativePCLayer*
        void SetLayerBelow(DiscriminativePCLayer *below) noexcept { layerBelow = below; }

        void ResetState() noexcept;

        const DiscriminativePCLayer &GetLayerAbove() const noexcept { return *layerAbove; }
        const DiscriminativePCLayer &GetLayerBelow() const noexcept { return *layerBelow; }

        /// @brief Makes the weights W randomized to [0.0, 0.1] using an OpenMP-parallelized uniform real distribution.
        /// @param twister The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &twister) noexcept;

    private:
        /// @brief Weights
        float *W;
        /// @brief Biases
        float *b;
        /// @brief Errors
        float *e;
        /// @brief Internal state
        float *z;

        /// @brief Used for `cblas_sgemm` optimization
        int batchSize;

        // @internal --- For inference ---
        float *mu;
        float *sigma_prime;
        float *dz_dt;
        float *bottom_up;
        // ------

        /// @brief Learning rate for weights
        float lr;
        /// @brief Learning rate for internal state
        float ir;
        /// @brief Weight decay (L2 regularization) coefficient
        float lmbda;
        /// @brief Flag to tell if `ClampState` was called.
        bool isClamped = false;

        /// @brief Pointer to next layer (or `nullptr` if last one)
        DiscriminativePCLayer *layerAbove;
        /// @brief Pointer to previous layer (or `nullptr` if first one)
        DiscriminativePCLayer *layerBelow;
        /// @brief Activation function, with parameters `(float *array, size_t arraysize)`
        void (*activation)(float *, size_t);
        /// @brief The derivative of the `activation` internal, with parameters `(float *array, size_t arraysize)`
        void (*activationDerivative)(float *, size_t);
    };

} // namespace Deep

/*
 * MHSA USING PCNs:
 * In a standard Transformer, Multi-Head Self-Attention (MHSA) computes Queries, Keys,
 * and Values in a rigid, one-shot forward pass to route token context. If implemented
 * using a PCN, the attention scores and context vectors would become dynamic latent
 * states (z). When a sequence is input, the attention routing wouldn't be calculated
 * instantly; instead, it would iteratively settle over artificial time. The network
 * would adjust its attention states to minimize the prediction error between the
 * bottom-up token data and the top-down expectations from higher layers. The Q, K,
 * and V projection matrices would serve as the local weights (W) for this mechanism,
 * and would be updated entirely in parallel via local Hebbian rules once the attention
 * routing reached its minimum-energy equilibrium.
 */
