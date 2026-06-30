#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include "Activations.h"
#include "Layer.h"

/**
 * @file PCLayer.h
 * @brief Defines the single-layer implementation of a PC model.
 *
 * This header includes implementations of state calculation, state updates, and learning.
 *
 * Usage:
 *  #include <PCLayer.h>
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
    class PCLayer : public Layer
    {
    public:
        /// @brief Constructor for a Deepity Layer
        /// @param size Size of layer
        /// @param nextSize Size of next layer (used for row size of W)
        /// @param batchSize Batch size (default to 1 for simplicity)
        /// @param learningRate Learning rate to update state
        /// @param act Activation function
        /// @param dAct Derivative of Activation function
        PCLayer(int size, int nextSize, int batchSize = 1, float learningRate = 1e-6,
              void (*act)(float *, size_t) = relu,
              void (*dAct)(float *, size_t) = dRelu);

        /// @brief Destructor
        ~PCLayer() override;

        /// @brief Copy constructor
        /// @param other Layer to copy from
        PCLayer(const PCLayer &other);
        /// @brief Copy assignment
        /// @param other Layer to copy from
        /// @return Copied layer
        PCLayer &operator=(const PCLayer &other);
        /// @brief Move constructor
        /// @param other Layer to move from
        PCLayer(PCLayer &&other);
        /// @brief Move assignment
        /// @param other Layer to move from
        /// @return Moved layer
        PCLayer &operator=(PCLayer &&other);

        /// @brief Calculates $E = \sum_l 1/2 ||z^{(l)} - \mu^{(l)}||^2$
        /// @return E
        float CalculateState() noexcept override;
        /// @brief Calculates $dz^{(l)}/dt = -e^{(l)} + (W^{(l-1)})^T e^{(l-1)} \odot \sigma'(W^{(l-1)}z^{(l)})$
        void UpdateState() noexcept override;
        /// @brief Calculates $\Delta W^{(l)} = -\eta e^{(l)} (z^{(l+1)})^T$
        void UpdateWeights() noexcept override;

        void Flush() noexcept override {} // no buffer

        /// @brief Clamps the layer to the input data
        /// @param inputData Input
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Unclamps the layer to the input data
        void UnclampState() noexcept;

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }
        size_t GetInputSize() const noexcept override { return size; }
        size_t GetOutputSize() const noexcept override { return nextSize; }
        size_t GetBatchSize() const noexcept override { return batchSize; }

        const float *GetWeights() const noexcept { return W; }

        void SetLayerAbove(PCLayer *above) noexcept { layerAbove = above; }
        void SetLayerBelow(PCLayer *below) noexcept { layerBelow = below; }

        /// @brief Makes the weights W randomized to [0.0, 0.1] using an OpenMP-parallelized uniform real distribution.
        /// @param twister The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &twister) noexcept;

    private:
        /// @brief Weights
        float *W;
        /// @brief Errors
        float *e;
        /// @brief Internal state
        float *z;

        int batchSize;

        // @internal --- For inference ---
        float *mu;
        float *sigma_prime;
        float *dz_dt;
        float *bottom_up;
        // ------

        /// @brief Learning rate for internal state
        float lr;
        /// @brief Flag to tell if `ClampState` was called.
        bool isClamped;

        /// @brief Pointer to next layer (or `nullptr` if last one)
        PCLayer *layerAbove;
        /// @brief Pointer to previous layer (or `nullptr` if first one)
        PCLayer *layerBelow;
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
