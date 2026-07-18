#pragma once

#include "Layer.h"
#include "Activations.h"
#include <cstddef>
#include <memory>

/**
 * @file RBLayer.h
 * @brief Restricted Boltzmann-style Predictive Coding (PC) Layer.
 *
 * Implements a single layer of a generative PC network. This layer maintains
 * its own state beliefs and manages bidirectional errors (bottom-up and top-down).
 * Memory is designed to be injected via a contiguous arena allocator to guarantee
 * 64-byte alignment for AVX/SIMD vectorization.
 *
 * @version 1.1
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    /// @brief Restricted Boltzmann-style Predictive Coding (PC) Layer.
    ///
    /// @see https://www.nature.com/articles/nn0199_79.pdf
    class RBLayer : public Layer
    {
    public:
        /**
         * @brief Constructs an RBLayer with specified dimensions and hyperparameters.
         * * @param inSize    Dimension of the bottom-up input.
         * @param outSize   Dimension of the top-down output (layer state size).
         * @param var       Bottom-up variance (precision inverse).
         * @param var_td    Top-down variance (precision inverse).
         * @param k_1       Forward integration time constant.
         * @param k_2       Backward integration time constant.
         * @param lmbda     Weight decay (L2 regularization) coefficient.
         * @param alpha     Learning rate for weight updates.
         * @param batchSize Number of parallel sequences to process.
         * @param stepSize  Number of inference iterations per state update.
         * @param act       Forward activation function (SIMD optimized).
         * @param dAct      Derivative of the activation function (SIMD optimized).
         */
        RBLayer(size_t inSize, size_t outSize,
                float var = 1.0f, float var_td = 10.0f,
                float k_1 = 1e-3f, float k_2 = 1e-5f,
                float lmbda = 1e-6f, float alpha = 1.0f,
                size_t batchSize = 64, int stepSize = 30,
                void(*act)(float *, size_t) = relu, void(*dAct)(float *, size_t, bool) = dRelu);

        ~RBLayer() override;

        /// @brief Executes a forward pass prediction without updating internal beliefs.
        void RunPrediction(const float *input, size_t currentBatchSize) noexcept;

        /// @brief Executes a single bidirectional inference step integrating bottom-up and top-down signals.
        void RunInferenceStep(const float *input, const float *topDown, size_t currentBatchSize) noexcept;

        /// @brief Computes the precision-weighted prediction errors.
        void CalcError(const float *bottomUp, const float *topDown, size_t currentBatchSize) noexcept;

        // --------------------------------------------------------
        // Core Execution (Predictive Coding Dynamics)
        // --------------------------------------------------------

        /// @brief Updates internal state beliefs via gradient descent on the prediction error.
        ///
        /// Integrates the state dynamics using the forward error, backward error, and a Gaussian prior (decay):
        /// \f[
        /// r \leftarrow r + k_1 \left( \frac{1}{\sigma^2} (I - rU)U^T + \frac{1}{\sigma^2_{td}}(td - r) - \alpha r \right)
        /// \f]
        /// @param bottomUp The input data \f$ I \f$ from the layer below.
        /// @param topDown The target data \f$ td \f$ from the layer above (can be nullptr).
        /// @param currentBatchSize The active batch dimension.
        void UpdateBeliefs(const float *bottomUp, const float *topDown, size_t currentBatchSize) noexcept;

        /// @brief Applies batched gradients to update the synaptic weights.
        ///
        /// Computes the weight update using the non-linear prediction error, applies L2 weight decay,
        /// and projects the rows of the weight matrix back to a unit norm:
        /// \f[
        /// \Delta U = \frac{k_2}{\sigma^2} r^T \left( (I - f(rU)) \odot f'(rU) \right) - k_2 \lambda U
        /// \f]
        /// @param currentBatchSize The active batch dimension.
        void UpdateWeights(size_t currentBatchSize) noexcept;

        /// @brief Calculates the total layer energy state.
        ///
        /// Computes the sum of squared bottom-up prediction errors:
        /// \f[
        /// E = \frac{1}{2} ||e_{bu}||^2 = \frac{1}{2} \sum (I - rU)^2
        /// \f]
        [[nodiscard]] float CalculateState() noexcept override;

        /// @brief Computes the state derivatives and applies them to current beliefs.
        ///
        /// Wrapper for \c UpdateBeliefs() using the internally cached input batch.
        /// \f[
        /// r \leftarrow r + k_1 \left( \frac{1}{\sigma^2} (I - rU)U^T - \alpha r \right)
        /// \f]
        void UpdateState() noexcept override;

        /// @brief Parameter-less override for generic layer weight updates.
        ///
        /// Wrapper for the batched \c UpdateWeights() using the internally cached input batch.
        /// \f[
        /// \Delta U = \frac{k_2}{\sigma^2} r^T \left( (I - f(rU)) \odot f'(rU) \right) - k_2 \lambda U
        /// \f]
        void UpdateWeights() noexcept override;

        void Flush() noexcept override;

        [[nodiscard]] const float *GetErrors() const noexcept override { return e_bu; }

        /// @brief Returns the total byte size required by this layer for flat allocation.
        [[nodiscard]] size_t GetTotalSize() const noexcept;

        /// @brief Attaches the layer to a pre-allocated, 64-byte aligned memory block.
        void Attach(float *ptr) noexcept;

        [[nodiscard]] float *GetWeights() const noexcept { return U; }

        [[nodiscard]] size_t GetBatchSize() const noexcept override;
        [[nodiscard]] size_t GetInputSize() const noexcept override;
        [[nodiscard]] size_t GetOutputSize() const noexcept override;

        [[nodiscard]] float *GetBeliefs() noexcept override;
        [[nodiscard]] const float *GetInferenceError() const noexcept;

#ifdef _DEBUG
        /// @brief Dumps internal layer statistics to stdout for profiling.
        void DebugStats(int layerIndex) const;
#endif

    private:
        /// @brief Caches the input pointer to avoid redundant CBLAS allocations.
        void CacheInput(const float *input, size_t currentBatchSize) noexcept;

        // --- Hyperparameters ---
        float var;
        float var_td;
        float k_1;
        float k_2;
        float lmbda;
        float alpha;
        int stepSize;

        // Batch Tracking
        size_t batchSize;
        size_t cachedBatchSize = 0;
        const float *cachedInput = nullptr;

        // SIMD Activation
        void (*act)(float *, size_t);
        void (*dAct)(float *, size_t, bool);

        // Arena Memory Pointers
        float *r = nullptr;    /// Beliefs/states array
        float *U = nullptr;    /// Synaptic weights matrix
        float *e_bu = nullptr; /// Bottom-up prediction error
        float *e_td = nullptr; /// Top-down prediction error

        // BLAS Temporary Buffers
        std::unique_ptr<float[]> inputBuffer;
        size_t pendingCount = 0;
        bool ownsMemory = false;

        float *tmpBelief = nullptr;
        float *tmpWeight1 = nullptr;
        float *tmpWeight2 = nullptr;
        float *tmpWeight3 = nullptr;
        float *tmpWeight4 = nullptr;
    };
} // namespace Deep