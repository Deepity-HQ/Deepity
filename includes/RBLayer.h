#pragma once
#include "Layer.h"
#include "Activations.h"
#include <cstddef>
#include <memory>

/**
 * @file RBLayer.h
 * @brief Defines the single-layer implementation of a Restricted Boltzmann-style PC model.
 *
 * This header includes implementations of state calculation, state updates, and learning.
 *
 * Usage:
 *  #include <RBLayer.h>
 *
 * Example:
 *  Deep::Layer layer(2, 3, 1, 1e-6);
 *  layer.ClampState({ 0.1f, 0.75f });
 *  layer.CalculateState();
 *
 * @note All members are stored as pointers except for the input itself.
 * @version 1.1
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    // Ensure this matches your project's definition of an activation function
    using ActivationFn = void (*)(float *, size_t);

    class RBLayer : public Layer
    {
    public:
        /// @brief Constructs an RBLayer with specified dimensions and hyperparameters.
        RBLayer(size_t inSize, size_t outSize, float var = 1.0f, float var_td = 10.0f, float k_1 = 1e-3, float k_2 = 1e-5, float lmbda = 1e-6, float alpha = 1.0f, size_t batchSize = 64, int stepSize = 30, ActivationFn act = relu, ActivationFn dAct = dRelu);

        ~RBLayer() override;

        // --- Core Execution (Overrides PredictiveLayer) ---
        void RunPrediction(const float *input, size_t currentBatchSize) noexcept;
        void RunInferenceStep(const float *input, const float *topDown, size_t currentBatchSize) noexcept;
        void CalcError(const float *bottomUp, const float *topDown, size_t currentBatchSize) noexcept;
        void UpdateBeliefs(const float *bottomUp, const float *topDown, size_t currentBatchSize) noexcept;
        void UpdateWeights(size_t currentBatchSize) noexcept;
        void Flush() noexcept override;

        float CalculateState() noexcept override;
        void UpdateState() noexcept override;
        void UpdateWeights() noexcept override;
        const float *GetErrors() const noexcept override { return e_bu; }

        // --- Memory Orchestration ---
        size_t GetTotalSize() const noexcept;
        void Attach(float *ptr) noexcept;
        float *GetWeights() const noexcept { return U; }

        // --- Getters ---
        size_t GetBatchSize() const noexcept override;
        size_t GetInputSize() const noexcept override;
        size_t GetOutputSize() const noexcept override;
        float *GetBeliefs() noexcept override;
        const float *GetInferenceError() const noexcept;

#ifdef _DEBUG
        void DebugStats(int layerIndex) const override;
#endif

    private:
        void CacheInput(const float *input, size_t currentBatchSize) noexcept;

        // Core Hyperparameters
        float var;
        float var_td;
        float k_1;
        float k_2;
        float lmbda;
        float alpha;
        size_t batchSize;
        const float *cachedInput = nullptr;
        size_t cachedBatchSize = 0;

        int stepSize;

        // Activation Functions
        ActivationFn act;
        ActivationFn dAct;

        // Raw Memory Pointers (Attached by PCNetwork's memory arena)
        float *r = nullptr;    // Beliefs/states
        float *U = nullptr;    // Weights
        float *e_bu = nullptr; // Bottom-up error
        float *e_td = nullptr; // Top-down error

        // Temporary Buffers for CBLAS Math
        std::unique_ptr<float[]> inputBuffer;
        size_t pendingCount = 0;

        float *tmpBelief = nullptr;
        float *tmpWeight1 = nullptr;
        float *tmpWeight2 = nullptr;
        float *tmpWeight3 = nullptr;
        float *tmpWeight4 = nullptr;
        bool ownsMemory = false;
    };
}