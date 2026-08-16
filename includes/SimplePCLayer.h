#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstdlib>
#include "Activations.h"
#include "AdamOptimizer.h"
#include "Layer.h"
#include "MemoryArena.h"

/**
 * @file SimplePCLayer.h
 * @brief DiscriminativePCLayer with precision entirely removed.
 *
 * Precision weighting (p, log_p, UpdatePrecision()) was found this session
 * to be an unused cost, not a benefit: every real training run used pr=0.0
 * (precision inert -- p stays permanently at 1, log_p at 0), yet
 * UpdatePrecision() was still called unconditionally every train_step(),
 * and every layer paid for the p buffer, the log_p buffer, and the
 * precision-weighted energy formula's extra multiply/log term on every
 * single CalculateState()/UpdateState() call, for zero behavioral benefit.
 *
 * This class removes all of that: energy reduces to plain
 * E = 0.5*sum(e^2) (no precision weighting, no -0.5*log(p) term), and
 * UpdateState()'s own term reduces to dz_dt = -e (no p multiplier, since
 * p=1 always made that multiplication a no-op anyway).
 *
 * Mirrors DiscriminativePCLayer's public interface as closely as possible
 * (same AddLayer-style construction pattern, same ActivationType-based
 * constructor, same GetBeliefs/GetErrors/GetWeights/GetBiases accessors)
 * so it's a near-drop-in swap for anyone not using precision -- which, per
 * the above, was everyone.
 *
 * @warning Not yet gradient-checked independently -- this is a direct
 * strip-down of DiscriminativePCLayer's already-verified math (same
 * CalculateState/UpdateState/UpdateWeights formulas, precision terms
 * removed), not new math, but the removal itself hasn't been re-verified
 * via finite-difference check. Do that before trusting this for real
 * training -- same discipline as every other change this session.
 */

namespace Deep
{
    class SimplePCNDiagnostics;

    class SimplePCLayer : public Layer
    {
    public:
        /// @brief Constructor for a SimplePCLayer.
        /// @param size Size of this layer's own belief (z)
        /// @param nextSize Size of the layer above's belief (this layer's
        ///        outgoing prediction target); 0 marks a terminal layer
        /// @param batchSize Batch size
        /// @param learningRate Learning rate for weight updates
        /// @param inferenceRate Inference rate (Euler integration step size)
        /// @param lmbda Weight decay (L2 regularization) coefficient
        /// @param act Activation function
        /// @param dAct Derivative of activation function
        SimplePCLayer(int size, int nextSize, int batchSize = 1,
                     float learningRate = 1e-6f, float inferenceRate = 0.1f, float lmbda = 1e-2f,
                     void (*act)(float *, size_t) = relu,
                     void (*dAct)(float *, size_t, bool) = dRelu);

        SimplePCLayer(int size, int nextSize, int batchSize = 1,
                     float learningRate = 1e-6f, float inferenceRate = 0.1f, float lmbda = 1e-2f,
                     ActivationType aType = ActivationType::RELU, ActivationType dType = ActivationType::dRELU);

        /// @brief Calculates the total network energy state.
        ///
        /// \f[
        /// E = \sum_l 1/2 ||z^{(l)} - \mu^{(l)}||^2
        /// \f]
        /// (No precision weighting -- see file-level note.)
        float CalculateState() noexcept override;

        /// @brief Computes the state derivatives for inference.
        ///
        /// \f[
        /// \frac{dz^{(l)}}{dt} = -e^{(l)} + (W^{(l-1)})^T e^{(l-1)} \odot \sigma'(W^{(l-1)}z^{(l)})
        /// \f]
        void UpdateState() noexcept override;

        /// @brief Computes weight updates via gradient descent, with L2 weight decay.
        void UpdateWeights() noexcept override;

        void Flush() noexcept override {}

        void ClampState(const std::vector<float> &inputData) noexcept;
        void UnclampState() noexcept;

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }
        size_t GetInputSize() const noexcept override { return size; }
        size_t GetOutputSize() const noexcept override { return nextSize; }
        size_t GetBatchSize() const noexcept override { return batchSize; }

        const float *GetWeights() const noexcept { return W; }
        float *GetWeights() noexcept { return W; }
        const float *GetBiases() const noexcept { return b; }
        float *GetBiases() noexcept { return b; }

        float GetLearningRate() const noexcept { return lr; }
        float GetInferenceRate() const noexcept { return ir; }
        float GetLambda() const noexcept { return lmbda; }

        void SetLearningRate(float lr) noexcept { this->lr = lr; }
        void SetInferenceRate(float ir) noexcept { this->ir = ir; }
        void SetLambda(float l) noexcept { this->lmbda = l; }
        
        void SetOptimizer(const OptimizerType o) noexcept { opt = o; }

        void SetLayerAbove(SimplePCLayer *above) noexcept { layerAbove = above; }
        void SetLayerBelow(SimplePCLayer *below) noexcept { layerBelow = below; }

        void ResetState() noexcept;

        const SimplePCLayer &GetLayerAbove() const noexcept { return *layerAbove; }
        const SimplePCLayer &GetLayerBelow() const noexcept { return *layerBelow; }

        void RandomizeWeights(std::mt19937 &twister) noexcept;

        ActivationType GetActivationType() const noexcept { return To_AType(activation); }
        ActivationType GetDerivativeType() const noexcept { return To_AType(activationDerivative); }

        size_t GetRequiredFloats() const noexcept;
        void BindMemory(MemoryArena &arena);

    private:
        std::unique_ptr<MemoryArena> localArena;
        float *W;
        float *b;
        float *e;
        float *z;

        int batchSize;

        float *mu;
        float *dz_dt;
        float *bottom_up;

        float lr;
        float ir;
        float lmbda;
        bool isClamped = false;

        SimplePCLayer *layerAbove;
        SimplePCLayer *layerBelow;
        ActivationFn activation;
        DerivativeFn activationDerivative;
        ActivationType activationType;
        OptimizerType opt = OptimizerType::SGD;
        // @private Optional Adam weights
        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;
        int t = 0;

        friend class SimplePCNDiagnostics;
    };

} // namespace Deep
