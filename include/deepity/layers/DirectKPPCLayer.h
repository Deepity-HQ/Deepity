#pragma once

#include <deepity/layers/Layer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/backend/IComputeBackend.h>
#include <memory>
#include <vector>
#include <random>

/**
 * @file DirectKPPCLayer.h
 * @brief Direct Kolen-Pollack predictive coding layer, routed through
 * IComputeBackend.
 *
 * @note As of this revision, ComputeMuOnly()'s bias-add and
 * UpdateWeights()'s bias-gradient accumulation both go through
 * AddBiasBroadcast()/SumRows() instead of a per-batch-row AxpyInto loop
 * -- same fix SimplePCLayer already had, ported here. biasGradScratch is
 * a new, small (nextSize-length) buffer allocated unconditionally
 * (regardless of optimizer) specifically for the SGD branch's
 * accumulate-not-overwrite bias update, which SumRows alone can't do.
 */

namespace Deep
{
    class DirectKPPCLayer : public Layer
    {
    protected:
        size_t size;
        size_t nextSize;
        size_t terminalSize;
        size_t batchSize;

        float lr;
        float ir;
        float fl;
        float lmbda;

        bool isClamped = false;
        bool muCacheValid = false;

        DirectKPPCLayer *layerAbove = nullptr;
        DirectKPPCLayer *layerBelow = nullptr;
        DirectKPPCLayer *terminalLayer = nullptr;

        ActivationType activationType;
        ActivationFn activation;
        DerivativeFn activationDerivative;
        DerivativeFn2 activationDerivativeInto;

        OptimizerType opt = OptimizerType::SGD;
        OptimizerType optPsi = OptimizerType::SGD;
        int t = 0;
        int tPsi = 0;

        using BackendDeleter = void (*)(IComputeBackend *);
        std::unique_ptr<IComputeBackend, BackendDeleter> backend;

        float *z = nullptr;
        float *e = nullptr;

        float *W = nullptr;
        float *b = nullptr;
        float *mu = nullptr;
        float *cachedMu = nullptr;
        float *Psi = nullptr;
        float *proj = nullptr;

        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;

        float *grad_Psi = nullptr;
        float *m_Psi = nullptr;
        float *v_Psi = nullptr;

        int *t_device = nullptr;
        float *lr_device = nullptr;
        int *tPsi_device = nullptr;
        float *fl_device = nullptr;

        float *zF = nullptr;
        float *zFDeriv = nullptr;
        float *feedbackScratch = nullptr;

        /// @brief nextSize-length scratch buffer for SumRows' output in
        /// UpdateWeights()'s SGD branch -- SGD needs `b += lr_batch *
        /// sum(local_grad)`, an accumulate, which SumRows alone can't
        /// express (it only overwrites). Allocated unconditionally
        /// (regardless of which optimizer is selected) since it's cheap
        /// -- at most `nextSize` floats -- and simpler than branching
        /// allocation on optimizer choice for this one small buffer.
        float *biasGradScratch = nullptr;

        std::unique_ptr<MemoryArena> localArena;

    public:
        DirectKPPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
                        float learningRate, float inferenceRate, float feedback, float lmbda,
                        ActivationType aType, ActivationType dType,
                        IComputeBackend *backend = nullptr);

        ~DirectKPPCLayer() override = default;

        template <typename ArenaT>
        void BindMemory(ArenaT &arena);
        size_t GetRequiredFloats() const noexcept;
        void RandomizeWeights(std::mt19937 &seedGenerator) noexcept;

        void SetLayerAbove(DirectKPPCLayer *l) noexcept { layerAbove = l; }
        void SetLayerBelow(DirectKPPCLayer *l) noexcept { layerBelow = l; }
        void SetTerminalLayer(DirectKPPCLayer *l) noexcept { terminalLayer = l; }

        /// @brief Matches Layer's virtual interface exactly (always
        /// computes real energy).
        float CalculateState() noexcept override { return CalculateState(true); }
        /// @brief NOT a virtual override -- see SimplePCLayer's
        /// identical pattern. Lets the settling loop skip the
        /// cublasSdot-based energy reduction (and its capture-time
        /// sync) when the caller doesn't need the value.
        float CalculateState(bool needEnergy) noexcept;

        void ComputeMuOnly() noexcept;
        void UpdateState() noexcept override;
        void UpdateWeights() noexcept override;
        void DirectFeedbackUpdate() noexcept;

        void ClampState(const std::vector<float> &inputData) noexcept;
        void UnclampState() noexcept;
        void ResetState() noexcept;

        /// @brief Whether this layer is currently clamped -- needed so
        /// DirectKPPCNetwork::ProjectForward() can skip overwriting an
        /// already-clamped layer's z with a forward-projected guess.
        /// Same real bug SimplePCNetwork::ProjectForward() had; fixed
        /// here for the same reason, before it gets exercised for the
        /// first time by moving ProjectForward() inside graph capture.
        bool IsClamped() const noexcept { return isClamped; }

        void SetOptimizer(OptimizerType o) noexcept { opt = o; }
        void SetPsiOptimizer(OptimizerType o) noexcept { optPsi = o; }
        void SetLearningRate(float learningRate) noexcept;
        void SetInferenceRate(float inferenceRate) noexcept { ir = inferenceRate; }
        void SetFeedbackRate(float feedbackRate) noexcept;
        void SetLambda(float lmbda) noexcept { this->lmbda = lmbda; }

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }
        const float *GetMu() const noexcept { return mu; }
        const float *GetWeights() const noexcept { return W; }
        const float *GetDirectFeedbackWeights() const noexcept { return Psi; }
        const float *GetBiases() const noexcept { return b; }

        size_t GetBatchSize() const noexcept override { return batchSize; }
        size_t GetInputSize() const noexcept override { return size; }
        size_t GetOutputSize() const noexcept override { return nextSize; }
        size_t GetTerminalSize() const noexcept { return terminalSize; }
    };
}