#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstdlib>
#include <deepity/Activations.h>
#include <deepity/AdamOptimizer.h>
#include <deepity/layers/Layer.h>
#include <deepity/MemoryArena.h>
#include <deepity/Im2Col.h>

/**
 * @file SimpleConvPCLayer.h
 * @brief ConvPCLayer with precision removed (mirrors SimplePCLayer's
 * relationship to DiscriminativePCLayer) AND AdamW/Adam support added
 * (mirrors SimplePCLayer's own AdamW integration).
 *
 * Precision removal: identical justification to SimplePCLayer -- every
 * real run used pr=0.0 (precision inert), yet every layer paid for the
 * p/log_p buffers and the extra multiply/log terms regardless.
 *
 * @warning Built on TOP of ConvPCLayer's just-verified feedback term
 * (Col2Im path, confirmed via tConvFeedbackVerify.cpp with a genuinely
 * unclamped middle layer -- worst rel err 1.09%). This class has NOT
 * been independently re-verified: neither the precision-removal (should
 * be inert, same argument as SimplePCLayer, but not yet confirmed) nor
 * the AdamW integration (a NEW port, not the one already gradient-checked
 * for SimplePCLayer -- same class of grad_scale-sign bug that was found
 * and fixed there is possible here too, and hasn't been ruled out).
 * Required before trusting this for real training:
 *   1. A finite-difference gradient check on UpdateWeights() (SGD path).
 *   2. The SAME feedback-term verification tConvFeedbackVerify.cpp did
 *      for ConvPCLayer, re-run against THIS class.
 *   3. A separate gradient check specifically for the AdamW path (same
 *      sign-convention pitfall found in SimplePCLayer -- local_grad's
 *      established sign convention may not match what AdamWUpdate()
 *      expects without an explicit sign flip on the raw-gradient GEMM).
 */

namespace Deep
{
    class SimpleConvPCNDiagnostics;

    class SimpleConvPCLayer : public Layer
    {
    public:
        SimpleConvPCLayer(int inChannels, int outChannels,
                          int inHeight, int inWidth,
                          int kernelH, int kernelW,
                          int strideH = 1, int strideW = 1,
                          int padH = 0, int padW = 0,
                          int batchSize = 1,
                          float learningRate = 1e-6f, float inferenceRate = 0.1f,
                          float lmbda = 1e-2f,
                          ActivationType aType = ActivationType::RELU,
                          ActivationType dType = ActivationType::dRELU);

        // --- SGD path ports ConvPCLayer's verified math directly (minus
        // precision terms). AdamW path is NEW, NOT yet independently
        // verified -- see file-level warning. ---
        float CalculateState() noexcept override;
        void UpdateState() noexcept override;
        void UpdateWeights() noexcept override;
        // -------------------------------------------------------------

        void Flush() noexcept override {}

        void ClampState(const std::vector<float> &inputData) noexcept;
        void UnclampState() noexcept;

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }
        size_t GetInputSize() const noexcept override { return (size_t)inChannels * inHeight * inWidth; }
        size_t GetOutputSize() const noexcept override
        {
            return outChannels > 0 ? (size_t)outChannels * outHeight * outWidth : 0;
        }
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

        void SetLayerAbove(SimpleConvPCLayer *above) noexcept { layerAbove = above; }
        void SetLayerBelow(SimpleConvPCLayer *below) noexcept { layerBelow = below; }

        void ComputeMuOnly() noexcept;

        void ResetState() noexcept;
        void RandomizeWeights(std::mt19937 &twister) noexcept;

        ActivationType GetActivationType() const noexcept { return To_AType(activation); }
        ActivationType GetDerivativeType() const noexcept { return To_AType(activationDerivative); }

        const float *GetMu() const noexcept { return mu; }
        int GetInChannels() const noexcept { return inChannels; }
        int GetOutChannels() const noexcept { return outChannels; }
        int GetInHeight() const noexcept { return inHeight; }
        int GetInWidth() const noexcept { return inWidth; }
        int GetOutHeight() const noexcept { return outHeight; }
        int GetOutWidth() const noexcept { return outWidth; }
        int GetKernelH() const noexcept { return kernelH; }
        int GetKernelW() const noexcept { return kernelW; }

        size_t GetRequiredFloats() const noexcept;
        void BindMemory(MemoryArena &arena);

    private:
        std::unique_ptr<MemoryArena> localArena;

        int inChannels, outChannels;
        int inHeight, inWidth;
        int outHeight, outWidth;
        int kernelH, kernelW;
        int strideH, strideW;
        int padH, padW;
        int batchSize;

        float *W = nullptr;
        float *b = nullptr;

        // State -- NO p/log_p (precision removed entirely, unlike ConvPCLayer)
        float *z = nullptr;
        float *e = nullptr;
        float *dz_dt = nullptr;

        float *mu = nullptr;
        float *colBuffer = nullptr;
        float *feedbackScratch = nullptr;
        float *bottom_up_cols = nullptr;
        float *colsRepacked = nullptr;
        float *lgRepacked = nullptr;
        float *muRepacked = nullptr;

        float *cachedMu = nullptr;
        bool muCacheValid = false;

        // Adam-only scratch (allocated conditionally, same pattern as
        // SimplePCLayer -- see GetRequiredFloats/BindMemory)
        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;
        int t = 0;

        float lr, ir, lmbda;
        bool isClamped = false;

        SimpleConvPCLayer *layerAbove = nullptr;
        SimpleConvPCLayer *layerBelow = nullptr;
        ActivationFn activation;
        DerivativeFn activationDerivative;
        ActivationType activationType;
        OptimizerType opt = OptimizerType::SGD; // explicit default -- see the real bug
                                                // this exact omission caused in
                                                // SimplePCLayer earlier this session

        friend class SimpleConvPCNDiagnostics;
    };

} // namespace Deep
