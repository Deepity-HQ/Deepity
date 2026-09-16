#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstdlib>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/MemoryArena.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Im2Col.h>
#include <deepity/backend/IComputeBackend.h>

/**
 * @file SimpleConvPCLayer.h
 * @brief ConvPCLayer with precision removed, AdamW/Adam support, now
 * routed through IComputeBackend for GPU portability (mirrors
 * SimplePCLayer's own IComputeBackend port).
 *
 * @warning CPU correctness re-verified tonight via five independent,
 * hand-computable tests (single-channel, multi-channel, real spatial
 * kernel, padding, and multi-channel+real-kernel combined) plus a
 * from-scratch Python im2col+GEMM reference comparison -- all matched
 * to float32 precision. The GPU path (once CUDABackend implements
 * Im2Col/Col2Im/RepackForBatchedGemm/MultiplyInto/Fill/
 * AddBiasPerChannel) has NOT yet been tested at all.
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
                          ActivationType dType = ActivationType::dRELU,
                          IComputeBackend *backend = nullptr);

        float CalculateState() noexcept override;
        void UpdateState() noexcept override;
        void UpdateWeights() noexcept override;

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

        void SetLearningRate(float lr) noexcept;
        void SetInferenceRate(float ir) noexcept { this->ir = ir; }
        void SetLambda(float l) noexcept { this->lmbda = l; }
        void SetOptimizer(const OptimizerType o) noexcept { opt = o; }
        bool IsClamped() const noexcept { return isClamped; }

        void SetLayerAbove(SimpleConvPCLayer *above) noexcept { layerAbove = above; }
        void SetLayerBelow(SimpleConvPCLayer *below) noexcept { layerBelow = below; }

        void ComputeMuOnly() noexcept;

        void ResetState() noexcept;
        void RandomizeWeights(std::mt19937 &twister) noexcept;

        ActivationType GetActivationType() const noexcept { return activationType; }
        ActivationType GetDerivativeType() const noexcept { return derivativeType; }

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

        template <typename ArenaT>
        void BindMemory(ArenaT &arena);

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

        // All-ones vector for the bias-gradient GEMM trick (grad_b =
        // lgRepacked @ ones) -- see UpdateWeights()'s implementation
        // comment for why this replaces the original per-row scalar
        // sum loop.
        float *onesVector = nullptr;

        float *cachedMu = nullptr;
        bool muCacheValid = false;

        float *grad_W = nullptr;
        float *grad_b = nullptr;
        float *m_W = nullptr;
        float *v_W = nullptr;
        float *m_b = nullptr;
        float *v_b = nullptr;

        // Device-resident t/lr -- same reasoning as SimplePCLayer's own
        // port: graph capture (once this reaches GPU) can't re-record
        // for every changed learning rate or Adam step count, so both
        // must live in device memory the graph reads from directly.
        int *t_device = nullptr;
        float *lr_device = nullptr;

        float lr, ir, lmbda;
        bool isClamped = false;

        SimpleConvPCLayer *layerAbove = nullptr;
        SimpleConvPCLayer *layerBelow = nullptr;
        ActivationType activationType;
        ActivationType derivativeType;
        OptimizerType opt = OptimizerType::SGD;

        using BackendDeleter = void (*)(IComputeBackend *);
        std::unique_ptr<IComputeBackend, BackendDeleter> backend;

        friend class SimpleConvPCNDiagnostics;
    };

} // namespace Deep