#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstdlib>
#include "Activations.h"
#include "Layer.h"
#include "MemoryArena.h"
#include "Im2Col.h"

/**
 * @file ConvPCLayer.h
 * @brief Convolutional counterpart to DiscriminativePCLayer.
 *
 * @warning CalculateState/UpdateState/UpdateWeights math is implemented in
 * ConvPCLayer.cpp but has ONLY been verified for its weight-gradient path
 * (UpdateWeights(), via finite-difference check with both layers clamped).
 * The Col2Im-based feedback term in UpdateState() has NOT yet been
 * exercised by any test -- both layers in the existing gradient check were
 * clamped, so that code path never actually ran. Do not trust the full
 * training loop (unclamped hidden layers) until:
 *   1. A gradient check with a genuinely UNCLAMPED middle layer passes,
 *      exercising the feedback term for real.
 *   2. A clean ASan/UBSan run on a tiny synthetic ConvPCNetwork.
 *   3. A synthetic floor test (same pattern as tDiagnose.cpp) passes.
 * Only then wire this into MNIST.
 *
 * Layout convention: NCHW, row-major, contiguous per channel per batch item.
 */

namespace Deep
{
    class PCNDiagnostics;

    class ConvPCLayer : public Layer
    {
    public:
        /// @brief Constructor for a convolutional PC layer.
        /// @param inChannels Number of input channels
        /// @param outChannels Number of output channels (0 marks a terminal
        ///        layer, matching DiscriminativePCLayer's convention where
        ///        nextSize=0 means "no outgoing prediction")
        /// @param inHeight Input feature-map height
        /// @param inWidth Input feature-map width
        /// @param kernelH Kernel height
        /// @param kernelW Kernel width
        /// @param strideH Vertical stride
        /// @param strideW Horizontal stride
        /// @param padH Vertical zero-padding
        /// @param padW Horizontal zero-padding
        /// @param batchSize Batch size
        /// @param learningRate Learning rate for weights
        /// @param inferenceRate Learning rate for internal state
        /// @param precisionRate Learning rate for precision
        /// @param lmbda Weight decay (L2 regularization) coefficient
        /// @param aType Activation type
        /// @param dType Activation derivative type
        ConvPCLayer(int inChannels, int outChannels,
                    int inHeight, int inWidth,
                    int kernelH, int kernelW,
                    int strideH = 1, int strideW = 1,
                    int padH = 0, int padW = 0,
                    int batchSize = 1,
                    float learningRate = 1e-6f, float inferenceRate = 0.1f,
                    float precisionRate = 0.01f, float lmbda = 1e-2f,
                    ActivationType aType = ActivationType::RELU,
                    ActivationType dType = ActivationType::dRELU);

        // --- Weight-gradient path verified; feedback term NOT YET verified
        // -- see file-level warning above ---
        float CalculateState() noexcept override;
        void UpdateState() noexcept override;
        void UpdateWeights() noexcept override;
        void UpdatePrecision() noexcept;
        // -------------------------------------------------------------

        void Flush() noexcept override {}

        void ClampState(const std::vector<float> &inputData) noexcept;
        void UnclampState() noexcept;

        float *GetBeliefs() noexcept override { return z; }
        const float *GetErrors() const noexcept override { return e; }

        /// @brief Flattened element count per batch item (inChannels*H*W).
        size_t GetInputSize() const noexcept override { return (size_t)inChannels * inHeight * inWidth; }
        /// @brief Flattened element count of this layer's OUTGOING
        /// prediction (outChannels*outH*outW), 0 for a terminal layer.
        size_t GetOutputSize() const noexcept override
        {
            return outChannels > 0 ? (size_t)outChannels * outHeight * outWidth : 0;
        }
        size_t GetBatchSize() const noexcept override { return batchSize; }

        const float *GetWeights() const noexcept { return W; }
        float *GetWeights() noexcept { return W; }
        const float *GetBiases() const noexcept { return b; }
        float *GetBiases() noexcept { return b; }
        const float *GetPrecisions() const noexcept { return p; }

        float GetLearningRate() const noexcept { return lr; }
        float GetInferenceRate() const noexcept { return ir; }
        float GetPrecisionRate() const noexcept { return pr; }
        float GetLambda() const noexcept { return lmbda; }

        void SetLearningRate(float lr) noexcept { this->lr = lr; }
        void SetInferenceRate(float ir) noexcept { this->ir = ir; }
        void SetPrecisionRate(float pr) noexcept { this->pr = pr; }
        void SetLambda(float l) noexcept { this->lmbda = l; }

        void SetLayerAbove(ConvPCLayer *above) noexcept { layerAbove = above; }
        void SetLayerBelow(ConvPCLayer *below) noexcept { layerBelow = below; }

        void ResetState() noexcept;

        void RandomizeWeights(std::mt19937 &twister) noexcept;

        /// @brief Rebuilds log_p from p -- required after a checkpoint load
        /// that only persists p (mirrors DiscriminativePCLayer's fix for the
        /// same p/log_p desync issue found in ModelIO::Load()).
        void ResyncLogPrecision() noexcept;

        ActivationType GetActivationType() const noexcept { return To_AType(activation); }
        ActivationType GetDerivativeType() const noexcept { return To_AType(activationDerivative); }

        // --- Conv-specific shape accessors (no DiscriminativePCLayer equivalent) ---
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

        // Shape
        int inChannels, outChannels;
        int inHeight, inWidth;
        int outHeight, outWidth;
        int kernelH, kernelW;
        int strideH, strideW;
        int padH, padW;
        int batchSize;

        // Weights: (outChannels, inChannels*kernelH*kernelW) flattened.
        float *W;
        float *b; // (outChannels)

        // State -- per batch item, (inChannels, inHeight, inWidth) flattened
        float *z;
        float *e;
        float *dz_dt;
        float *p;
        float *log_p;

        // Outgoing prediction -- per batch item, (outChannels, outHeight, outWidth)
        float *mu;

        // im2col scratch -- (inChannels*kernelH*kernelW, outHeight*outWidth)
        // per batch item. Holds THIS layer's own im2col(z) result, computed
        // in CalculateState() and consumed LATER by UpdateWeights() for the
        // weight-gradient GEMM. MUST NOT be overwritten by anything in
        // between (see feedbackScratch below, which exists specifically to
        // avoid the earlier design's footgun of reusing this buffer for
        // UpdateState()'s feedback term too).
        float *colBuffer;

        // Scratch for UpdateState()'s feedback term ONLY -- holds the
        // transposed-GEMM result (colRows x colCols) before the Col2Im
        // scatter into dz_dt. Separate from colBuffer specifically so
        // UpdateState() and UpdateWeights() no longer have a hard ordering
        // dependency on which runs first / whether colBuffer's contents are
        // still valid -- each function now owns its own scratch space.
        float *feedbackScratch;

        // local_grad scratch -- (outChannels, outHeight*outWidth) per batch
        // item, holds (e_above * p_above * mu(f')). Recomputed independently
        // in both UpdateState() and UpdateWeights() (cheap elementwise pass,
        // not worth caching across the two calls).
        float *bottom_up_cols;

        // Repacked scratch for the forward pass's and UpdateWeights()'s
        // single-batched-GEMM paths (see conversation notes: sum_b(A_b @
        // B_b^T) == A_stacked @ B_stacked^T for UpdateWeights; for the
        // forward pass, W @ COLS_stacked computes ALL batch items'
        // independent predictions in one call, since GEMM only sums over
        // rows, never across columns -- no cross-batch mixing occurs).
        // colBuffer/bottom_up_cols are batch-major (batch, row, col);
        // these hold the SAME data repacked row-major (row, batch, col).
        // Dedicated buffers, deliberately NOT reusing feedbackScratch or
        // any other existing buffer, to avoid recreating the cross-function
        // buffer-reuse fragility that was fixed earlier this session.
        float *colsRepacked;
        float *lgRepacked;
        float *muRepacked; // (outChannels, batchSize*colCols) -- forward pass GEMM output, before scatter back to mu

        float lr, ir, pr, lmbda;
        bool isClamped = false;

        ConvPCLayer *layerAbove;
        ConvPCLayer *layerBelow;
        ActivationFn activation;
        DerivativeFn activationDerivative;
        ActivationType activationType;

        friend class PCNDiagnostics;
    };

} // namespace Deep

/*
 * DESIGN NOTES -- conv energy/state/weight derivation, adapted from
 * DiscriminativePCLayer's already-verified formulas:
 *
 * DiscriminativePCLayer (dense):
 *   mu       = f(z @ W^T + b)                    -- GEMM
 *   e        = z_this - mu_incoming(from below)
 *   dz_dt   += -p*e  (own term)
 *              + (e_above * p_above) @ W (feedback, GEMM, no transpose)
 *   dW      += lr * (e_above * p_above)^T @ z     -- GEMM
 *
 * ConvPCLayer (conv):
 *   cols     = Im2Col(z)                          -- (inC*kH*kW, outH*outW)
 *              stored in colBuffer, REUSED LATER by UpdateWeights()
 *   mu       = f(W_flat @ cols + b)                -- GEMM, same as dense
 *   e        = z_this - mu_incoming(from below)    -- UNCHANGED, elementwise
 *   feedback_cols = W_flat^T @ (e_above * p_above) -- GEMM, written to
 *              feedbackScratch (NOT colBuffer)
 *   dz_dt   += -p*e (own term, UNCHANGED)
 *              + Col2Im(feedback_cols)             -- scatter back to (inC,H,W)
 *   dW_flat += lr * (e_above * p_above) @ cols^T   -- GEMM, using colBuffer
 *              as saved by CalculateState() this same step
 *
 * The forward GEMM and the feedback GEMM are transposes of the SAME weight
 * matrix, mirroring how DiscriminativePCLayer's forward and feedback GEMMs
 * both use W with CblasTrans flipped between them.
 */