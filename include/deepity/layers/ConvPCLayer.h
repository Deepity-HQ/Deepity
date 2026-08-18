#pragma once

#include <vector>
#include <stdexcept>
#include <random>
#include <memory>
#include <cstdlib>
#include <deepity/Activations.h>
#include <deepity/layers/Layer.h>
#include <deepity/MemoryArena.h>
#include <deepity/Im2Col.h>

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

        /// @brief Calculate energy/prediction errors for this layer.
        /// @warning Weight-gradient path verified; feedback term NOT YET verified—see file-level warning.
        float CalculateState() noexcept override;
        /// @brief Update latent beliefs (z/r) via inference gradient.
        /// @warning Feedback term not yet verified—see file-level warning.
        void UpdateState() noexcept override;
        /// @brief Hebbian/gradient weight update.
        /// @warning Weight-gradient path verified; feedback term NOT YET verified—see file-level warning.
        void UpdateWeights() noexcept override;
        void UpdatePrecision() noexcept;

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

        /// @name Tensor shape parameters
        /// @{
        int inChannels, outChannels;
        int inHeight, inWidth;
        int outHeight, outWidth;
        int kernelH, kernelW;
        int strideH, strideW;
        int padH, padW;
        int batchSize;
        /// @}

        /// @name Weight and bias buffers
        /// @{
        float *W;  ///< Weights: (outChannels, inChannels*kernelH*kernelW)
        float *b;  ///< Biases: (outChannels)

        float *z;      ///< Beliefs/activations: (outChannels, outHeight, outWidth) per batch item
        float *e;      ///< Prediction errors: (outChannels, outHeight, outWidth) per batch item
        float *dz_dt;  ///< State derivatives: (outChannels, outHeight, outWidth) per batch item
        float *p;      ///< Precisions: (outChannels,)
        float *log_p;  ///< Log-precisions: (outChannels,)
        /// @}

        /// @brief Predictions from above (incoming error feedback)
        float *mu;

        /// @name Scratch buffers for convolution operations
        /// @{
        /// Im2Col intermediate: (inChannels*kernelH*kernelW, outHeight*outWidth) per batch item.
        /// Holds this layer's im2col(z) result computed in CalculateState().
        /// Reused by UpdateWeights() for weight-gradient GEMM.
        /// Must NOT be overwritten between CalculateState() and UpdateWeights().
        float *colBuffer;

        /// Feedback term scratch: holds transposed-GEMM result before Col2Im scatter into dz_dt.
        /// Separate from colBuffer to avoid ordering dependencies between UpdateState() and UpdateWeights().
        float *feedbackScratch;

        /// Bottom-up error modulation: (outChannels, outHeight*outWidth) per batch item.
        /// Holds (e_above * p_above * mu(f')). Recomputed independently in both
        /// UpdateState() and UpdateWeights() (cheap elementwise, not cached).
        float *bottom_up_cols;

        /// Repacked column buffer: row-major layout (row, batch, col) for single-batched GEMM.
        /// Holds colBuffer's data transposed from batch-major to enable efficient GEMM operations.
        /// Separate buffer to avoid buffer-reuse fragility across UpdateState() and UpdateWeights().
        float *colsRepacked;
        /// Repacked bottom-up gradient: row-major layout for efficient GEMM operations.
        float *lgRepacked;
        /// Forward-pass GEMM output: (outChannels, batchSize*colCols), scattered back to mu.
        float *muRepacked;
        /// @}

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

/**
 * @page ConvPCLayer_math ConvPCLayer Mathematical Derivation
 *
 * @section design_notes Design Notes
 *
 * Conv energy/state/weight derivation, adapted from DiscriminativePCLayer's
 * already-verified formulas:
 *
 * @subsection dense_formulas DiscriminativePCLayer (dense)
 * \f[
 * \begin{align}
 *   \mu &= f(z \cdot W^T + b) && \text{GEMM} \\
 *   e &= z_{this} - \mu_{incoming}(\text{from below}) \\
 *   \frac{dz}{dt} &+= -p \cdot e && \text{(own term)} \\
 *   &+ (e_{above} \cdot p_{above}) \cdot W && \text{(feedback, GEMM, no transpose)} \\
 *   dW &+= lr \cdot (e_{above} \cdot p_{above})^T \cdot z && \text{GEMM}
 * \end{align}
 * \f]
 *
 * @subsection conv_formulas ConvPCLayer (conv)
 * \f[
 * \begin{align}
 *   cols &= \text{Im2Col}(z) && (inC \cdot kH \cdot kW, outH \cdot outW) \\
 *   &\text{stored in colBuffer, REUSED LATER by UpdateWeights()} \\
 *   \mu &= f(W_{flat} \cdot cols + b) && \text{GEMM, same as dense} \\
 *   e &= z_{this} - \mu_{incoming}(\text{from below}) && \text{UNCHANGED, elementwise} \\
 *   feedback\_cols &= W_{flat}^T \cdot (e_{above} \cdot p_{above}) && \text{GEMM, written to feedbackScratch (NOT colBuffer)} \\
 *   \frac{dz}{dt} &+= -p \cdot e && \text{(own term, UNCHANGED)} \\
 *   &+ \text{Col2Im}(feedback\_cols) && \text{(scatter back to $(inC,H,W)$)} \\
 *   dW_{flat} &+= lr \cdot (e_{above} \cdot p_{above}) \cdot cols^T && \text{GEMM, using colBuffer saved by CalculateState()} \\
 * \end{align}
 * \f]
 *
 * The forward GEMM and the feedback GEMM are transposes of the SAME weight
 * matrix, mirroring how DiscriminativePCLayer's forward and feedback GEMMs
 * both use W with CblasTrans flipped between them.
 */
