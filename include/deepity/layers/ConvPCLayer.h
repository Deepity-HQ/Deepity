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
 *
 * @note Single-instance layer; scratch/column buffers are bound into a
 * MemoryArena rather than stored in a container, unlike
 * DiscriminativePCNetwork's vector-of-layers.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
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
        /// @return This layer's energy contribution at the current state.
        float CalculateState() noexcept override;
        /// @brief Update latent beliefs (z/r) via inference gradient.
        /// @warning Feedback term not yet verified—see file-level warning.
        void UpdateState() noexcept override;
        /// @brief Hebbian/gradient weight update.
        /// @warning Weight-gradient path verified; feedback term NOT YET verified—see file-level warning.
        void UpdateWeights() noexcept override;
        /// @brief Updates this layer's precision estimate from current
        /// prediction errors, using the configured precision rate (pr).
        void UpdatePrecision() noexcept;

        /// @brief No-op; exists for Layer interface conformance.
        void Flush() noexcept override {}

        /// @brief Clamps this layer's beliefs to externally-provided data,
        /// preventing them from being updated by UpdateState().
        /// @param inputData Flattened input data matching
        /// (inChannels, inHeight, inWidth) per batch item.
        void ClampState(const std::vector<float> &inputData) noexcept;
        /// @brief Releases a previous ClampState() call, allowing this
        /// layer's beliefs to update normally again.
        void UnclampState() noexcept;

        /// @brief Returns this layer's belief buffer.
        /// @return Pointer to (outChannels, outHeight, outWidth) beliefs
        /// per batch item.
        float *GetBeliefs() noexcept override { return z; }
        /// @brief Returns this layer's prediction-error buffer.
        /// @return Pointer to (outChannels, outHeight, outWidth) errors
        /// per batch item.
        const float *GetErrors() const noexcept override { return e; }

        /// @brief Flattened element count per batch item (inChannels*H*W).
        size_t GetInputSize() const noexcept override { return (size_t)inChannels * inHeight * inWidth; }
        /// @brief Flattened element count of this layer's OUTGOING
        /// prediction (outChannels*outH*outW), 0 for a terminal layer.
        size_t GetOutputSize() const noexcept override
        {
            return outChannels > 0 ? (size_t)outChannels * outHeight * outWidth : 0;
        }
        /// @brief Returns the batch size this layer was constructed with.
        /// @return The batch size.
        size_t GetBatchSize() const noexcept override { return batchSize; }

        /// @brief Returns this layer's weight buffer.
        /// @return Pointer to (outChannels, inChannels*kernelH*kernelW) weights.
        const float *GetWeights() const noexcept { return W; }
        /// @brief Returns this layer's weight buffer.
        /// @return Pointer to (outChannels, inChannels*kernelH*kernelW) weights.
        float *GetWeights() noexcept { return W; }
        /// @brief Returns this layer's bias buffer.
        /// @return Pointer to (outChannels,) biases.
        const float *GetBiases() const noexcept { return b; }
        /// @brief Returns this layer's bias buffer.
        /// @return Pointer to (outChannels,) biases.
        float *GetBiases() noexcept { return b; }
        /// @brief Returns this layer's precision buffer.
        /// @return Pointer to (outChannels,) precisions.
        const float *GetPrecisions() const noexcept { return p; }

        /// @brief Returns the learning rate used for weight updates.
        /// @return The learning rate.
        float GetLearningRate() const noexcept { return lr; }
        /// @brief Returns the learning rate used for internal-state updates.
        /// @return The inference rate.
        float GetInferenceRate() const noexcept { return ir; }
        /// @brief Returns the learning rate used for precision updates.
        /// @return The precision rate.
        float GetPrecisionRate() const noexcept { return pr; }
        /// @brief Returns the weight-decay (L2 regularization) coefficient.
        /// @return Lambda.
        float GetLambda() const noexcept { return lmbda; }

        /// @brief Sets the learning rate used for weight updates.
        /// @param lr The new learning rate.
        void SetLearningRate(float lr) noexcept { this->lr = lr; }
        /// @brief Sets the learning rate used for internal-state updates.
        /// @param ir The new inference rate.
        void SetInferenceRate(float ir) noexcept { this->ir = ir; }
        /// @brief Sets the learning rate used for precision updates.
        /// @param pr The new precision rate.
        void SetPrecisionRate(float pr) noexcept { this->pr = pr; }
        /// @brief Sets the weight-decay (L2 regularization) coefficient.
        /// @param l The new lambda value.
        void SetLambda(float l) noexcept { this->lmbda = l; }

        /// @brief Sets the layer immediately above this one in the network.
        /// @param above Pointer to the layer above; may be nullptr for a
        /// terminal layer.
        void SetLayerAbove(ConvPCLayer *above) noexcept { layerAbove = above; }
        /// @brief Sets the layer immediately below this one in the network.
        /// @param below Pointer to the layer below; may be nullptr for the
        /// input layer.
        void SetLayerBelow(ConvPCLayer *below) noexcept { layerBelow = below; }

        /// @brief Resets this layer's beliefs/errors back to their initial
        /// values, without touching learned weights.
        void ResetState() noexcept;

        /// @brief Randomizes this layer's weights (and biases) in place.
        /// @param twister The classic Mersenne Twister
        void RandomizeWeights(std::mt19937 &twister) noexcept;

        /// @brief Rebuilds log_p from p -- required after a checkpoint load
        /// that only persists p (mirrors DiscriminativePCLayer's fix for the
        /// same p/log_p desync issue found in ModelIO::Load()).
        void ResyncLogPrecision() noexcept;

        /// @brief Fast-path forward projection that skips state initialization
         void ComputeMuOnly() noexcept;

        /// @brief Returns this layer's configured activation type.
        /// @return The activation type.
        ActivationType GetActivationType() const noexcept { return To_AType(activation); }
        /// @brief Returns this layer's configured activation-derivative type.
        /// @return The activation-derivative type.
        ActivationType GetDerivativeType() const noexcept { return To_AType(activationDerivative); }

        /// @brief Returns the number of input channels.
        int GetInChannels() const noexcept { return inChannels; }
        /// @brief Returns the number of output channels (0 for a terminal layer).
        int GetOutChannels() const noexcept { return outChannels; }
        /// @brief Returns the input feature-map height.
        int GetInHeight() const noexcept { return inHeight; }
        /// @brief Returns the input feature-map width.
        int GetInWidth() const noexcept { return inWidth; }
        /// @brief Returns the output feature-map height.
        int GetOutHeight() const noexcept { return outHeight; }
        /// @brief Returns the output feature-map width.
        int GetOutWidth() const noexcept { return outWidth; }
        /// @brief Returns the convolution kernel height.
        int GetKernelH() const noexcept { return kernelH; }
        /// @brief Returns the convolution kernel width.
        int GetKernelW() const noexcept { return kernelW; }

        /// @brief Computes the total number of floats this layer requires
        /// from a MemoryArena (weights, biases, beliefs, errors, and every
        /// scratch buffer combined).
        /// @return The required float count.
        size_t GetRequiredFloats() const noexcept;
        /// @brief Binds this layer's weight/state/scratch buffers into the
        /// supplied arena. Must be called before any other operation.
        /// @param arena The MemoryArena to bind into.
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
        float *W = nullptr; ///< Weights: (outChannels, inChannels*kernelH*kernelW)
        float *b = nullptr; ///< Biases: (outChannels)

        float *z = nullptr;     ///< Beliefs/activations: (outChannels, outHeight, outWidth) per batch item
        float *e = nullptr;     ///< Prediction errors: (outChannels, outHeight, outWidth) per batch item
        float *dz_dt = nullptr; ///< State derivatives: (outChannels, outHeight, outWidth) per batch item
        float *p = nullptr;     ///< Precisions: (outChannels,)
        float *log_p = nullptr; ///< Log-precisions: (outChannels,)
        /// @}

        /// @brief Predictions from above (incoming error feedback)
        float *mu = nullptr;

        /// @name Scratch buffers for convolution operations
        /// @{
        /// Im2Col intermediate: (inChannels*kernelH*kernelW, outHeight*outWidth) per batch item.
        /// Holds this layer's im2col(z) result computed in CalculateState().
        /// Reused by UpdateWeights() for weight-gradient GEMM.
        /// Must NOT be overwritten between CalculateState() and UpdateWeights().
        float *colBuffer = nullptr;

        /// Feedback term scratch: holds transposed-GEMM result before Col2Im scatter into dz_dt.
        /// Separate from colBuffer to avoid ordering dependencies between UpdateState() and UpdateWeights().
        float *feedbackScratch = nullptr;

        /// Bottom-up error modulation: (outChannels, outHeight*outWidth) per batch item.
        /// Holds (e_above * p_above * mu(f')). Recomputed independently in both
        /// UpdateState() and UpdateWeights() (cheap elementwise, not cached).
        float *bottom_up_cols = nullptr;

        /// Repacked column buffer: row-major layout (row, batch, col) for single-batched GEMM.
        /// Holds colBuffer's data transposed from batch-major to enable efficient GEMM operations.
        /// Separate buffer to avoid buffer-reuse fragility across UpdateState() and UpdateWeights().
        float *colsRepacked = nullptr;
        /// Repacked bottom-up gradient: row-major layout for efficient GEMM operations.
        float *lgRepacked = nullptr;
        /// Forward-pass GEMM output: (outChannels, batchSize*colCols), scattered back to mu.
        float *muRepacked = nullptr;
        /// @}

        /// @brief Cache for the clamped input forward-projection 
        float *cachedMu = nullptr;
        /// @brief Flag to determine if the mu cache is currently valid
        bool muCacheValid = false;

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
