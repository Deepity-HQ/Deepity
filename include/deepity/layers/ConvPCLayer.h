#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Im2Col.h>
#include <deepity/utils/MemoryArena.h>
#include <map>
#include <memory>
#include <random>
#include <vector>

/**
 * @file ConvPCLayer.h
 * @brief Convolutional counterpart to DiscriminativePCLayer, precision-
 * weighted, routed through IComputeBackend for GPU portability.
 *
 * @note CalculateState/UpdateState/UpdateWeights math, including the
 * Col2Im-based feedback term, is finite-difference verified on CPU
 * (tConvPCLayerStateVerify.cpp, tConvDiagnose.cpp). The GPU path is
 * structurally reviewed only, not compiled or run.
 *
 * Layout convention: NCHW, row-major, contiguous per channel per batch item.
 *
 * @note Single-instance layer; scratch/column buffers are bound into a
 * MemoryArena rather than stored in a container.
 * @version 2.0
 * @date 2026-09-27
 * @author Jack Rose
 */

namespace Deep {

/// @brief Convolutional Predictive Coding layer (im2col-based),
/// precision-weighted, routed through IComputeBackend for GPU portability.
class ConvPCLayer : public Layer {
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
  /// @param backend Compute backend to run on; defaults to a new
  /// CPUBackend if nullptr.
  ConvPCLayer(int inChannels, int outChannels, int inHeight, int inWidth,
              int kernelH, int kernelW, int strideH = 1, int strideW = 1,
              int padH = 0, int padW = 0, int batchSize = 1,
              float learningRate = 1e-6f, float inferenceRate = 0.1f,
              float precisionRate = 0.01f, float lmbda = 1e-2f,
              ActivationType aType = ActivationType::RELU,
              ActivationType dType = ActivationType::dRELU,
              IComputeBackend *backend = nullptr);

  /// @brief Calculate energy/prediction errors for this layer.
  /// @return This layer's energy contribution at the current state.
  float CalculateState() noexcept override { return CalculateState(true); }
  /// @brief Same as CalculateState(), but can skip computing (and
  /// returning) the energy. On CUDABackend, needEnergy=false also means
  /// no blocking host sync, required for any settling-loop caller that
  /// runs this inside a captured CUDA graph region (see
  /// ConvPCNetwork::TrainStep()'s GPU branch).
  /// @param needEnergy Whether to compute and return the energy.
  float CalculateState(bool needEnergy) noexcept;
  /// @brief Update latent beliefs (z/r) via inference gradient.
  void UpdateState() noexcept override;
  /// @brief Hebbian/gradient weight update.
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

  /// @brief Returns this layer's top-down prediction buffer.
  const float *GetMu() const noexcept { return mu; }

  /// @brief Flattened element count per batch item (inChannels*H*W).
  size_t GetInputSize() const noexcept override {
    return (size_t)inChannels * inHeight * inWidth;
  }
  /// @brief Flattened element count of this layer's OUTGOING
  /// prediction (outChannels*outH*outW), 0 for a terminal layer.
  size_t GetOutputSize() const noexcept override {
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
  /// @return Pointer to (inChannels*inHeight*inWidth,) precisions, one
  /// per own-position, NOT per outChannel, despite this layer's
  /// outgoing prediction being channel-shaped; precision weights THIS
  /// layer's own error against layerBelow, which is input-shaped.
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
  /// @param learningRate The new learning rate.
  void SetLearningRate(float learningRate) noexcept;
  /// @brief Sets the learning rate used for internal-state updates.
  /// @param inferenceRate The new inference rate.
  void SetInferenceRate(float inferenceRate) noexcept { ir = inferenceRate; }
  /// @brief Sets the learning rate used for precision updates.
  /// @param precisionRate The new precision rate.
  void SetPrecisionRate(float precisionRate) noexcept { pr = precisionRate; }
  /// @brief Sets the weight-decay (L2 regularization) coefficient.
  /// @param l The new lambda value.
  void SetLambda(float l) noexcept { lmbda = l; }
  /// @brief Sets the weight optimizer. Call BEFORE Compile().
  void SetOptimizer(const OptimizerType o) noexcept { opt = o; }
  /// @brief Whether ClampState() is currently active on this layer.
  bool IsClamped() const noexcept { return isClamped; }

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

  std::map<std::string, TensorDescriptor> GetStateDict() const override;

  /// @brief Rebuilds log_p from p, required after a checkpoint load
  /// that only persists p (mirrors DiscriminativePCLayer's fix for the
  /// same p/log_p desync issue found in ModelIO::Load()).
  void ResyncLogPrecision() noexcept;

  /// @brief Fast-path forward projection that skips state initialization
  void ComputeMuOnly() noexcept;

  /// @brief Returns this layer's configured activation type.
  ActivationType GetActivationType() const noexcept { return activationType; }
  /// @brief Returns this layer's configured activation-derivative type.
  ActivationType GetDerivativeType() const noexcept { return derivativeType; }

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
  /// @tparam ArenaT Either MemoryArena or DeviceMemoryArena.
  /// @param arena The arena to bind into.
  template <typename ArenaT>
  void BindMemory(ArenaT &arena);

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

  float *z = nullptr; ///< Beliefs/activations: (inChannels, inHeight,
                      ///< inWidth) per batch item
  float *e = nullptr; ///< Prediction errors: same shape as z.
  float *dz_dt = nullptr; ///< State derivatives: same shape as z.
  float *p = nullptr;     ///< Precisions: (inChannels*inHeight*inWidth,),
                          ///< see GetPrecisions()'s doc for why this is
                          ///< input-shaped, not outChannels-shaped.
  float *log_p = nullptr; ///< Log-precisions: same shape as p.
  /// @}

  /// @brief Predictions from above (incoming error feedback)
  float *mu = nullptr;

  /// @name Scratch buffers for convolution operations
  /// @{
  /// Im2Col intermediate: (inChannels*kernelH*kernelW, outHeight*outWidth) per
  /// batch item. Holds this layer's im2col(z) result computed in
  /// CalculateState(). Reused by UpdateWeights() for weight-gradient GEMM. Must
  /// NOT be overwritten between CalculateState() and UpdateWeights().
  float *colBuffer = nullptr;

  /// Feedback term scratch: holds transposed-GEMM result before Col2Im scatter
  /// into dz_dt. Separate from colBuffer to avoid ordering dependencies between
  /// UpdateState() and UpdateWeights().
  float *feedbackScratch = nullptr;

  /// Bottom-up error modulation: (outChannels, outHeight*outWidth) per batch
  /// item. Holds (e_above * p_above * mu(f')). Recomputed independently in both
  /// UpdateState() and UpdateWeights() (cheap elementwise, not cached).
  float *bottom_up_cols = nullptr;

  /// Repacked column buffer: row-major layout (row, batch, col) for
  /// single-batched GEMM.
  float *colsRepacked = nullptr;
  /// Repacked bottom-up gradient: row-major layout for efficient GEMM
  /// operations.
  float *lgRepacked = nullptr;

  /// All-ones vector for the bias-gradient GEMM trick (grad_b =
  /// lgRepacked @ ones), matching SimpleConvPCLayer::UpdateWeights()'s
  /// own use of the same trick.
  float *onesVector = nullptr;
  /// @}

  /// @brief Cache for the clamped input forward-projection
  float *cachedMu = nullptr;
  /// @brief Flag to determine if the mu cache is currently valid
  bool muCacheValid = false;

  float *grad_W = nullptr;
  float *grad_b = nullptr;
  float *m_W = nullptr;
  float *v_W = nullptr;
  float *m_b = nullptr;
  float *v_b = nullptr;

  /// @brief Device-resident Adam step count and learning rate, same
  /// reasoning as SimpleConvPCLayer/SimplePCLayer's own ports.
  int *t_device = nullptr;
  float *lr_device = nullptr;

  float lr, ir, pr, lmbda;
  bool isClamped = false;

  ConvPCLayer *layerAbove = nullptr;
  ConvPCLayer *layerBelow = nullptr;
  ActivationType activationType;
  ActivationType derivativeType;
  OptimizerType opt = OptimizerType::SGD;

  using BackendDeleter = void (*)(IComputeBackend *);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend;
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
 *   &+ (e_{above} \cdot p_{above}) \cdot W && \text{(feedback, GEMM, no
 * transpose)} \\ dW &+= lr \cdot (e_{above} \cdot p_{above})^T \cdot z &&
 * \text{GEMM}
 * \end{align}
 * \f]
 *
 * @subsection conv_formulas ConvPCLayer (conv)
 * \f[
 * \begin{align}
 *   cols &= \text{Im2Col}(z) && (inC \cdot kH \cdot kW, outH \cdot outW) \\
 *   &\text{stored in colBuffer, REUSED LATER by UpdateWeights()} \\
 *   \mu &= f(W_{flat} \cdot cols + b) && \text{GEMM, same as dense} \\
 *   e &= z_{this} - \mu_{incoming}(\text{from below}) && \text{UNCHANGED,
 * elementwise} \\
 *   feedback\_cols &= W_{flat}^T \cdot (e_{above} \cdot p_{above}) &&
 * \text{GEMM, written to feedbackScratch (NOT colBuffer)} \\
 *   \frac{dz}{dt} &+= -p \cdot e && \text{(own term, UNCHANGED)} \\
 *   &+ \text{Col2Im}(feedback\_cols) && \text{(scatter back to $(inC,H,W)$)} \\
 *   dW_{flat} &+= lr \cdot (e_{above} \cdot p_{above}) \cdot cols^T &&
 * \text{GEMM, using colBuffer saved by CalculateState()} \\
 * \end{align}
 * \f]
 *
 * The forward GEMM and the feedback GEMM are transposes of the SAME weight
 * matrix, mirroring how DiscriminativePCLayer's forward and feedback GEMMs
 * both use W with CblasTrans flipped between them.
 */
