#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Im2Col.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file FullConvPCLayer.h
 * @brief Convolutional analog of FullPCLayer: every PC variant this
 * codebase knows about, each independently toggleable, off by default so
 * plain defaults reproduce SimpleConvPCLayer exactly (NOT the
 * precision-weighted ConvPCLayer -- FullPCLayer's own dense lineage is
 * DirectKPPCLayer, which has no precision-weighting either).
 *
 * Conv layers use a DIFFERENT activation-placement convention than
 * FullPCLayer: dense keeps `mu` raw/linear and activates `z` into a
 * separate `zF` used as the GEMM input; here (matching
 * ConvPCLayer/SimpleConvPCLayer exactly) the activation is applied
 * directly to `mu` in ComputeMuOnly(). This means UpdateWeights() and
 * ComputeAdjoint() both rely on `mu` holding phi'(mu) (not phi(raw_mu))
 * by the time they run -- see UpdateState()/ComputeAdjoint()'s own docs.
 */

namespace Deep
{
/// @brief Convolutional PC layer combining DKP direct feedback, muPC
/// scaling, residual connections, momentum settling, cross-entropy
/// terminal loss, iPC, and ePC settling, every extra off by default.
class FullConvPCLayer : public Layer
{
protected:
  int inChannels, outChannels;
  int inHeight, inWidth;
  int outHeight, outWidth;
  int kernelH, kernelW;
  int strideH, strideW;
  int padH, padW;
  int batchSize;
  int terminalSize; ///< Size of the network's final output layer (flat).

  float lr, ir, fl, lmbda;
  bool isClamped = false;

  // ir (unlike lr/fl) has no device-resident buffer: it's consumed BY
  // VALUE inside UpdateState()/ComputeAdjoint()'s AxpyInto()/Scale() calls,
  // which CUDA graph capture bakes into the captured kernel launch
  // parameters. A later SetInferenceRate() call updates this host member
  // but has zero effect on an already-captured graph's replays -- unlike
  // lr/fl, which read through lr_device/fl_device each replay. This flag
  // lets FullConvPCNetwork::TrainStep() force a recapture whenever ir
  // actually changes, instead of silently replaying a stale value.
  bool irDirty = false;

  // muPC scaling: mu = Activation(a * conv(z,W) + b) [+ z if useResidual].
  float a = 1.0f;
  bool useResidual = false;
  bool useMuPCInit = false;

  bool useMomentum = false;
  float momentumBeta = 0.9f;

  bool useCrossEntropy = false;
  bool useEPC = false;

  FullConvPCLayer* layerAbove = nullptr;
  FullConvPCLayer* layerBelow = nullptr;
  FullConvPCLayer* terminalLayer = nullptr;

  ActivationType activationType;
  ActivationType derivativeType;
  OptimizerType opt = OptimizerType::SGD;
  OptimizerType optPsi = OptimizerType::SGD;

  using BackendDeleter = void (*)(IComputeBackend*);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend;

  float* z = nullptr;
  float* e = nullptr;
  float* dz_dt = nullptr;
  float* rowEnergies = nullptr; ///< Per-row energy scratch, cross-entropy path only.

  float* W = nullptr;       ///< [outChannels, inChannels*kernelH*kernelW]
  float* b = nullptr;       ///< [outChannels]
  float* mu = nullptr;      ///< [batch, outChannels, outHeight, outWidth]
  float* cachedMu = nullptr;
  bool muCacheValid = false;

  float* colBuffer = nullptr;
  float* feedbackScratch = nullptr;
  float* bottom_up_cols = nullptr;
  float* colsRepacked = nullptr;
  float* lgRepacked = nullptr;
  float* onesVector = nullptr;

  float* v = nullptr;       ///< Momentum buffer, shape matches dz_dt.
  float* adjoint = nullptr; ///< ePC's chained backward-sweep signal, shape matches dz_dt.

  /// Direct feedback (DFA): per-channel random projection from the
  /// terminal's error, shape [outChannels, terminalSize] -- NOT
  /// per-pixel (that would tie its size to a specific spatial
  /// resolution). See DirectFeedbackUpdate().
  float* Psi = nullptr;
  float* projChannel = nullptr; ///< [batch, outChannels], Psi's raw projection.
  float* proj = nullptr;        ///< [batch, outChannels, outHeight, outWidth], broadcast of projChannel.

  float* grad_W = nullptr;
  float* grad_b = nullptr;
  float* m_W = nullptr;
  float* v_W = nullptr;
  float* m_b = nullptr;
  float* v_b = nullptr;
  float* grad_Psi = nullptr;
  float* m_Psi = nullptr;
  float* v_Psi = nullptr;

  int* t_device = nullptr;
  float* lr_device = nullptr;
  int* tPsi_device = nullptr;
  float* fl_device = nullptr;

  std::unique_ptr<MemoryArena> localArena;

public:
  /// @param terminalSize the size of the network's final output layer
  /// (e.g. 10 for MNIST), required directly since the true terminal
  /// layer isn't known until the whole network is assembled.
  FullConvPCLayer(int inChannels, int outChannels, int inHeight, int inWidth, int kernelH,
                  int kernelW, int strideH, int strideW, int padH, int padW, int terminalSize,
                  int batchSize, float learningRate, float inferenceRate, float feedback,
                  float lmbda, ActivationType aType, ActivationType dType,
                  IComputeBackend* backend = nullptr);

  ~FullConvPCLayer() override = default;

  template <typename ArenaT> void BindMemory(ArenaT& arena);
  size_t GetRequiredFloats() const noexcept;
  void RandomizeWeights(std::mt19937& seedGenerator) noexcept;

  void SetLayerAbove(FullConvPCLayer* l) noexcept { layerAbove = l; }
  void SetLayerBelow(FullConvPCLayer* l) noexcept { layerBelow = l; }
  void SetTerminalLayer(FullConvPCLayer* l) noexcept { terminalLayer = l; }

  void SetMuPCScale(float a) noexcept { this->a = a; }
  float GetMuPCScale() const noexcept { return a; }
  void SetMuPCInit(bool enabled) noexcept { useMuPCInit = enabled; }
  bool GetMuPCInit() const noexcept { return useMuPCInit; }

  void SetResidual(bool enabled) noexcept { useResidual = enabled; }
  bool GetResidual() const noexcept { return useResidual; }

  void SetCrossEntropy(bool enabled) noexcept { useCrossEntropy = enabled; }
  bool GetCrossEntropy() const noexcept { return useCrossEntropy; }

  void SetMomentum(bool enabled, float beta = 0.9f) noexcept
  {
    useMomentum = enabled;
    momentumBeta = beta;
  }
  bool GetMomentum() const noexcept { return useMomentum; }

  /// @see Goemaere et al., "ePC: Fast and Deep Predictive Coding in
  /// Digital Simulation", https://arxiv.org/abs/2505.20137
  void SetUseEPC(bool enabled) noexcept { useEPC = enabled; }
  bool GetUseEPC() const noexcept { return useEPC; }

  void SetOptimizer(OptimizerType o) noexcept { opt = o; }
  void SetPsiOptimizer(OptimizerType o) noexcept { optPsi = o; }
  void SetLearningRate(float learningRate) noexcept;
  void SetInferenceRate(float inferenceRate) noexcept
  {
    ir = inferenceRate;
    irDirty = true;
  }
  void SetFeedbackRate(float feedbackRate) noexcept;
  void SetLambda(float lmbda) noexcept { this->lmbda = lmbda; }

  /// @brief Whether ir has changed since the last successful graph
  /// capture (see irDirty's own comment for why this exists).
  bool IsIrDirty() const noexcept { return irDirty; }
  /// @brief Call only after a graph capture that included this layer's
  /// current ir actually succeeds.
  void ClearIrDirty() noexcept { irDirty = false; }

  float CalculateState() noexcept override { return CalculateState(true); }
  float CalculateState(bool needEnergy) noexcept;
  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept override;
  void UpdateWeights() noexcept override;
  void DirectFeedbackUpdate() noexcept;

  /// @brief ePC forward-sweep step: z := layerBelow->GetMu() + e.
  void ReconstructBelief() noexcept;
  /// @brief ePC backward-sweep step. `mu` must currently hold its
  /// activated value (i.e. ComputeMuOnly() ran this step and nothing
  /// has touched `mu` since); on return `mu` holds phi'(mu) instead,
  /// satisfying the same invariant UpdateState() relies on for
  /// UpdateWeights() -- do NOT call ComputeMuOnly() again before
  /// UpdateWeights() runs.
  /// @param adjointAboveScale -1.0 for the topmost hidden layer (the
  /// true backprop seed is dLoss/dmu_terminal = -e_terminal under this
  /// codebase's error convention), 1.0 (default) otherwise.
  void ComputeAdjoint(const float* adjointAbove, float adjointAboveScale = 1.0f) noexcept;
  /// @brief e := (1-ir)*e - ir*adjoint. Call ComputeAdjoint() first.
  void UpdateErrorEPC() noexcept;
  const float* GetAdjoint() const noexcept { return adjoint; }
  /// @brief Converts `mu` from its activated value to phi'(mu) in place,
  /// matching UpdateState()'s own fallback branch for a clamped layer.
  /// Needed under ePC for any clamped weight-bearing layer (i.e. the
  /// input layer) that ComputeAdjoint() never touches -- its own
  /// UpdateWeights() still needs mu holding the derivative. Do NOT call
  /// this on a layer ComputeAdjoint() already ran on this step (it would
  /// take the derivative of an already-differentiated value).
  void EnsureMuHoldsDerivative() noexcept;

  void ClampState(const std::vector<float>& inputData) noexcept;
  void UnclampState() noexcept;
  void ResetState() noexcept;
  bool IsClamped() const noexcept { return isClamped; }
  void InvalidateMuCache() noexcept { muCacheValid = false; }

  float* GetBeliefs() noexcept override { return z; }
  const float* GetErrors() const noexcept override { return e; }
  const float* GetMu() const noexcept { return mu; }
  const float* GetWeights() const noexcept { return W; }
  const float* GetDirectFeedbackWeights() const noexcept { return Psi; }
  const float* GetBiases() const noexcept { return b; }

  size_t GetBatchSize() const noexcept override { return batchSize; }
  size_t GetInputSize() const noexcept override
  {
    return (size_t)inChannels * inHeight * inWidth;
  }
  size_t GetOutputSize() const noexcept override
  {
    return outChannels > 0 ? (size_t)outChannels * outHeight * outWidth : 0;
  }
  int GetInChannels() const noexcept { return inChannels; }
  int GetOutChannels() const noexcept { return outChannels; }
  int GetInHeight() const noexcept { return inHeight; }
  int GetInWidth() const noexcept { return inWidth; }
  int GetOutHeight() const noexcept { return outHeight; }
  int GetOutWidth() const noexcept { return outWidth; }
  int GetKernelH() const noexcept { return kernelH; }
  int GetKernelW() const noexcept { return kernelW; }
  int GetTerminalSize() const noexcept { return terminalSize; }
};
} // namespace Deep
