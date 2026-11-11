#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/Im2Col.h>
#include <deepity/utils/MaxPool2D.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file FullConvPCLayer.h
 * @brief Convolutional analog of FullPCLayer: muPC scaling, residual
 * connections, DKP direct feedback, momentum settling, cross-entropy
 * terminal loss, and ePC settling, each independently toggleable, off
 * by default so plain defaults reproduce SimpleConvPCLayer exactly.
 *
 * Activation is applied directly to `mu` in place (in ComputeMuOnly()),
 * unlike FullPCLayer's dense convention of activating a separate `zF`.
 * UpdateWeights() and ComputeAdjoint() both require `mu` to hold
 * phi'(mu), not phi(raw_mu), by the time they run.
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

  // outHeight/outWidth are mu's own conv-native resolution.
  // poolOutHeight/poolOutWidth are what this layer exposes to the layer
  // above (GetOutHeight/Width(), GetMu()): equal to outHeight/outWidth
  // when pooling is off, smaller when HasPooling() is true.
  int poolH = 1, poolW = 1, poolStrideH = 1, poolStrideW = 1;
  int poolOutHeight, poolOutWidth;

  float lr, ir, fl, lmbda;
  float adamEpsilon = 1e-8f;
  bool isClamped = false;

  // Set whenever SetInferenceRate() changes ir after a CUDA graph
  // capture; see IsIrDirty()/ClearIrDirty().
  bool irDirty = false;

  // Set whenever fl crosses the zero/nonzero boundary after a CUDA
  // graph capture; see IsFlZeroDirty()/ClearFlZeroDirty().
  bool flZeroDirty = false;

  // muPC scaling: mu = Activation(a * conv(z,W) + b) [+ z if useResidual].
  float a = 1.0f;
  bool useResidual = false;
  bool useMuPCInit = false;
  bool usePCXInit = false;

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

  float* W = nullptr;  ///< [outChannels, inChannels*kernelH*kernelW]
  float* b = nullptr;  ///< [outChannels]
  float* mu = nullptr; ///< [batch, outChannels, outHeight, outWidth]
  float* cachedMu = nullptr;
  bool muCacheValid = false;

  // mu's pre-activation value, snapshotted in ComputeMuOnly() right
  // before Activation() overwrites mu in place.
  float* preact = nullptr;
  float* cachedPreact = nullptr;

  /// [batch, outChannels, poolOutHeight, poolOutWidth], GetMu()'s actual
  /// return value when HasPooling(); allocated only then.
  float* pooledMu = nullptr;
  /// Same shape as pooledMu, argmax position per Deep::MaxPool2DForward.
  int* poolArgmax = nullptr;
  /// [batch, outChannels, outHeight, outWidth] scratch UnpoolError()
  /// scatters an incoming pooled error/adjoint into before any code here
  /// combines it with mu (which stays at conv-native resolution).
  float* unpoolScratch = nullptr;

  float* colBuffer = nullptr;
  float* feedbackScratch = nullptr;
  float* bottom_up_cols = nullptr;
  float* colsRepacked = nullptr;
  float* lgRepacked = nullptr;
  float* onesVector = nullptr;

  float* v = nullptr;       ///< Momentum buffer, shape matches dz_dt.
  float* adjoint = nullptr; ///< ePC's chained backward-sweep signal, shape matches dz_dt.

  /// Direct feedback (DFA): per-channel random projection from the
  /// terminal's error, shape [outChannels, terminalSize]. See
  /// DirectFeedbackUpdate().
  float* Psi = nullptr;
  float* projChannel = nullptr; ///< [batch, outChannels], Psi's raw projection.
  float* proj = nullptr; ///< [batch, outChannels, outHeight, outWidth], broadcast of projChannel.

  float* grad_W = nullptr;
  float* grad_b = nullptr;
  float* m_W = nullptr;
  float* v_W = nullptr;
  float* m_b = nullptr;
  float* v_b = nullptr;
  float* grad_Psi = nullptr;
  float* m_Psi = nullptr;
  float* v_Psi = nullptr;

  /// @name GetStateDict() host staging
  /// Host-side mirrors of W/b/Psi and their Adam moments, for
  /// SyncStateDictFromDevice()/SyncStateDictToDevice() to copy into/out
  /// of when `backend` keeps the real buffers in device memory. Sized
  /// lazily by GetStateDict() itself (via EnsureStateDictStaging()).
  ///@{
  mutable std::vector<float> hostW, hostB, hostPsi;
  mutable std::vector<float> hostM_W, hostV_W, hostM_b, hostV_b, hostM_Psi, hostV_Psi;
  ///@}

  /// @brief Resizes the host staging buffers above to match W/b/Psi's
  /// current shapes. Idempotent (a no-op once sizes already match), so
  /// GetStateDict(), SyncStateDictFromDevice(), and SyncStateDictToDevice()
  /// can all call it unconditionally.
  void EnsureStateDictStaging() const noexcept;

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
                  float lmbda, ActivationType aType, ActivationType dType, int poolH = 1,
                  int poolW = 1, int poolStrideH = 1, int poolStrideW = 1,
                  IComputeBackend* backend = nullptr);

  ~FullConvPCLayer() override = default;

  template <typename ArenaT> void BindMemory(ArenaT& arena);
  size_t GetRequiredFloats() const noexcept;
  /// @param distribution Distribution for W only (ignored if usePCXInit
  /// is set); Psi always uses its own fixed fan-scaled init. The
  /// default "normal" (with no "(a,b)") falls back to this class's own
  /// default.
  void RandomizeWeights(std::mt19937& seedGenerator, const char* distribution = "normal") noexcept;

  void SetLayerAbove(FullConvPCLayer* l) noexcept
  {
    layerAbove = l;
  }
  void SetLayerBelow(FullConvPCLayer* l) noexcept
  {
    layerBelow = l;
  }
  void SetTerminalLayer(FullConvPCLayer* l) noexcept
  {
    terminalLayer = l;
  }

  void SetMuPCScale(float a) noexcept
  {
    this->a = a;
  }
  float GetMuPCScale() const noexcept
  {
    return a;
  }
  void SetMuPCInit(bool enabled) noexcept
  {
    useMuPCInit = enabled;
  }
  bool GetMuPCInit() const noexcept
  {
    return useMuPCInit;
  }
  /// @brief Enables PCX-style init: weights uniform in +-1/sqrt(fan_in),
  /// not Kaiming-normal. Takes priority over useMuPCInit if both are set.
  void SetPCXInit(bool enabled) noexcept
  {
    usePCXInit = enabled;
  }
  bool GetPCXInit() const noexcept
  {
    return usePCXInit;
  }

  void SetResidual(bool enabled) noexcept
  {
    useResidual = enabled;
  }
  bool GetResidual() const noexcept
  {
    return useResidual;
  }

  void SetCrossEntropy(bool enabled) noexcept
  {
    useCrossEntropy = enabled;
  }
  bool GetCrossEntropy() const noexcept
  {
    return useCrossEntropy;
  }

  void SetMomentum(bool enabled, float beta = 0.9f) noexcept
  {
    useMomentum = enabled;
    momentumBeta = beta;
  }
  bool GetMomentum() const noexcept
  {
    return useMomentum;
  }

  /// @see Goemaere et al., "ePC: Fast and Deep Predictive Coding in
  /// Digital Simulation", https://arxiv.org/abs/2505.20137
  void SetUseEPC(bool enabled) noexcept
  {
    useEPC = enabled;
  }
  bool GetUseEPC() const noexcept
  {
    return useEPC;
  }

  void SetOptimizer(OptimizerType o) noexcept
  {
    opt = o;
  }
  void SetPsiOptimizer(OptimizerType o) noexcept
  {
    optPsi = o;
  }
  void SetLearningRate(float learningRate) noexcept;
  void SetInferenceRate(float inferenceRate) noexcept
  {
    ir = inferenceRate;
    irDirty = true;
  }
  void SetFeedbackRate(float feedbackRate) noexcept;
  void SetLambda(float lmbda) noexcept
  {
    this->lmbda = lmbda;
  }
  /// @brief Sets Adam/AdamW's denominator stabilizer (default 1e-8).
  void SetAdamEpsilon(float eps) noexcept
  {
    adamEpsilon = eps;
  }

  /// @brief Whether ir has changed since the last successful graph
  /// capture.
  bool IsIrDirty() const noexcept
  {
    return irDirty;
  }
  /// @brief Call only after a graph capture that included this layer's
  /// current ir actually succeeds.
  void ClearIrDirty() noexcept
  {
    irDirty = false;
  }

  /// @brief Whether fl crossed the zero/nonzero boundary since the last
  /// successful graph capture.
  bool IsFlZeroDirty() const noexcept
  {
    return flZeroDirty;
  }
  /// @brief Call only after a graph capture that included this layer's
  /// current fl-is-zero-or-not status actually succeeds.
  void ClearFlZeroDirty() noexcept
  {
    flZeroDirty = false;
  }
  /// @brief Whether this layer's DirectFeedbackUpdate() is currently a
  /// no-op (fl == 0).
  bool IsFlZero() const noexcept
  {
    return fl == 0.0f;
  }

  float CalculateState() noexcept override
  {
    return CalculateState(true);
  }
  float CalculateState(bool needEnergy) noexcept;
  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept override;
  void UpdateWeights() noexcept override;
  /// @brief Subtracts each row's mean from W (see IComputeBackend::CenterRows()).
  void CenterWeights() noexcept override
  {
    if (W != nullptr)
      backend->CenterRows(W, (size_t)outChannels, (size_t)inChannels * kernelH * kernelW);
  }
  void DirectFeedbackUpdate() noexcept;

  /// @brief Pre-grows the backend's cuDNN conv workspace for this
  /// layer's own shape. Must be called once per layer, after Compile()
  /// binds its memory but before the network's first
  /// BeginGraphCapture(). No-op when outChannels==0 (the terminal layer).
  void PrepareConvWorkspace() noexcept;

  /// @brief ePC forward-sweep step: z := layerBelow->GetMu() + e.
  void ReconstructBelief() noexcept;
  /// @brief ePC backward-sweep step. `mu` must currently hold its
  /// activated value (i.e. ComputeMuOnly() ran this step and nothing
  /// has touched `mu` since); on return `mu` holds phi'(mu) instead,
  /// satisfying the same invariant UpdateState() relies on for
  /// UpdateWeights() -- do NOT call ComputeMuOnly() again before
  /// UpdateWeights() runs.
  /// @param adjointAboveScale -1.0 for the topmost hidden layer, 1.0
  /// (default) otherwise.
  void ComputeAdjoint(const float* adjointAbove, float adjointAboveScale = 1.0f) noexcept;
  /// @brief e := (1-ir)*e - ir*adjoint. Call ComputeAdjoint() first.
  void UpdateErrorEPC() noexcept;
  const float* GetAdjoint() const noexcept
  {
    return adjoint;
  }
  /// @brief Converts `mu` from its activated value to phi'(mu) in
  /// place. Call under ePC for a clamped weight-bearing layer that
  /// ComputeAdjoint() never touches, before UpdateWeights(). Do not
  /// call on a layer ComputeAdjoint() already ran this step.
  void EnsureMuHoldsDerivative() noexcept;

  void ClampState(const std::vector<float>& inputData) noexcept;
  void UnclampState() noexcept;
  void ResetState() noexcept;
  bool IsClamped() const noexcept
  {
    return isClamped;
  }
  void InvalidateMuCache() noexcept
  {
    muCacheValid = false;
  }

  float* GetBeliefs() noexcept override
  {
    return z;
  }
  const float* GetErrors() const noexcept override
  {
    return e;
  }
  const float* GetMu() const noexcept
  {
    return HasPooling() ? pooledMu : mu;
  }
  const float* GetWeights() const noexcept
  {
    return W;
  }
  /// @brief Mutable overload of GetWeights() above, needed so backend
  /// copies (e.g. CopyFromHost in set_weights bindings) can write into W.
  float* GetWeights() noexcept
  {
    return W;
  }
  const float* GetDirectFeedbackWeights() const noexcept
  {
    return Psi;
  }
  const float* GetBiases() const noexcept
  {
    return b;
  }
  /// @brief Mutable overload of GetBiases() above, needed so backend
  /// copies (e.g. CopyFromHost in set_biases bindings) can write into b.
  float* GetBiases() noexcept
  {
    return b;
  }
  /// @brief Returns the compute backend this layer routes all math
  /// through, so Python bindings can safely copy device-resident
  /// buffers to the host instead of exposing a raw device pointer
  /// as if it were host memory (see LayerBindings.cpp's
  /// ViewOwnedBy / CopyFromBackend).
  IComputeBackend* GetBackend() const noexcept
  {
    return backend.get();
  }

  /// @brief Returns {error mean, error RMS, weight L2 norm, fraction of
  /// mu elements exactly 0} for this layer's current state. Call right
  /// after TrainStep() for diagnostics on the state that drove the last
  /// weight update.
  std::vector<float> GetDiagnosticStats() const noexcept;

  /// @copydoc Deep::Layer::GetStateDict
  /// @return {} if outChannels == 0 (this layer owns no W/b/Psi).
  /// Otherwise W, b, Psi, and each of their Adam moments (m_W, v_W, m_b,
  /// v_b, m_Psi, v_Psi), pointing at this layer's host staging buffers
  /// -- always safe for ModelIO to read or write directly, regardless
  /// of backend.
  std::map<std::string, TensorDescriptor> GetStateDict() const override;
  /// @copydoc Deep::Layer::SyncStateDictFromDevice
  void SyncStateDictFromDevice() const override;
  /// @copydoc Deep::Layer::SyncStateDictToDevice
  void SyncStateDictToDevice() override;

  size_t GetBatchSize() const noexcept override
  {
    return batchSize;
  }
  size_t GetInputSize() const noexcept override
  {
    return (size_t)inChannels * inHeight * inWidth;
  }
  size_t GetOutputSize() const noexcept override
  {
    return outChannels > 0 ? (size_t)outChannels * poolOutHeight * poolOutWidth : 0;
  }
  int GetInChannels() const noexcept
  {
    return inChannels;
  }
  int GetOutChannels() const noexcept
  {
    return outChannels;
  }
  int GetInHeight() const noexcept
  {
    return inHeight;
  }
  int GetInWidth() const noexcept
  {
    return inWidth;
  }
  int GetOutHeight() const noexcept
  {
    return poolOutHeight;
  }
  int GetOutWidth() const noexcept
  {
    return poolOutWidth;
  }
  int GetKernelH() const noexcept
  {
    return kernelH;
  }
  int GetKernelW() const noexcept
  {
    return kernelW;
  }
  int GetTerminalSize() const noexcept
  {
    return terminalSize;
  }

  bool HasPooling() const noexcept
  {
    return poolH > 1 || poolW > 1 || poolStrideH > 1 || poolStrideW > 1;
  }

protected:
  /// @brief When HasPooling(), scatters @p pooledError (sized
  /// poolOutHeight*poolOutWidth) into unpoolScratch at conv-native
  /// resolution and returns that; otherwise returns @p pooledError
  /// unchanged.
  const float* UnpoolError(const float* pooledError) noexcept;
};
} // namespace Deep
