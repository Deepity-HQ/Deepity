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
 * @file FullConvPCLayerT.h
 * @brief Templated counterpart to FullConvPCLayer, over a storage
 * type `T` and an accumulator type `AccumT` -- see FullPCLayerT.h's
 * own doc comment for the T/AccumT split's full rationale, and
 * SimpleConvPCLayerT.h's own class comment for why every conv-specific
 * scratch buffer here (colBuffer, feedbackScratch, bottom_up_cols,
 * colsRepacked, lgRepacked, proj, projChannel, unpoolScratch, pooledMu,
 * v, adjoint) stays T*, not AccumT* -- same deliberate, documented
 * scope reduction (and the same deferred-validation caveat for
 * sub-8-bit T) as SimpleConvPCLayerT/ConvPCLayerT, NOT FullPCLayerT's
 * own fully AccumT-widened derivative-chain treatment. The Adam/AdamW
 * optimizer state (grad_W/m_W/v_W/..., grad_Psi/m_Psi/v_Psi) still
 * follows the usual AccumT-accumulator convention regardless.
 * float/double are entirely unaffected either way.
 *
 * Deliberately does NOT inherit from Layer, same reasoning as
 * FullPCLayerT/SimpleConvPCLayerT. GetStateDict()/SyncStateDict*()/
 * GetDiagnosticStats() are NOT ported here, same as FullPCLayerT's
 * own precedent -- ModelIO<T>/checkpointing is a separate, later step
 * in the templating plan.
 */

namespace Deep
{
/// @brief Templated counterpart to FullConvPCLayer, see this file's
/// own comment for the full rationale.
template <typename T, typename AccumT = T> class FullConvPCLayerT
{
public:
  /// @param terminalSize the size of the network's final output layer
  /// (e.g. 10 for MNIST), required directly since the true terminal
  /// layer isn't known until the whole network is assembled.
  FullConvPCLayerT(int inChannels, int outChannels, int inHeight, int inWidth, int kernelH,
                   int kernelW, int strideH, int strideW, int padH, int padW, int terminalSize,
                   int batchSize, AccumT learningRate, AccumT inferenceRate, AccumT feedback,
                   AccumT lmbda, ActivationType aType, ActivationType dType, int poolH = 1,
                   int poolW = 1, int poolStrideH = 1, int poolStrideW = 1,
                   IComputeBackendT<T, AccumT>* backend = nullptr);

  ~FullConvPCLayerT() = default;

  template <typename ArenaT> void BindMemory(ArenaT& arena);
  size_t GetRequiredElements() const noexcept;
  void RandomizeWeights(std::mt19937& seedGenerator, const char* distribution = "normal") noexcept;

  void SetLayerAbove(FullConvPCLayerT* l) noexcept
  {
    layerAbove = l;
  }
  void SetLayerBelow(FullConvPCLayerT* l) noexcept
  {
    layerBelow = l;
  }
  void SetTerminalLayer(FullConvPCLayerT* l) noexcept
  {
    terminalLayer = l;
  }

  void SetMuPCScale(AccumT a) noexcept
  {
    this->a = a;
  }
  AccumT GetMuPCScale() const noexcept
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

  void SetMomentum(bool enabled, AccumT beta = AccumT(0.9)) noexcept
  {
    useMomentum = enabled;
    momentumBeta = beta;
  }
  bool GetMomentum() const noexcept
  {
    return useMomentum;
  }

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
  void SetLearningRate(AccumT learningRate) noexcept;
  void SetInferenceRate(AccumT inferenceRate) noexcept
  {
    ir = inferenceRate;
    irDirty = true;
  }
  void SetFeedbackRate(AccumT feedbackRate) noexcept;
  void SetLambda(AccumT lmbda) noexcept
  {
    this->lmbda = lmbda;
  }
  void SetAdamEpsilon(AccumT eps) noexcept
  {
    adamEpsilon = eps;
  }

  bool IsIrDirty() const noexcept
  {
    return irDirty;
  }
  void ClearIrDirty() noexcept
  {
    irDirty = false;
  }
  bool IsFlZeroDirty() const noexcept
  {
    return flZeroDirty;
  }
  void ClearFlZeroDirty() noexcept
  {
    flZeroDirty = false;
  }
  bool IsFlZero() const noexcept
  {
    return fl == AccumT(0);
  }

  AccumT CalculateState() noexcept
  {
    return CalculateState(true);
  }
  AccumT CalculateState(bool needEnergy) noexcept;
  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept;
  void UpdateWeights() noexcept;
  void DirectFeedbackUpdate() noexcept;

  void PrepareConvWorkspace() noexcept
  {
    if (outChannels == 0)
      return;
    backend->PrepareConvWorkspace(batchSize, inChannels, inHeight, inWidth, outChannels, kernelH,
                                  kernelW, strideH, strideW, padH, padW);
  }

  void ReconstructBelief() noexcept;
  void ComputeAdjoint(const T* adjointAbove, AccumT adjointAboveScale = AccumT(1)) noexcept;
  void UpdateErrorEPC() noexcept;
  const T* GetAdjoint() const noexcept
  {
    return adjoint;
  }
  void EnsureMuHoldsDerivative() noexcept;

  void ClampState(const std::vector<T>& inputData) noexcept;
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

  T* GetBeliefs() noexcept
  {
    return z;
  }
  const T* GetErrors() const noexcept
  {
    return e;
  }
  const T* GetMu() const noexcept
  {
    return HasPooling() ? pooledMu : mu;
  }
  const T* GetWeights() const noexcept
  {
    return W;
  }
  T* GetWeights() noexcept
  {
    return W;
  }
  const T* GetDirectFeedbackWeights() const noexcept
  {
    return Psi;
  }
  const T* GetBiases() const noexcept
  {
    return b;
  }
  T* GetBiases() noexcept
  {
    return b;
  }
  IComputeBackendT<T, AccumT>* GetBackend() const noexcept
  {
    return backend.get();
  }

  size_t GetBatchSize() const noexcept
  {
    return batchSize;
  }
  size_t GetInputSize() const noexcept
  {
    return (size_t)inChannels * inHeight * inWidth;
  }
  size_t GetOutputSize() const noexcept
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
  const T* UnpoolError(const T* pooledError) noexcept;

  int inChannels, outChannels;
  int inHeight, inWidth;
  int outHeight, outWidth;
  int kernelH, kernelW;
  int strideH, strideW;
  int padH, padW;
  int batchSize;
  int terminalSize;

  int poolH = 1, poolW = 1, poolStrideH = 1, poolStrideW = 1;
  int poolOutHeight, poolOutWidth;

  AccumT lr, ir, fl, lmbda;
  AccumT adamEpsilon = AccumT(1e-8);
  bool isClamped = false;

  bool irDirty = false;
  bool flZeroDirty = false;

  AccumT a = AccumT(1);
  bool useResidual = false;
  bool useMuPCInit = false;
  bool usePCXInit = false;

  bool useMomentum = false;
  AccumT momentumBeta = AccumT(0.9);

  bool useCrossEntropy = false;
  bool useEPC = false;

  FullConvPCLayerT* layerAbove = nullptr;
  FullConvPCLayerT* layerBelow = nullptr;
  FullConvPCLayerT* terminalLayer = nullptr;

  ActivationType activationType;
  ActivationType derivativeType;
  OptimizerType opt = OptimizerType::SGD;
  OptimizerType optPsi = OptimizerType::SGD;

  using BackendDeleter = void (*)(IComputeBackendT<T, AccumT>*);
  std::unique_ptr<IComputeBackendT<T, AccumT>, BackendDeleter> backend;

  T* z = nullptr;
  T* e = nullptr;
  T* dz_dt = nullptr;
  std::vector<AccumT> rowEnergies; ///< Per-row energy scratch, cross-entropy path only.

  T* W = nullptr;
  T* b = nullptr;
  T* mu = nullptr;
  T* cachedMu = nullptr;
  bool muCacheValid = false;

  T* preact = nullptr;
  T* cachedPreact = nullptr;

  T* pooledMu = nullptr;
  // Owned directly, not arena-allocated: the original's "allocate as
  // T, reinterpret_cast to int*" trick only happens to work when
  // sizeof(T)==sizeof(int) (i.e. T=float); for any other T (double,
  // or a 1-byte quantized struct) it either wastes space or writes
  // int32 argmax data across fewer bytes than were actually
  // allocated, corrupting whatever the arena placed right after it.
  std::vector<int> poolArgmax;
  T* unpoolScratch = nullptr;

  T* colBuffer = nullptr;
  T* feedbackScratch = nullptr;
  T* bottom_up_cols = nullptr;
  T* colsRepacked = nullptr;
  T* lgRepacked = nullptr;
  T* onesVector = nullptr;

  T* v = nullptr;
  T* adjoint = nullptr;

  T* Psi = nullptr;
  T* projChannel = nullptr;
  T* proj = nullptr;

  std::vector<AccumT> grad_W;
  std::vector<AccumT> grad_b;
  std::vector<AccumT> m_W;
  std::vector<AccumT> v_W;
  std::vector<AccumT> m_b;
  std::vector<AccumT> v_b;
  std::vector<AccumT> grad_Psi;
  std::vector<AccumT> m_Psi;
  std::vector<AccumT> v_Psi;

  int t_device = 0;
  AccumT lr_device = AccumT(0);
  int tPsi_device = 0;
  AccumT fl_device = AccumT(0);

  std::unique_ptr<MemoryArenaT<T>> localArena;
};
} // namespace Deep
