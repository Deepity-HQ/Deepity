#pragma once

#include <cstdlib>
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
 * @file ConvPCLayerT.h
 * @brief Templated counterpart to ConvPCLayer, over a storage type `T`
 * and an accumulator type `AccumT`. Every scratch buffer here
 * (colBuffer, feedbackScratch, bottom_up_cols, colsRepacked,
 * lgRepacked) stays T*, not AccumT*, same as SimpleConvPCLayerT.
 *
 * Does not inherit from Layer. Does not implement GetStateDict()/checkpointing.
 */

namespace Deep
{
/// @brief Templated counterpart to ConvPCLayer.
template <typename T, typename AccumT = T> class ConvPCLayerT
{
public:
  /// @param inChannels,outChannels Input/output channel counts.
  /// @param inHeight,inWidth Input spatial dimensions.
  /// @param kernelH,kernelW Convolution kernel size.
  /// @param strideH,strideW Convolution stride.
  /// @param padH,padW Zero-padding on each spatial dimension.
  /// @param batchSize Batch size.
  /// @param learningRate Learning rate for W/b.
  /// @param inferenceRate Inference rate (Euler integration step size).
  /// @param precisionRate Learning rate for precision.
  /// @param lmbda Weight decay (L2 regularization) coefficient.
  /// @param aType Activation type.
  /// @param dType Activation derivative type.
  /// @param backend Compute backend to run on; defaults to a new
  /// CPUBackendT<T, AccumT> if nullptr.
  ConvPCLayerT(int inChannels, int outChannels, int inHeight, int inWidth, int kernelH,
              int kernelW, int strideH = 1, int strideW = 1, int padH = 0, int padW = 0,
              int batchSize = 1, AccumT learningRate = AccumT(1e-6),
              AccumT inferenceRate = AccumT(0.1), AccumT precisionRate = AccumT(0.01),
              AccumT lmbda = AccumT(1e-2), ActivationType aType = ActivationType::RELU,
              ActivationType dType = ActivationType::dRELU,
              IComputeBackendT<T, AccumT>* backend = nullptr);

  ~ConvPCLayerT() = default;

  AccumT CalculateState() noexcept
  {
    return CalculateState(true);
  }
  AccumT CalculateState(bool needEnergy) noexcept;
  void UpdateState() noexcept;
  void UpdateWeights() noexcept;
  /// @brief Subtracts each row's mean from W (see IComputeBackend::CenterRows()).
  void CenterWeights() noexcept
  {
    if (W != nullptr)
      backend->CenterRows(W, (size_t)outChannels, (size_t)inChannels * kernelH * kernelW);
  }
  /// @brief Updates this layer's precision estimate from current
  /// prediction errors, using the configured precision rate (pr).
  void UpdatePrecision() noexcept;

  void ClampState(const std::vector<T>& inputData) noexcept;
  void UnclampState() noexcept;

  T* GetBeliefs() noexcept
  {
    return z;
  }
  const T* GetErrors() const noexcept
  {
    return e;
  }
  const T* GetPrecisions() const noexcept
  {
    return p;
  }
  size_t GetInputSize() const noexcept
  {
    return (size_t)inChannels * inHeight * inWidth;
  }
  size_t GetOutputSize() const noexcept
  {
    return outChannels > 0 ? (size_t)outChannels * outHeight * outWidth : 0;
  }
  size_t GetBatchSize() const noexcept
  {
    return batchSize;
  }

  const T* GetWeights() const noexcept
  {
    return W;
  }
  T* GetWeights() noexcept
  {
    return W;
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

  AccumT GetLearningRate() const noexcept
  {
    return lr;
  }
  AccumT GetInferenceRate() const noexcept
  {
    return ir;
  }
  AccumT GetPrecisionRate() const noexcept
  {
    return pr;
  }
  AccumT GetLambda() const noexcept
  {
    return lmbda;
  }

  void SetLearningRate(AccumT learningRate) noexcept;
  void SetInferenceRate(AccumT inferenceRate) noexcept
  {
    ir = inferenceRate;
  }
  void SetPrecisionRate(AccumT precisionRate) noexcept
  {
    pr = precisionRate;
  }
  void SetLambda(AccumT l) noexcept
  {
    lmbda = l;
  }
  void SetOptimizer(const OptimizerType o) noexcept
  {
    opt = o;
  }
  bool IsClamped() const noexcept
  {
    return isClamped;
  }

  void InvalidateMuCache() noexcept
  {
    muCacheValid = false;
  }

  void SetLayerAbove(ConvPCLayerT* above) noexcept
  {
    layerAbove = above;
  }
  void SetLayerBelow(ConvPCLayerT* below) noexcept
  {
    layerBelow = below;
  }

  void ComputeMuOnly() noexcept;

  /// @brief Pre-sizes the CUDA backend's cuDNN workspace for this
  /// layer's convolution shape. Must run before the network's first
  /// CUDA graph capture. No-op on CPU.
  void PrepareConvWorkspace() noexcept
  {
    if (outChannels == 0)
      return;
    backend->PrepareConvWorkspace(batchSize, inChannels, inHeight, inWidth, outChannels, kernelH,
                                  kernelW, strideH, strideW, padH, padW);
  }

  void ResetState() noexcept;
  void RandomizeWeights(std::mt19937& twister, const char* distribution = "normal") noexcept;

  /// @brief Rebuilds log_p from p. Call after a checkpoint load that
  /// only restores p.
  void ResyncLogPrecision() noexcept;

  ActivationType GetActivationType() const noexcept
  {
    return activationType;
  }
  ActivationType GetDerivativeType() const noexcept
  {
    return derivativeType;
  }

  const T* GetMu() const noexcept
  {
    return mu;
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
    return outHeight;
  }
  int GetOutWidth() const noexcept
  {
    return outWidth;
  }
  int GetKernelH() const noexcept
  {
    return kernelH;
  }
  int GetKernelW() const noexcept
  {
    return kernelW;
  }

  size_t GetRequiredElements() const noexcept;
  template <typename ArenaT> void BindMemory(ArenaT& arena);

private:
  std::unique_ptr<MemoryArenaT<T>> localArena;

  int inChannels, outChannels;
  int inHeight, inWidth;
  int outHeight, outWidth;
  int kernelH, kernelW;
  int strideH, strideW;
  int padH, padW;
  int batchSize;

  T* W = nullptr;
  T* b = nullptr;

  T* z = nullptr;
  T* e = nullptr;
  T* dz_dt = nullptr;
  T* p = nullptr;
  T* log_p = nullptr;

  T* mu = nullptr;
  T* colBuffer = nullptr;
  T* feedbackScratch = nullptr;
  T* bottom_up_cols = nullptr;
  T* colsRepacked = nullptr;
  T* lgRepacked = nullptr;

  T* onesVector = nullptr;

  T* cachedMu = nullptr;
  bool muCacheValid = false;

  T* preact = nullptr;
  T* cachedPreact = nullptr;

  std::vector<AccumT> grad_W;
  std::vector<AccumT> grad_b;
  std::vector<AccumT> m_W;
  std::vector<AccumT> v_W;
  std::vector<AccumT> m_b;
  std::vector<AccumT> v_b;

  int t_device = 0;
  AccumT lr_device = AccumT(0);

  AccumT lr, ir, pr, lmbda;
  bool isClamped = false;

  ConvPCLayerT* layerAbove = nullptr;
  ConvPCLayerT* layerBelow = nullptr;
  ActivationType activationType;
  ActivationType derivativeType;
  OptimizerType opt = OptimizerType::SGD;

  using BackendDeleter = void (*)(IComputeBackendT<T, AccumT>*);
  std::unique_ptr<IComputeBackendT<T, AccumT>, BackendDeleter> backend;
};

} // namespace Deep
