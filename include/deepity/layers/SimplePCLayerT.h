#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file SimplePCLayerT.h
 * @brief Templated counterpart to SimplePCLayer, over a storage type
 * `T` and an accumulator type `AccumT`.
 *
 * Unlike SimplePCLayer, has no raw function-pointer constructor; use
 * the ActivationType constructor instead. Does not inherit from Layer.
 */

namespace Deep
{
/// @brief Templated counterpart to SimplePCLayer.
template <typename T, typename AccumT = T> class SimplePCLayerT
{
protected:
  size_t size;      ///< This layer's own belief (z) size.
  size_t nextSize;  ///< Size of the layer above (0 marks a terminal layer).
  size_t batchSize; ///< Batch size.

  AccumT lr;    ///< Learning rate for weight updates.
  AccumT ir;    ///< Inference rate (Euler integration step size).
  AccumT lmbda; ///< Weight decay (L2 regularization) coefficient.

  bool isClamped = false; ///< Whether ClampState() is currently active.

  SimplePCLayerT* layerAbove = nullptr; ///< The next layer up, or nullptr if this is terminal.
  SimplePCLayerT* layerBelow = nullptr; ///< The next layer down, or nullptr if this is the input layer.

  ActivationType activationType; ///< This layer's activation type.
  OptimizerType opt = OptimizerType::SGD; ///< Optimizer for W/b.

  using BackendDeleter = void (*)(IComputeBackendT<T, AccumT>*);
  std::unique_ptr<IComputeBackendT<T, AccumT>, BackendDeleter> backend; ///< This layer's compute backend.

  T* W = nullptr; ///< Weight matrix, shape [nextSize, size].
  T* b = nullptr; ///< Bias vector, length nextSize.
  T* e = nullptr; ///< This layer's error buffer, shape [batchSize, size].
  T* z = nullptr; ///< This layer's belief buffer, shape [batchSize, size].

  T* mu = nullptr;       ///< Outgoing prediction buffer, shape [batchSize, nextSize].
  T* cachedMu = nullptr; ///< Cached mu, valid when muCacheValid is true.
  T* zF = nullptr;       ///< This layer's own activated belief, phi(z).

  std::vector<AccumT> feedbackScratch;      ///< UpdateState()'s feedback GEMM output, e_above@W.
  std::vector<AccumT> feedbackDerivScratch; ///< UpdateState()'s f'(z), see UpdateState()'s own body.

  bool useCrossEntropy = false; ///< Whether softmax cross-entropy energy is enabled, see SetCrossEntropy().
  std::vector<AccumT> rowEnergies; ///< Per-row energy scratch for the cross-entropy path.

  std::vector<AccumT> grad_W; ///< W's gradient scratch buffer.
  std::vector<AccumT> grad_b; ///< b's gradient scratch buffer.
  std::vector<AccumT> m_W;    ///< Adam/AdamW first-moment buffer for W.
  std::vector<AccumT> v_W;    ///< Adam/AdamW second-moment buffer for W.
  std::vector<AccumT> m_b;    ///< Adam/AdamW first-moment buffer for b.
  std::vector<AccumT> v_b;    ///< Adam/AdamW second-moment buffer for b.

  int t_device = 0;             ///< Adam/AdamW step count for W/b.
  AccumT lr_device = AccumT(0); ///< Mirror of `lr`, passed by address to the backend.

  AccumT muCacheThreshold = AccumT(-1); ///< -1 = disabled; see SetMuCacheThreshold().
  bool muCacheValid = false;           ///< true once mu computed at least once since ClampState().

  std::unique_ptr<MemoryArenaT<T>>
      localArena; ///< This layer's own arena, used when it isn't bound into a network-wide one.

public:
  /// @param size Size of this layer's own belief (z)
  /// @param nextSize Size of the layer above's belief; 0 marks a terminal layer
  /// @param batchSize Batch size
  /// @param learningRate Learning rate for weight updates
  /// @param inferenceRate Inference rate (Euler integration step size)
  /// @param lmbda Weight decay (L2 regularization) coefficient
  /// @param aType Activation type
  /// @param dType Unused; the derivative type is derived from @p aType
  /// internally. Kept for call-site parity with SimplePCLayer.
  /// @param backend Compute backend to run on; defaults to a new
  /// CPUBackendT<T, AccumT> if nullptr.
  SimplePCLayerT(size_t size, size_t nextSize, size_t batchSize, AccumT learningRate,
                AccumT inferenceRate, AccumT lmbda, ActivationType aType, ActivationType dType,
                IComputeBackendT<T, AccumT>* backend = nullptr);

  ~SimplePCLayerT() = default;

  /// @brief Calculates the total network energy state, and this
  /// layer's outgoing prediction (mu). See SimplePCLayer's own doc comment.
  AccumT CalculateState(bool needEnergy = true) noexcept;

  void UpdateState() noexcept;
  void UpdateWeights() noexcept;

  void ClampState(const std::vector<T>& inputData) noexcept;
  void UnclampState() noexcept;

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
  const T* GetMu() const noexcept
  {
    return mu;
  }
  const T* GetErrors() const noexcept
  {
    return e;
  }
  size_t GetInputSize() const noexcept
  {
    return size;
  }
  size_t GetOutputSize() const noexcept
  {
    return nextSize;
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
  AccumT GetLambda() const noexcept
  {
    return lmbda;
  }

  void SetLearningRate(AccumT learningRate) noexcept;
  void SetInferenceRate(AccumT inferenceRate) noexcept
  {
    ir = inferenceRate;
  }
  void SetLambda(AccumT l) noexcept
  {
    lmbda = l;
  }

  void SetOptimizer(OptimizerType o) noexcept
  {
    opt = o;
  }

  void SetMuCacheThreshold(AccumT threshold) noexcept
  {
    muCacheThreshold = threshold;
  }
  AccumT GetMuCacheThreshold() const noexcept
  {
    return muCacheThreshold;
  }

  /// @brief Computes only mu (forward prediction), skipping error/energy.
  /// @param zFAlreadyComputed See SimplePCLayer's own doc comment.
  void ComputeMuOnly(bool zFAlreadyComputed = false) noexcept;

  void SetCrossEntropy(bool enabled) noexcept
  {
    useCrossEntropy = enabled;
  }
  bool GetCrossEntropy() const noexcept
  {
    return useCrossEntropy;
  }

  void SetLayerAbove(SimplePCLayerT* above) noexcept
  {
    layerAbove = above;
  }
  void SetLayerBelow(SimplePCLayerT* below) noexcept
  {
    layerBelow = below;
  }

  void ResetState() noexcept;

  const SimplePCLayerT& GetLayerAbove() const noexcept
  {
    return *layerAbove;
  }
  const SimplePCLayerT& GetLayerBelow() const noexcept
  {
    return *layerBelow;
  }

  void RandomizeWeights(std::mt19937& twister, const char* distribution = "normal") noexcept;

  ActivationType GetActivationType() const noexcept
  {
    return activationType;
  }

  size_t GetRequiredElements() const noexcept;
  template <typename ArenaT> void BindMemory(ArenaT& arena);
};
} // namespace Deep
