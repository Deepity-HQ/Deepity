#pragma once

#include <deepity/backend/IComputeBackend.h>
#include <deepity/layers/Layer.h>
#include <deepity/utils/Activations.h>
#include <deepity/utils/AdamOptimizer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <map>
#include <memory>
#include <random>
#include <vector>

/**
 * @file DirectKPPCLayerT.h
 * @brief Templated counterpart to DirectKPPCLayer, over a storage type
 * `T` and an accumulator type `AccumT`. Does not inherit from Layer.
 */

namespace Deep
{
/// @brief Templated counterpart to DirectKPPCLayer.
template <typename T, typename AccumT = T> class DirectKPPCLayerT
{
protected:
  size_t size;         ///< This layer's own belief (z) size.
  size_t nextSize;     ///< Size of the layer above (0 marks a terminal layer).
  size_t terminalSize; ///< Size of the network's final output layer.
  size_t batchSize;    ///< Batch size.

  AccumT lr;    ///< Learning rate for W.
  AccumT ir;    ///< Inference rate (Euler integration step size).
  AccumT fl;    ///< Feedback rate for Psi.
  AccumT lmbda; ///< Weight decay (L2 regularization) coefficient, shared between W and Psi.

  bool isClamped = false;    ///< Whether ClampState() is currently active.
  bool muCacheValid = false; ///< Whether `cachedMu` holds a valid, up-to-date value.

  DirectKPPCLayerT* layerAbove = nullptr; ///< The next layer up, or nullptr if this is terminal.
  DirectKPPCLayerT* layerBelow = nullptr; ///< The next layer down, or nullptr if the input layer.
  DirectKPPCLayerT* terminalLayer = nullptr; ///< The network's terminal layer, for DirectFeedbackUpdate().

  ActivationType activationType; ///< This layer's activation type.

  OptimizerType opt = OptimizerType::SGD;    ///< Optimizer for W/b.
  OptimizerType optPsi = OptimizerType::SGD; ///< Optimizer for Psi.

  using BackendDeleter = void (*)(IComputeBackendT<T, AccumT>*);
  std::unique_ptr<IComputeBackendT<T, AccumT>, BackendDeleter> backend; ///< This layer's compute backend.

  T* z = nullptr; ///< This layer's belief buffer, shape [batchSize, size].
  T* e = nullptr; ///< This layer's error buffer, shape [batchSize, size].

  T* W = nullptr;        ///< Weight matrix, shape [nextSize, size].
  T* b = nullptr;        ///< Bias vector, length nextSize.
  T* mu = nullptr;       ///< Forward prediction buffer, shape [batchSize, nextSize].
  T* cachedMu = nullptr; ///< Cached mu, valid when muCacheValid is true.
  T* preact = nullptr;   ///< Pre-activation snapshot of mu, see DirectKPPCLayer's own doc comment.
  T* cachedPreact = nullptr; ///< Cached pre-activation, valid alongside cachedMu.

  std::vector<AccumT> derivScratch;  ///< nextSize-shaped scratch holding f'(preact).
  std::vector<AccumT> feedbackError; ///< nextSize-shaped scratch, see DirectKPPCLayer's own doc comment.
  T* Psi = nullptr; ///< Direct feedback (DFA) weight matrix, shape [size, terminalSize].
  std::vector<AccumT> proj; ///< Scratch buffer for DirectFeedbackUpdate()'s W update.

  std::vector<AccumT> grad_W; ///< W's gradient scratch buffer; also SGD's row-sum scratch for grad_b.
  std::vector<AccumT> grad_b; ///< b's gradient scratch buffer; also SGD's row-sum scratch.
  std::vector<AccumT> m_W;    ///< Adam/AdamW first-moment buffer for W.
  std::vector<AccumT> v_W;    ///< Adam/AdamW second-moment buffer for W.
  std::vector<AccumT> m_b;    ///< Adam/AdamW first-moment buffer for b.
  std::vector<AccumT> v_b;    ///< Adam/AdamW second-moment buffer for b.

  std::vector<AccumT> grad_Psi; ///< Psi's gradient scratch buffer (Adam/AdamW only).
  std::vector<AccumT> m_Psi;    ///< Adam/AdamW first-moment buffer for Psi.
  std::vector<AccumT> v_Psi;    ///< Adam/AdamW second-moment buffer for Psi.

  int t_device = 0;             ///< Adam/AdamW step count for W/b.
  AccumT lr_device = AccumT(0); ///< Mirror of `lr`, passed by address to the backend.
  int tPsi_device = 0;          ///< Adam/AdamW step count for Psi.
  AccumT fl_device = AccumT(0); ///< Mirror of `fl`, passed by address to the backend.

  std::vector<AccumT> feedbackScratch; ///< Scratch buffer for the feedback GEMM's output.

  std::unique_ptr<MemoryArenaT<T>>
      localArena; ///< This layer's own arena, used when it isn't bound into a network-wide one.

public:
  /// @param size input size
  /// @param nextSize output size (0 marks a terminal layer)
  /// @param terminalSize the size of the network's final output layer
  /// @param batchSize batch size
  /// @param learningRate learning rate for W
  /// @param inferenceRate inference rate (Euler integration step size)
  /// @param feedback feedback rate for Psi
  /// @param lmbda weight decay (L2 regularization) coefficient
  /// @param aType activation type
  /// @param dType Unused; the derivative type is derived from @p aType
  /// internally. Kept for call-site parity with DirectKPPCLayer.
  /// @param backend Compute backend to run on; defaults to a new
  /// CPUBackendT<T, AccumT> if nullptr.
  DirectKPPCLayerT(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
                   AccumT learningRate, AccumT inferenceRate, AccumT feedback, AccumT lmbda,
                   ActivationType aType, ActivationType dType,
                   IComputeBackendT<T, AccumT>* backend = nullptr);

  ~DirectKPPCLayerT() = default;

  template <typename ArenaT> void BindMemory(ArenaT& arena);
  size_t GetRequiredElements() const noexcept;
  /// @brief Randomizes W and Psi. Psi always uses its own fixed
  /// fan-scaled init, regardless of `distribution`.
  void RandomizeWeights(std::mt19937& seedGenerator, const char* distribution = "normal") noexcept;

  std::map<std::string, TensorDescriptorT<T>> GetStateDict() const;

  void SetLayerAbove(DirectKPPCLayerT* l) noexcept
  {
    layerAbove = l;
  }
  void SetLayerBelow(DirectKPPCLayerT* l) noexcept
  {
    layerBelow = l;
  }
  void SetTerminalLayer(DirectKPPCLayerT* l) noexcept
  {
    terminalLayer = l;
  }

  /// @brief Always computes real energy.
  AccumT CalculateState() noexcept
  {
    return CalculateState(true);
  }
  AccumT CalculateState(bool needEnergy) noexcept;

  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept;
  void UpdateWeights() noexcept;
  /// @brief Subtracts each row's mean from W (see IComputeBackend::CenterRows()).
  void CenterWeights() noexcept
  {
    if (W != nullptr)
      backend->CenterRows(W, nextSize, size);
  }
  /// @brief Perturbs W using the layer above's Psi and the terminal
  /// layer's error, the DFA phase, run once per batch before settling
  /// begins.
  void DirectFeedbackUpdate() noexcept;

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
  }
  void SetFeedbackRate(AccumT feedbackRate) noexcept;
  void SetLambda(AccumT l) noexcept
  {
    lmbda = l;
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
    return mu;
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
    return size;
  }
  size_t GetOutputSize() const noexcept
  {
    return nextSize;
  }
  size_t GetTerminalSize() const noexcept
  {
    return terminalSize;
  }
};
} // namespace Deep
