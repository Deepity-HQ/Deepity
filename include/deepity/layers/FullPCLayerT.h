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
 * @file FullPCLayerT.h
 * @brief Templated counterpart to FullPCLayer, over a storage type `T`
 * (the layer's own beliefs/errors/weights/activations) and an
 * accumulator type `AccumT` (the Adam/AdamW optimizer's gradient/
 * moment state).
 *
 * Does not inherit from Layer, so a FullPCLayerT<T> cannot be used
 * polymorphically as a Layer*. Every other layer (SimplePCLayer,
 * DirectKPPCLayer, GaussSeidelPCLayer, the conv family) still derives
 * from Layer as usual.
 */

namespace Deep
{
/// @brief Templated counterpart to FullPCLayer.
template <typename T, typename AccumT = T>
class FullPCLayerT
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

  AccumT a = AccumT(1); ///< muPC forward-scaling factor, see SetMuPCScale().
  bool useResidual = false; ///< Whether the residual/skip connection is enabled, see SetResidual().

  bool useMuPCInit = false; ///< Whether unit-variance W init is enabled, see SetMuPCInit().

  bool isClamped = false;    ///< Whether ClampState() is currently active.
  bool muCacheValid = false; ///< Whether `cachedMu` holds a valid, up-to-date value.

  FullPCLayerT* layerAbove = nullptr; ///< The next layer up, or nullptr if this is terminal.
  FullPCLayerT* layerBelow =
      nullptr; ///< The next layer down, or nullptr if this is the input layer.
  FullPCLayerT* terminalLayer =
      nullptr; ///< The network's terminal layer, for DirectFeedbackUpdate().

  ActivationType activationType; ///< This layer's activation type.

  OptimizerType opt = OptimizerType::SGD;    ///< Optimizer for W/b.
  OptimizerType optPsi = OptimizerType::SGD; ///< Optimizer for Psi.

  /// @brief Custom deleter type for `backend`, needed since
  /// IComputeBackendT<T, AccumT> is an abstract base, see the member below.
  using BackendDeleter = void (*)(IComputeBackendT<T, AccumT>*);
  std::unique_ptr<IComputeBackendT<T, AccumT>, BackendDeleter> backend; ///< This layer's compute backend.

  T* z = nullptr; ///< This layer's belief buffer, shape [batchSize, size].
  T* e = nullptr; ///< This layer's error buffer, shape [batchSize, size].

  T* W = nullptr;        ///< Weight matrix, shape [nextSize, size].
  T* b = nullptr;        ///< Bias vector, length nextSize.
  T* mu = nullptr;       ///< Forward prediction buffer, shape [batchSize, nextSize].
  T* cachedMu = nullptr; ///< Cached mu, valid when muCacheValid is true.
  T* preact = nullptr;   ///< Pre-activation snapshot of mu, see FullPCLayer's own doc comment.
  T* cachedPreact = nullptr; ///< Cached pre-activation, valid alongside cachedMu.
  /// @brief nextSize-shaped scratch holding f'(preact), kept at AccumT
  /// precision (not T) since the raw derivative can underflow to zero
  /// at low-precision T.
  std::vector<AccumT> derivScratch;
  std::vector<AccumT> feedbackError; ///< nextSize-shaped scratch, see FullPCLayer's own doc comment, kept at AccumT precision.
  T* Psi = nullptr;  ///< Direct feedback (DFA) weight matrix, shape [size, terminalSize].
  std::vector<AccumT> proj; ///< Scratch buffer for DirectFeedbackUpdate()'s W update.

  std::vector<AccumT> grad_W; ///< W's gradient scratch buffer.
  std::vector<AccumT> grad_b; ///< b's gradient scratch buffer.
  std::vector<AccumT> m_W;    ///< Adam/AdamW first-moment buffer for W.
  std::vector<AccumT> v_W;    ///< Adam/AdamW second-moment buffer for W.
  std::vector<AccumT> m_b;    ///< Adam/AdamW first-moment buffer for b.
  std::vector<AccumT> v_b;    ///< Adam/AdamW second-moment buffer for b.

  std::vector<AccumT> grad_Psi; ///< Psi's gradient scratch buffer.
  std::vector<AccumT> m_Psi;    ///< Adam/AdamW first-moment buffer for Psi.
  std::vector<AccumT> v_Psi;    ///< Adam/AdamW second-moment buffer for Psi.

  int t_device = 0;        ///< Adam/AdamW step count for W/b.
  AccumT lr_device = AccumT(0); ///< Mirror of `lr`, passed by address to the backend.
  int tPsi_device = 0;      ///< Adam/AdamW step count for Psi.
  AccumT fl_device = AccumT(0); ///< Mirror of `fl`, passed by address to the backend.

  std::vector<AccumT> feedbackScratch; ///< Scratch buffer for the feedback GEMM's output.

  T* v = nullptr;        ///< Momentum buffer for the settling update, see SetMomentum().
  bool useMomentum = false;  ///< Whether momentum settling is enabled, see SetMomentum().
  AccumT momentumBeta = AccumT(0.9); ///< Momentum EMA decay rate, see SetMomentum().

  bool useCrossEntropy = false; ///< Whether softmax cross-entropy energy is enabled.
  /// @brief Per-row energy scratch for the cross-entropy path. Unused
  /// on CPU, kept for CUDA-path parity.
  std::vector<AccumT> rowEnergies;

  bool useEPC = false;  ///< Whether ePC settling is enabled, see SetUseEPC().
  T* adjoint = nullptr; ///< Chained backward-sweep signal, shape [batchSize, size].

  std::unique_ptr<MemoryArenaT<T>>
      localArena; ///< This layer's own arena, used when it isn't bound into a network-wide one.

  /// @name GetStateDict() host staging
  /// Host-side mirrors of W/b/Psi, for SyncStateDictFromDevice()/
  /// SyncStateDictToDevice() to copy into/out of. GetStateDict() always
  /// returns pointers into these buffers, never a live device pointer.
  ///@{
  mutable std::vector<T> hostW, hostB, hostPsi;
  ///@}

  /// @brief Resizes the host staging buffers above to match W/b/Psi's
  /// current shapes. Idempotent, so GetStateDict(),
  /// SyncStateDictFromDevice(), and SyncStateDictToDevice() can all
  /// call it unconditionally.
  void EnsureStateDictStaging() const noexcept;

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
  /// internally. Kept for call-site parity with FullPCLayer.
  /// @param backend Compute backend to run on; defaults to a new
  /// CPUBackendT<T, AccumT> if nullptr.
  FullPCLayerT(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
              AccumT learningRate, AccumT inferenceRate, AccumT feedback, AccumT lmbda,
              ActivationType aType, ActivationType dType,
              IComputeBackendT<T, AccumT>* backend = nullptr);

  ~FullPCLayerT() = default;

  template <typename ArenaT> void BindMemory(ArenaT& arena);
  size_t GetRequiredElements() const noexcept;
  void RandomizeWeights(std::mt19937& seedGenerator, const char* distribution = "normal") noexcept;

  std::map<std::string, TensorDescriptorT<T>> GetStateDict() const;
  /// @brief Copies W/b/Psi into the host staging buffers GetStateDict()
  /// exposes. Call before reading a GetStateDict() pointer.
  void SyncStateDictFromDevice() const;
  /// @brief Copies the host staging buffers (after writing into a
  /// GetStateDict() pointer) back onto the real W/b/Psi.
  void SyncStateDictToDevice();

  void SetLayerAbove(FullPCLayerT* l) noexcept
  {
    layerAbove = l;
  }
  void SetLayerBelow(FullPCLayerT* l) noexcept
  {
    layerBelow = l;
  }
  void SetTerminalLayer(FullPCLayerT* l) noexcept
  {
    terminalLayer = l;
  }

  /// @brief Sets the muPC forward-scaling factor and rescales DKP's
  /// feedback rate to match.
  void SetMuPCScale(AccumT a) noexcept
  {
    this->a = a;
    fl_device = fl * a;
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
  void DirectFeedbackUpdate() noexcept;

  void ReconstructBelief() noexcept;
  void ComputeAdjoint(const T* adjointAbove, AccumT adjointAboveScale = AccumT(1)) noexcept;
  void UpdateErrorEPC() noexcept;
  const T* GetAdjoint() const noexcept
  {
    return adjoint;
  }

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

  void SetMomentum(bool enabled, AccumT beta = AccumT(0.9)) noexcept
  {
    useMomentum = enabled;
    momentumBeta = beta;
  }
  bool GetMomentum() const noexcept
  {
    return useMomentum;
  }
  AccumT GetMomentumBeta() const noexcept
  {
    return momentumBeta;
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
  }
  void SetFeedbackRate(AccumT feedbackRate) noexcept;
  void SetLambda(AccumT lmbda) noexcept
  {
    this->lmbda = lmbda;
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
