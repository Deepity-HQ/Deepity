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
 * moment state, always at least as wide as T -- see
 * IComputeBackendT<T, AccumT>'s own doc comment). Produced by
 * mechanically retyping FullPCLayer's own `float` members/parameters to
 * `T`/`AccumT`, NOT a reimplementation -- every method here must stay
 * line-for-line equivalent to FullPCLayer's own method bodies, so that
 * FullPCLayerT<float, float> reproduces FullPCLayer's output bit-exactly
 * (the quantization plan's verification gate #1).
 *
 * Deliberately does NOT inherit from Layer: Layer's virtual interface
 * (GetBeliefs() etc.) is fixed at `float*`, which only FullPCLayerT<float,
 * ...> could ever satisfy. Nothing in this pass needs a FullPCLayerT<T>
 * to be used polymorphically as a Layer*, so this is a free-standing
 * class template instead (the same position FullPCNetwork already has).
 * Every OTHER layer (SimplePCLayer, DirectKPPCLayer, GaussSeidelPCLayer,
 * the conv family) is unaffected and keeps deriving from Layer exactly
 * as before.
 */

namespace Deep
{
/// @brief Templated counterpart to FullPCLayer, see this file's own
/// comment for the full rationale.
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
  T* derivScratch = nullptr; ///< nextSize-shaped scratch holding f'(preact).
  T* feedbackError = nullptr; ///< nextSize-shaped scratch, see FullPCLayer's own doc comment.
  T* Psi = nullptr;  ///< Direct feedback (DFA) weight matrix, shape [size, terminalSize].
  T* proj = nullptr; ///< Scratch buffer for DirectFeedbackUpdate()'s W update.

  // grad_*/m_*/v_* and lr_device/fl_device/t_device/tPsi_device are all
  // AccumT-typed (or int, for the step counters) optimizer-accumulator
  // state, owned directly rather than arena-allocated like every T-typed
  // buffer above -- see BindAdamStateT()/BindLearningRateStateT()'s own
  // doc comments in LayerBindHelpers.h for why (in short: a layer's
  // MemoryArenaT<T>::Allocate() returns T*, which isn't AccumT* unless
  // T==AccumT).
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

  T* feedbackScratch = nullptr; ///< Scratch buffer for the feedback GEMM's output.

  T* v = nullptr;        ///< Momentum buffer for the settling update, see SetMomentum().
  bool useMomentum = false;  ///< Whether momentum settling is enabled, see SetMomentum().
  AccumT momentumBeta = AccumT(0.9); ///< Momentum EMA decay rate, see SetMomentum().

  bool useCrossEntropy = false; ///< Whether softmax cross-entropy energy is enabled.
  /// @brief Per-row energy scratch for the cross-entropy path. Owned
  /// directly (not arena-allocated like every other buffer here):
  /// it's AccumT-typed, and CPUBackendT has no AccumT-typed Zero()/
  /// allocation primitive of its own to hand it a slot in the T-typed
  /// arena when T != AccumT (see IComputeBackendT's own doc comment on
  /// why AccumT buffers are the optimizer's domain, not the backend's).
  /// Harmless: it's unused on CPU (CPUBackendT's own
  /// ComputeSoftmaxCrossEntropyErrorAndEnergy() ignores it, matching
  /// the non-templated CPUBackend), kept only for CUDA-path parity.
  std::vector<AccumT> rowEnergies;

  bool useEPC = false;  ///< Whether ePC settling is enabled, see SetUseEPC().
  T* adjoint = nullptr; ///< Chained backward-sweep signal, shape [batchSize, size].

  T* biasGradScratch = nullptr; ///< nextSize-length scratch for SumRows' output.

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
  /// @param dType activation derivative type (unused, kept for call-site
  /// parity with FullPCLayer's own constructor; derivative type is
  /// always derived from aType via ToDerivativeType() at each use site)
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

  /// @brief Always computes real energy; not a Layer:: override (see
  /// this file's own class comment on why FullPCLayerT no longer
  /// derives from Layer).
  AccumT CalculateState() noexcept
  {
    return CalculateState(true);
  }
  AccumT CalculateState(bool needEnergy) noexcept;

  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept;
  void UpdateWeights() noexcept;
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
