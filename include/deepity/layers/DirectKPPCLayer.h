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
 * @file DirectKPPCLayer.h
 * @brief Direct Kolen-Pollack predictive coding layer, routed through
 * IComputeBackend. Bias updates go through AddBiasBroadcast()/SumRows();
 * biasGradScratch is a small (nextSize-length) accumulator SumRows alone
 * can't provide, used by the SGD branch's bias update.
 */

namespace Deep {
/// @brief Predictive Coding layer using Direct Kolen-Pollack feedback
/// alignment (DFA): a per-layer feedback matrix Psi, trained to
/// gradually correlate with W's transpose chain, replaces symmetric
/// weight transport for the feedback term. See DirectKPPCNetwork.h's
/// class docs and the paper's Appendix A.1.
class DirectKPPCLayer : public Layer {
protected:
  size_t size;         ///< This layer's own belief (z) size.
  size_t nextSize;      ///< Size of the layer above (0 marks a terminal layer).
  size_t terminalSize;  ///< Size of the network's final output layer.
  size_t batchSize;     ///< Batch size.

  float lr;     ///< Learning rate for W.
  float ir;     ///< Inference rate (Euler integration step size).
  float fl;     ///< Feedback rate for Psi.
  float lmbda;  ///< Weight decay (L2 regularization) coefficient, shared between W and Psi.

  bool isClamped = false;   ///< Whether ClampState() is currently active.
  bool muCacheValid = false; ///< Whether `cachedMu` holds a valid, up-to-date value.

  DirectKPPCLayer *layerAbove = nullptr;   ///< The next layer up, or nullptr if this is terminal.
  DirectKPPCLayer *layerBelow = nullptr;   ///< The next layer down, or nullptr if this is the input layer.
  DirectKPPCLayer *terminalLayer = nullptr; ///< The network's terminal layer, for DirectFeedbackUpdate().

  ActivationType activationType;  ///< This layer's activation type.
  ActivationFn activation;        ///< Resolved function pointer for activationType.
  DerivativeFn activationDerivative;      ///< Resolved function pointer for the derivative.
  DerivativeFn2 activationDerivativeInto; ///< Two-buffer variant of activationDerivative.

  OptimizerType opt = OptimizerType::SGD;    ///< Optimizer for W/b.
  OptimizerType optPsi = OptimizerType::SGD; ///< Optimizer for Psi.

  /// @brief Custom deleter type for `backend`, needed since IComputeBackend
  /// is an abstract base, see the member below.
  using BackendDeleter = void (*)(IComputeBackend *);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend; ///< This layer's compute backend.

  float *z = nullptr; ///< This layer's belief buffer, shape [batchSize, size].
  float *e = nullptr; ///< This layer's error buffer, shape [batchSize, size].

  float *W = nullptr;       ///< Weight matrix, shape [nextSize, size].
  float *b = nullptr;       ///< Bias vector, length nextSize.
  float *mu = nullptr;      ///< Forward prediction buffer, shape [batchSize, nextSize].
  float *cachedMu = nullptr; ///< Cached mu, valid when muCacheValid is true.
  float *Psi = nullptr;     ///< Direct feedback (DFA) weight matrix, shape [size, terminalSize].
  float *proj = nullptr;    ///< Scratch buffer: terminal error projected back through layerAbove's Psi, for DirectFeedbackUpdate()'s W update.

  float *grad_W = nullptr; ///< W's gradient scratch buffer.
  float *grad_b = nullptr; ///< b's gradient scratch buffer.
  float *m_W = nullptr;    ///< Adam/AdamW first-moment buffer for W.
  float *v_W = nullptr;    ///< Adam/AdamW second-moment buffer for W.
  float *m_b = nullptr;    ///< Adam/AdamW first-moment buffer for b.
  float *v_b = nullptr;    ///< Adam/AdamW second-moment buffer for b.

  float *grad_Psi = nullptr; ///< Psi's gradient scratch buffer.
  float *m_Psi = nullptr;    ///< Adam/AdamW first-moment buffer for Psi.
  float *v_Psi = nullptr;    ///< Adam/AdamW second-moment buffer for Psi.

  int *t_device = nullptr;      ///< Device-resident mirror of `t`, for graph-captured Adam steps.
  float *lr_device = nullptr;   ///< Device-resident mirror of `lr`.
  int *tPsi_device = nullptr;   ///< Device-resident mirror of `tPsi`.
  float *fl_device = nullptr;   ///< Device-resident mirror of `fl`.

  float *zF = nullptr;             ///< Activated belief, phi(z), used as the forward GEMM's input.
  float *feedbackScratch = nullptr; ///< Scratch buffer for the feedback GEMM's output.

  /// @brief nextSize-length scratch buffer for SumRows' output in
  /// UpdateWeights()'s SGD branch, SGD needs `b += lr_batch *
  /// sum(local_grad)`, an accumulate, which SumRows alone can't
  /// express (it only overwrites). Allocated unconditionally
  /// (regardless of which optimizer is selected) since it's cheap
  ///, at most `nextSize` floats, and simpler than branching
  /// allocation on optimizer choice for this one small buffer.
  float *biasGradScratch = nullptr;

  std::unique_ptr<MemoryArena> localArena; ///< This layer's own arena, used when it isn't bound into a network-wide one.

public:
  /// @param size input size
  /// @param nextSize output size (0 marks a terminal layer)
  /// @param terminalSize the size of the network's final output layer
  /// (e.g. 10 for MNIST), required directly, not inferred, since the
  /// true terminal layer isn't known until the whole network has been
  /// assembled.
  /// @param batchSize batch size
  /// @param learningRate learning rate for W
  /// @param inferenceRate inference rate (Euler integration step size)
  /// @param feedback feedback rate for Psi
  /// @param lmbda weight decay (L2 regularization) coefficient, shared
  /// between W and Psi
  /// @param aType activation type
  /// @param dType activation derivative type
  /// @param backend Compute backend to run on; defaults to a new
  /// CPUBackend if nullptr.
  DirectKPPCLayer(size_t size, size_t nextSize, size_t terminalSize,
                  size_t batchSize, float learningRate, float inferenceRate,
                  float feedback, float lmbda, ActivationType aType,
                  ActivationType dType, IComputeBackend *backend = nullptr);

  ~DirectKPPCLayer() override = default;

  /// @brief Binds this layer's buffers into a pre-allocated arena
  /// (MemoryArena for CPU, DeviceMemoryArena for GPU).
  /// @tparam ArenaT Either MemoryArena or DeviceMemoryArena.
  template <typename ArenaT> void BindMemory(ArenaT &arena);
  /// @brief Total floats this layer needs from its MemoryArena.
  size_t GetRequiredFloats() const noexcept;
  /// @brief Randomizes W and Psi.
  void RandomizeWeights(std::mt19937 &seedGenerator) noexcept;

  std::map<std::string, TensorDescriptor> GetStateDict() const override;

  /// @brief Sets the layer immediately above this one in the network.
  void SetLayerAbove(DirectKPPCLayer *l) noexcept { layerAbove = l; }
  /// @brief Sets the layer immediately below this one in the network.
  void SetLayerBelow(DirectKPPCLayer *l) noexcept { layerBelow = l; }
  /// @brief Sets the network's terminal layer, for DirectFeedbackUpdate().
  void SetTerminalLayer(DirectKPPCLayer *l) noexcept { terminalLayer = l; }

  /// @brief Matches Layer's virtual interface exactly (always
  /// computes real energy).
  float CalculateState() noexcept override { return CalculateState(true); }
  /// @brief NOT a virtual override, see SimplePCLayer's
  /// identical pattern. Lets the settling loop skip the
  /// cublasSdot-based energy reduction (and its capture-time
  /// sync) when the caller doesn't need the value.
  float CalculateState(bool needEnergy) noexcept;

  /// @brief Computes only mu (forward prediction), skipping error/energy.
  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept override;
  void UpdateWeights() noexcept override;
  /// @brief Perturbs W using the layer above's Psi and the terminal
  /// layer's error, the DFA phase, run once per batch before settling
  /// begins.
  void DirectFeedbackUpdate() noexcept;

  /// @brief Clamps this layer's beliefs to `inputData`, fixing them
  /// against UpdateState() until UnclampState() is called.
  void ClampState(const std::vector<float> &inputData) noexcept;
  /// @brief Releases a previous ClampState() call.
  void UnclampState() noexcept;
  /// @brief Resets this layer's beliefs (z) without touching learned weights.
  void ResetState() noexcept;

  /// @brief Whether this layer is currently clamped, needed so
  /// DirectKPPCNetwork::ProjectForward() can skip overwriting an
  /// already-clamped layer's z with a forward-projected guess.
  /// Same real bug SimplePCNetwork::ProjectForward() had; fixed
  /// here for the same reason, before it gets exercised for the
  /// first time by moving ProjectForward() inside graph capture.
  bool IsClamped() const noexcept { return isClamped; }

  /// @brief Sets the optimizer for W. Call before Compile().
  void SetOptimizer(OptimizerType o) noexcept { opt = o; }
  /// @brief Sets the optimizer for Psi. Call before Compile().
  void SetPsiOptimizer(OptimizerType o) noexcept { optPsi = o; }
  /// @brief Sets the learning rate, propagating it to the optimizer state.
  void SetLearningRate(float learningRate) noexcept;
  /// @brief Sets the inference rate.
  void SetInferenceRate(float inferenceRate) noexcept { ir = inferenceRate; }
  /// @brief Sets the feedback rate, propagating it to the optimizer state.
  void SetFeedbackRate(float feedbackRate) noexcept;
  /// @brief Sets the weight decay coefficient.
  void SetLambda(float lmbda) noexcept { this->lmbda = lmbda; }

  float *GetBeliefs() noexcept override { return z; }
  const float *GetErrors() const noexcept override { return e; }
  /// @brief Returns this layer's forward prediction buffer.
  const float *GetMu() const noexcept { return mu; }
  /// @brief Returns the weight matrix W, shape [nextSize, size].
  const float *GetWeights() const noexcept { return W; }
  /// @brief Returns the direct feedback (DFA) weight matrix Psi, shape [size, terminalSize].
  const float *GetDirectFeedbackWeights() const noexcept { return Psi; }
  /// @brief Returns the bias vector, length nextSize.
  const float *GetBiases() const noexcept { return b; }

  size_t GetBatchSize() const noexcept override { return batchSize; }
  size_t GetInputSize() const noexcept override { return size; }
  size_t GetOutputSize() const noexcept override { return nextSize; }
  /// @brief Returns the network's final output layer size, as given at construction.
  size_t GetTerminalSize() const noexcept { return terminalSize; }
};
} // namespace Deep
