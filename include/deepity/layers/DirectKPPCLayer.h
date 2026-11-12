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

namespace Deep
{
/// @brief Predictive Coding layer using Direct Kolen-Pollack feedback
/// alignment (DFA): a per-layer feedback matrix Psi, trained to
/// gradually correlate with W's transpose chain, replaces symmetric
/// weight transport for the feedback term. See DirectKPPCNetwork.h's
/// class docs and the paper's Appendix A.1.
class DirectKPPCLayer : public Layer
{
protected:
  size_t size;         ///< This layer's own belief (z) size.
  size_t nextSize;     ///< Size of the layer above (0 marks a terminal layer).
  size_t terminalSize; ///< Size of the network's final output layer.
  size_t batchSize;    ///< Batch size.
  /// @brief Psi's actual row width; defaults to terminalSize (tight
  /// packing), overridden only via BindExternalMemory() for a batched
  /// run's layers. See GetPsiStride()'s own comment.
  size_t psiStride;

  float lr;    ///< Learning rate for W.
  float ir;    ///< Inference rate (Euler integration step size).
  float fl;    ///< Feedback rate for Psi.
  float lmbda; ///< Weight decay (L2 regularization) coefficient, shared between W and Psi.

  bool isClamped = false;    ///< Whether ClampState() is currently active.
  bool muCacheValid = false; ///< Whether `cachedMu` holds a valid, up-to-date value.

  DirectKPPCLayer* layerAbove = nullptr; ///< The next layer up, or nullptr if this is terminal.
  DirectKPPCLayer* layerBelow =
      nullptr; ///< The next layer down, or nullptr if this is the input layer.
  DirectKPPCLayer* terminalLayer =
      nullptr; ///< The network's terminal layer, for DirectFeedbackUpdate().

  ActivationType activationType;          ///< This layer's activation type.
  ActivationFn activation;                ///< Resolved function pointer for activationType.
  DerivativeFn activationDerivative;      ///< Resolved function pointer for the derivative.
  DerivativeFn2 activationDerivativeInto; ///< Two-buffer variant of activationDerivative.

  OptimizerType opt = OptimizerType::SGD;    ///< Optimizer for W/b.
  OptimizerType optPsi = OptimizerType::SGD; ///< Optimizer for Psi.

  /// @brief Custom deleter type for `backend`, needed since IComputeBackend
  /// is an abstract base, see the member below.
  using BackendDeleter = void (*)(IComputeBackend*);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend; ///< This layer's compute backend.

  float* z = nullptr; ///< This layer's belief buffer, shape [batchSize, size].
  float* e = nullptr; ///< This layer's error buffer, shape [batchSize, size].

  float* W = nullptr;        ///< Weight matrix, shape [nextSize, size].
  float* b = nullptr;        ///< Bias vector, length nextSize.
  float* mu = nullptr;       ///< Forward prediction buffer, shape [batchSize, nextSize].
  float* cachedMu = nullptr; ///< Cached mu, valid when muCacheValid is true.
  /// @brief Pre-activation value of mu (W@z+b, before Activation() is
  /// applied in place), shape [batchSize, nextSize]. f'(preact), not
  /// f'(z) or a shortcut from mu itself, is what UpdateState() and
  /// UpdateWeights() apply to the feedback signal.
  float* preact = nullptr;
  /// @brief Cached pre-activation, valid alongside cachedMu when this
  /// layer is clamped.
  float* cachedPreact = nullptr;
  /// @brief nextSize-shaped scratch holding f'(preact).
  float* derivScratch = nullptr;
  /// @brief nextSize-shaped scratch holding e_above (or, in
  /// DirectFeedbackUpdate(), proj) pre-multiplied by f'(preact).
  float* feedbackError = nullptr;
  float* Psi = nullptr;  ///< Direct feedback (DFA) weight matrix, shape [size, terminalSize].
  float* proj = nullptr; ///< Scratch buffer: terminal error projected back through layerAbove's
                         ///< Psi, for DirectFeedbackUpdate()'s W update.

  float* grad_W = nullptr; ///< W's gradient scratch buffer.
  float* grad_b = nullptr; ///< b's gradient scratch buffer.
  float* m_W = nullptr;    ///< Adam/AdamW first-moment buffer for W.
  float* v_W = nullptr;    ///< Adam/AdamW second-moment buffer for W.
  float* m_b = nullptr;    ///< Adam/AdamW first-moment buffer for b.
  float* v_b = nullptr;    ///< Adam/AdamW second-moment buffer for b.

  float* grad_Psi = nullptr; ///< Psi's gradient scratch buffer.
  float* m_Psi = nullptr;    ///< Adam/AdamW first-moment buffer for Psi.
  float* v_Psi = nullptr;    ///< Adam/AdamW second-moment buffer for Psi.

  int* t_device = nullptr;    ///< Device-resident mirror of `t`, for graph-captured Adam steps.
  float* lr_device = nullptr; ///< Device-resident mirror of `lr`.
  int* tPsi_device = nullptr; ///< Device-resident mirror of `tPsi`.
  float* fl_device = nullptr; ///< Device-resident mirror of `fl`.

  float* feedbackScratch = nullptr; ///< Scratch buffer for the feedback GEMM's output.

  /// @brief nextSize-length scratch for SumRows' output in
  /// UpdateWeights()'s SGD branch.
  float* biasGradScratch = nullptr;

  std::unique_ptr<MemoryArena>
      localArena; ///< This layer's own arena, used when it isn't bound into a network-wide one.

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
  DirectKPPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
                  float learningRate, float inferenceRate, float feedback, float lmbda,
                  ActivationType aType, ActivationType dType, IComputeBackend* backend = nullptr);

  ~DirectKPPCLayer() override = default;

  /// @brief Binds this layer's buffers into a pre-allocated arena
  /// (MemoryArena for CPU, DeviceMemoryArena for GPU).
  /// @tparam ArenaT Either MemoryArena or DeviceMemoryArena.
  template <typename ArenaT> void BindMemory(ArenaT& arena);
  /// @brief Total floats this layer needs from its MemoryArena.
  size_t GetRequiredFloats() const noexcept;

  /// @brief Every pointer BindMemory() would normally carve out of this
  /// layer's own arena slice, supplied instead from the caller's own
  /// storage. Used by DirectKPPCNetwork's batched-GPU fast path: a run
  /// of uniform-width layers gets ITS buffers aliased into one flat,
  /// per-quantity buffer spanning the whole run (so the fast path's
  /// batched GEMMs/elementwise calls can address all of them in one
  /// dispatch), while every method below (CalculateState, UpdateState,
  /// UpdateWeights, DirectFeedbackUpdate, ...) keeps running completely
  /// unchanged -- they only ever read `this->z`/`this->W`/etc., never
  /// caring where the memory actually lives. Every field must be
  /// supplied (same invariant GetRequiredFloats() documents: Adam state
  /// is always allocated regardless of the optimizer actually selected,
  /// so switching to Adam/AdamW after Compile() stays safe even for an
  /// externally-bound layer).
  /// @warning Does not Zero() anything -- the caller zeroed the whole
  /// flat buffer once, up front; redoing it per layer here would be
  /// `batchedRunCount` separate calls instead of one.
  struct ExternalMemory
  {
    float* z;
    float* e;
    float* W;
    float* b;
    float* mu;
    float* cachedMu;
    float* preact;
    float* cachedPreact;
    float* derivScratch;
    float* feedbackError;
    float* proj;
    float* Psi;
    size_t psiStride; ///< See DirectKPPCLayer::GetPsiStride()'s own comment.
    float* feedbackScratch;
    float* biasGradScratch;
    float* grad_W;
    float* m_W;
    float* v_W;
    float* grad_b;
    float* m_b;
    float* v_b;
    float* grad_Psi;
    float* m_Psi;
    float* v_Psi;
    int* t_device;
    float* lr_device;
    int* tPsi_device;
    float* fl_device;
  };
  /// @brief Binds this layer's buffers directly to externally-supplied
  /// pointers instead of carving them out of an arena. See
  /// ExternalMemory's own doc comment above.
  void BindExternalMemory(const ExternalMemory& mem) noexcept;
  /// @brief Randomizes W and Psi.
  /// @param distribution Distribution for W only, e.g. "normal(0,1)" or
  /// "uniform(-0.3,0.3)"; Psi always uses its own fixed fan-scaled init.
  /// The default "normal" (with no "(a,b)") falls back to this class's
  /// own default init for W.
  void RandomizeWeights(std::mt19937& seedGenerator, const char* distribution = "normal") noexcept;

  std::map<std::string, TensorDescriptor> GetStateDict() const override;

  /// @brief Sets the layer immediately above this one in the network.
  void SetLayerAbove(DirectKPPCLayer* l) noexcept
  {
    layerAbove = l;
  }
  /// @brief Sets the layer immediately below this one in the network.
  void SetLayerBelow(DirectKPPCLayer* l) noexcept
  {
    layerBelow = l;
  }
  /// @brief Sets the network's terminal layer, for DirectFeedbackUpdate().
  void SetTerminalLayer(DirectKPPCLayer* l) noexcept
  {
    terminalLayer = l;
  }

  /// @brief Matches Layer's virtual interface exactly (always
  /// computes real energy).
  float CalculateState() noexcept override
  {
    return CalculateState(true);
  }
  /// @brief Lets the settling loop skip the energy reduction when the
  /// caller doesn't need the value.
  float CalculateState(bool needEnergy) noexcept;

  /// @brief Computes only mu (forward prediction), skipping error/energy.
  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept override;
  /// @brief Subtracts each output unit's mean incoming weight from W
  /// (two backend GEMMs, so CPU and GPU). Removes the common-mode drift that pushes
  /// ReLU units below zero, where they stop learning.
  void CenterWeights() noexcept override;
  void UpdateWeights() noexcept override;
  /// @brief Perturbs W using the layer above's Psi and the terminal
  /// layer's error, the DFA phase, run once per batch before settling
  /// begins.
  void DirectFeedbackUpdate() noexcept;

  /// @brief Clamps this layer's beliefs to `inputData`, fixing them
  /// against UpdateState() until UnclampState() is called.
  void ClampState(const std::vector<float>& inputData) noexcept;
  /// @brief Releases a previous ClampState() call.
  void UnclampState() noexcept;
  /// @brief Resets this layer's beliefs (z) without touching learned weights.
  void ResetState() noexcept;

  /// @brief Whether ClampState() is currently active on this layer.
  bool IsClamped() const noexcept
  {
    return isClamped;
  }

  /// @brief Forces the next ComputeMuOnly() call to recompute mu instead
  /// of reusing a cached value, without otherwise touching clamp state.
  /// Call after a CUDA-graph (re)capture.
  void InvalidateMuCache() noexcept
  {
    muCacheValid = false;
  }

  /// @brief Sets the optimizer for W. Call before Compile().
  void SetOptimizer(OptimizerType o) noexcept
  {
    opt = o;
  }
  /// @brief Returns the optimizer currently selected for W.
  OptimizerType GetOptimizer() const noexcept
  {
    return opt;
  }
  /// @brief Sets the optimizer for Psi. Call before Compile().
  void SetPsiOptimizer(OptimizerType o) noexcept
  {
    optPsi = o;
  }
  /// @brief Returns the optimizer currently selected for Psi.
  OptimizerType GetPsiOptimizer() const noexcept
  {
    return optPsi;
  }
  /// @brief Sets the learning rate, propagating it to the optimizer state.
  void SetLearningRate(float learningRate) noexcept;
  /// @brief Returns the current learning rate for W.
  float GetLearningRate() const noexcept
  {
    return lr;
  }
  /// @brief Sets the inference rate.
  void SetInferenceRate(float inferenceRate) noexcept
  {
    ir = inferenceRate;
  }
  /// @brief Sets the feedback rate, propagating it to the optimizer state.
  void SetFeedbackRate(float feedbackRate) noexcept;
  /// @brief Returns the current feedback rate for Psi.
  float GetFeedbackRate() const noexcept
  {
    return fl;
  }
  /// @brief Sets the weight decay coefficient.
  void SetLambda(float lmbda) noexcept
  {
    this->lmbda = lmbda;
  }
  /// @brief Returns the current weight decay coefficient.
  float GetLambda() const noexcept
  {
    return lmbda;
  }
  /// @brief Returns the current inference rate.
  float GetInferenceRate() const noexcept
  {
    return ir;
  }

  float* GetBeliefs() noexcept override
  {
    return z;
  }
  const float* GetErrors() const noexcept override
  {
    return e;
  }
  /// @brief Returns this layer's forward prediction buffer.
  const float* GetMu() const noexcept
  {
    return mu;
  }
  /// @brief Returns the weight matrix W, shape [nextSize, size].
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
  /// @brief Returns the direct feedback (DFA) weight matrix Psi, shape [size, terminalSize].
  const float* GetDirectFeedbackWeights() const noexcept
  {
    return Psi;
  }
  /// @brief Mutable overload of GetDirectFeedbackWeights() above, same
  /// reason as GetWeights()'s/GetBiases()'s own mutable overloads.
  float* GetDirectFeedbackWeights() noexcept
  {
    return Psi;
  }
  /// @brief Returns the bias vector, length nextSize.
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

  size_t GetBatchSize() const noexcept override
  {
    return batchSize;
  }
  size_t GetInputSize() const noexcept override
  {
    return size;
  }
  size_t GetOutputSize() const noexcept override
  {
    return nextSize;
  }
  /// @brief Returns the network's final output layer size, as given at construction.
  size_t GetTerminalSize() const noexcept
  {
    return terminalSize;
  }
  /// @brief Returns Psi's actual row width -- equal to terminalSize
  /// unless DirectKPPCNetwork::Compile() widened it (padded with
  /// permanently-zero columns) so a batched run's Psi GEMM clears
  /// CUDABackend::MatMulBatched()'s kMinSafeDim safely. See
  /// BindExternalMemory()'s own comment.
  size_t GetPsiStride() const noexcept
  {
    return psiStride;
  }
  /// @brief Returns this layer's activation type.
  ActivationType GetActivationType() const noexcept
  {
    return activationType;
  }
};
} // namespace Deep
