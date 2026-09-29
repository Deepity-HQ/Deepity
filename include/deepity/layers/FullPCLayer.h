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
 * @file FullPCLayer.h
 * @brief PC layer combining DKP direct feedback (DirectKPPCLayer's
 * structure) with muPC scaling (`a`) and optional residual connections
 * (`useResidual`), each off by default so plain defaults reproduce
 * DirectKPPCLayer exactly.
 *
 * `a` and `useResidual` are set by the owning network at Compile() time
 * (via SetMuPCScale/SetResidual), not computed here, this layer, at
 * construction time, doesn't know the full network's shape (width N,
 * depth L, input dim) that muPC's own aL formula needs.
 */

namespace Deep
{
/// @brief PC layer combining muPC scaling, optional residual
/// connections, and DKP direct feedback, every extra off by default,
/// so plain defaults reproduce DirectKPPCLayer exactly.
class FullPCLayer : public Layer
{
protected:
  size_t size;         ///< This layer's own belief (z) size.
  size_t nextSize;      ///< Size of the layer above (0 marks a terminal layer).
  size_t terminalSize;  ///< Size of the network's final output layer.
  size_t batchSize;     ///< Batch size.

  float lr;     ///< Learning rate for W.
  float ir;     ///< Inference rate (Euler integration step size).
  float fl;     ///< Feedback rate for Psi.
  float lmbda;  ///< Weight decay (L2 regularization) coefficient, shared between W and Psi.

  // muPC scaling: mu = a * (W @ phi(z)) + b [+ z if useResidual].
  // a=1.0, useResidual=false reproduces plain PC exactly.
  float a = 1.0f;           ///< muPC forward-scaling factor, see SetMuPCScale().
  bool useResidual = false; ///< Whether the residual/skip connection is enabled, see SetResidual().

  // muPC's OTHER table-1 change, alongside `a`: standard PC initializes
  // W with fan-in/fan-out-scaled variance (RandomizeWeights()'s default,
  // b_l = N_{l-1}^-1 in the paper's notation); muPC instead uses UNIT
  // variance for W, letting `a` alone carry the width/depth scaling.
  // false (the default) keeps RandomizeWeights()'s existing fan-scaled
  // behavior, matching plain PC exactly.
  bool useMuPCInit = false; ///< Whether unit-variance W init is enabled, see SetMuPCInit().

  bool isClamped = false;   ///< Whether ClampState() is currently active.
  bool muCacheValid = false; ///< Whether `cachedMu` holds a valid, up-to-date value.

  FullPCLayer* layerAbove = nullptr;   ///< The next layer up, or nullptr if this is terminal.
  FullPCLayer* layerBelow = nullptr;   ///< The next layer down, or nullptr if this is the input layer.
  FullPCLayer* terminalLayer = nullptr; ///< The network's terminal layer, for DirectFeedbackUpdate().

  ActivationType activationType; ///< This layer's activation type.

  OptimizerType opt = OptimizerType::SGD;    ///< Optimizer for W/b.
  OptimizerType optPsi = OptimizerType::SGD; ///< Optimizer for Psi.

  /// @brief Custom deleter type for `backend`, needed since IComputeBackend
  /// is an abstract base, see the member below.
  using BackendDeleter = void (*)(IComputeBackend*);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend; ///< This layer's compute backend.

  float* z = nullptr; ///< This layer's belief buffer, shape [batchSize, size].
  float* e = nullptr; ///< This layer's error buffer, shape [batchSize, size].

  float* W = nullptr;       ///< Weight matrix, shape [nextSize, size].
  float* b = nullptr;       ///< Bias vector, length nextSize.
  float* mu = nullptr;      ///< Forward prediction buffer, shape [batchSize, nextSize].
  float* cachedMu = nullptr; ///< Cached mu, valid when muCacheValid is true.
  float* Psi = nullptr;     ///< Direct feedback (DFA) weight matrix, shape [size, terminalSize].
  float* proj = nullptr;    ///< Scratch buffer: terminal error projected back through layerAbove's Psi, for DirectFeedbackUpdate()'s W update.

  float* grad_W = nullptr; ///< W's gradient scratch buffer.
  float* grad_b = nullptr; ///< b's gradient scratch buffer.
  float* m_W = nullptr;    ///< Adam/AdamW first-moment buffer for W.
  float* v_W = nullptr;    ///< Adam/AdamW second-moment buffer for W.
  float* m_b = nullptr;    ///< Adam/AdamW first-moment buffer for b.
  float* v_b = nullptr;    ///< Adam/AdamW second-moment buffer for b.

  float* grad_Psi = nullptr; ///< Psi's gradient scratch buffer.
  float* m_Psi = nullptr;    ///< Adam/AdamW first-moment buffer for Psi.
  float* v_Psi = nullptr;    ///< Adam/AdamW second-moment buffer for Psi.

  int* t_device = nullptr;      ///< Device-resident Adam/AdamW step count for W/b, for graph-captured steps.
  float* lr_device = nullptr;   ///< Device-resident mirror of `lr`.
  int* tPsi_device = nullptr;   ///< Device-resident Adam/AdamW step count for Psi.
  float* fl_device = nullptr;   ///< Device-resident mirror of `fl`.

  float* zF = nullptr;             ///< Activated belief, phi(z), used as the forward GEMM's input.
  float* feedbackScratch = nullptr; ///< Scratch buffer for the feedback GEMM's output.

  float* v = nullptr;          ///< Momentum buffer for the settling update, see SetMomentum().
  bool useMomentum = false;    ///< Whether momentum settling is enabled, see SetMomentum().
  float momentumBeta = 0.9f;   ///< Momentum EMA decay rate, see SetMomentum().

  bool useCrossEntropy = false; ///< Whether softmax cross-entropy energy is enabled, see SetCrossEntropy().
  float* rowEnergies = nullptr; ///< Per-row energy scratch for the cross-entropy path, length batchSize.

  // ePC (Goemaere/Innocenti et al., see SetUseEPC()): `e` itself becomes
  // the settled variable instead of `z`, via a full top-to-bottom
  // backward sweep each settling step rather than a one-hop message.
  // `adjoint` holds this layer's chained backprop sensitivity so the
  // layer below can read it as ITS incoming signal, see ComputeAdjoint().
  bool useEPC = false;       ///< Whether ePC settling is enabled, see SetUseEPC().
  float* adjoint = nullptr; ///< Chained backward-sweep signal, shape [batchSize, size].

  /// @brief nextSize-length scratch for SumRows' output in
  /// UpdateWeights()'s SGD branch (accumulate, which SumRows
  /// alone can't express). Same reasoning as DirectKPPCLayer's
  /// identical buffer.
  float* biasGradScratch = nullptr;

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
  FullPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
              float learningRate, float inferenceRate, float feedback, float lmbda,
              ActivationType aType, ActivationType dType, IComputeBackend* backend = nullptr);

  ~FullPCLayer() override = default;

  /// @brief Binds this layer's buffers into a pre-allocated arena
  /// (MemoryArena for CPU, DeviceMemoryArena for GPU).
  /// @tparam ArenaT Either MemoryArena or DeviceMemoryArena.
  template <typename ArenaT> void BindMemory(ArenaT& arena);
  /// @brief Total floats this layer needs from its MemoryArena.
  size_t GetRequiredFloats() const noexcept;
  /// @brief Randomizes W and Psi.
  void RandomizeWeights(std::mt19937& seedGenerator) noexcept;

  std::map<std::string, TensorDescriptor> GetStateDict() const override;

  /// @brief Sets the layer immediately above this one in the network.
  void SetLayerAbove(FullPCLayer* l) noexcept
  {
    layerAbove = l;
  }
  /// @brief Sets the layer immediately below this one in the network.
  void SetLayerBelow(FullPCLayer* l) noexcept
  {
    layerBelow = l;
  }
  /// @brief Sets the network's terminal layer, for DirectFeedbackUpdate().
  void SetTerminalLayer(FullPCLayer* l) noexcept
  {
    terminalLayer = l;
  }

  /// @brief Set by the owning network at Compile() time, once the
  /// full layer_sizes list (and therefore d_in/N/L) is known.
  /// 1.0 (the default) reproduces plain, unscaled PC.
  void SetMuPCScale(float a) noexcept
  {
    this->a = a;
  }
  /// @brief Returns the current muPC forward-scaling factor, see
  /// SetMuPCScale().
  float GetMuPCScale() const noexcept
  {
    return a;
  }

  /// @brief Set by the owning network at Compile() time, alongside
  /// SetMuPCScale() (both are Table 1 of the muPC paper). Only affects
  /// the NEXT RandomizeWeights() call; false (the default) reproduces
  /// plain PC's fan-scaled init exactly.
  void SetMuPCInit(bool enabled) noexcept
  {
    useMuPCInit = enabled;
  }
  /// @brief Returns whether unit-variance W init is enabled, see
  /// SetMuPCInit().
  bool GetMuPCInit() const noexcept
  {
    return useMuPCInit;
  }

  /// @brief Set by the owning network at Compile() time. Requires
  /// size == nextSize (checked at the point it's actually used,
  /// not here, since nextSize may not be finalized yet when this
  /// is called). false (the default) reproduces plain PC.
  void SetResidual(bool enabled) noexcept
  {
    useResidual = enabled;
  }
  /// @brief Returns whether the residual connection is enabled, see
  /// SetResidual().
  bool GetResidual() const noexcept
  {
    return useResidual;
  }

  /// @brief Enables softmax cross-entropy energy for THIS layer's error
  /// against layerBelow's mu (i.e. this only makes sense set on the
  /// terminal layer, against the second-to-last layer's logits, NOT
  /// looped over every layer the way SetMuPCScale/SetResidual/
  /// SetMomentum's network-level setters are). OFF by default (plain
  /// Gaussian energy, matching DirectKPPCLayer exactly).
  void SetCrossEntropy(bool enabled) noexcept
  {
    useCrossEntropy = enabled;
  }
  /// @brief Returns whether softmax cross-entropy energy is enabled,
  /// see SetCrossEntropy().
  bool GetCrossEntropy() const noexcept
  {
    return useCrossEntropy;
  }

  /// @brief Matches Layer's virtual interface exactly (always computes
  /// real energy).
  float CalculateState() noexcept override
  {
    return CalculateState(true);
  }
  /// @brief NOT a virtual override, see SimplePCLayer's identical
  /// pattern. Lets the settling loop skip the energy reduction (and its
  /// capture-time sync) when the caller doesn't need the value.
  float CalculateState(bool needEnergy) noexcept;

  /// @brief Computes only mu (forward prediction), skipping error/energy.
  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept override;
  void UpdateWeights() noexcept override;
  /// @brief Perturbs W using the layer above's Psi and the terminal
  /// layer's error, the DFA phase, run once per batch before settling
  /// begins.
  void DirectFeedbackUpdate() noexcept;

  /// @brief ePC forward-sweep step: reconstructs this layer's belief as
  /// z := layerBelow->GetMu() + e (the paper's s_i := ŝ_i + ε_i), rather
  /// than integrating z locally. Requires layerBelow != nullptr (the
  /// input layer's z is the clamped input, never reconstructed this
  /// way). See SetUseEPC().
  void ReconstructBelief() noexcept;
  /// @brief ePC backward-sweep step: computes this layer's chained
  /// backprop sensitivity (`adjoint`) from the sensitivity passed down
  /// from the layer above, via the same feedback GEMM UpdateState()
  /// uses, but materialized into `adjoint` instead of applied inline,
  /// so the layer below can read it as ITS `adjointAbove`. See
  /// SetUseEPC().
  /// @param adjointAbove The layer above's `adjoint` (or, for the
  /// topmost hidden layer, the terminal's own error buffer).
  /// @param adjointAboveScale Extra scale folded into the GEMM alongside
  /// `a` (default 1.0). The topmost hidden layer needs -1.0: the true
  /// backprop seed is dLoss/dmu_terminal, but Deepity's own error
  /// convention stores e_terminal = -dLoss/dmu_terminal (see
  /// IComputeBackend::ComputeSoftmaxCrossEntropyErrorAndEnergy's docs,
  /// same sign relationship holds for the Gaussian terminal too), so the
  /// terminal's raw error needs negating before it's a valid seed. Every
  /// other layer already receives a correctly-signed `adjoint` from the
  /// layer above and uses the default.
  void ComputeAdjoint(const float* adjointAbove, float adjointAboveScale = 1.0f) noexcept;
  /// @brief ePC's error-update step: e := (1-ir)*e - ir*adjoint, the
  /// paper's `ε -= λ*(ε+adjoint)` gradient-descent rule. Call
  /// ComputeAdjoint() first. See SetUseEPC().
  void UpdateErrorEPC() noexcept;
  /// @brief Returns this layer's chained backward-sweep signal, see
  /// ComputeAdjoint().
  const float* GetAdjoint() const noexcept
  {
    return adjoint;
  }

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

  /// @brief Forces the next ComputeMuOnly() call to recompute mu from
  /// scratch, even if this layer is clamped. Needed under iPC: weights
  /// change every settling step, so a clamped layer's cached mu (valid
  /// under the standard, two-phase assumption that W is fixed throughout
  /// settling) goes stale the moment UpdateWeights() runs mid-loop.
  void InvalidateMuCache() noexcept
  {
    muCacheValid = false;
  }

  /// @brief Enables momentum (inertial) settling: the update direction
  /// is EMA-smoothed (decay `beta`) before being applied to z, instead
  /// of applied directly each step. OFF by default. `v` is allocated
  /// unconditionally in BindMemory() regardless of this flag's value at
  /// that time (same reasoning as biasGradScratch, cheap, and avoids
  /// an ordering hazard if this is called after Compile()). Reset to
  /// zero at the start of every TrainStep()/Predict() call, same as z.
  void SetMomentum(bool enabled, float beta = 0.9f) noexcept
  {
    useMomentum = enabled;
    momentumBeta = beta;
  }
  /// @brief Returns whether momentum settling is enabled, see SetMomentum().
  bool GetMomentum() const noexcept
  {
    return useMomentum;
  }
  /// @brief Returns the momentum EMA decay rate, see SetMomentum().
  float GetMomentumBeta() const noexcept
  {
    return momentumBeta;
  }

  /// @brief Enables ePC settling: `e` (not `z`) becomes the primary
  /// variable, updated each settling step from a full top-to-bottom
  /// backward sweep (ReconstructBelief()/ComputeAdjoint()/
  /// UpdateErrorEPC(), orchestrated by FullPCNetwork) instead of a
  /// one-hop local message. Only meaningful on hidden layers (input has
  /// no free error; the terminal's error IS the loss, computed by
  /// CalculateState() unchanged). OFF by default.
  /// @see Goemaere et al., "ePC: Fast and Deep Predictive Coding in
  /// Digital Simulation", https://arxiv.org/abs/2505.20137
  void SetUseEPC(bool enabled) noexcept
  {
    useEPC = enabled;
  }
  /// @brief Returns whether ePC settling is enabled, see SetUseEPC().
  bool GetUseEPC() const noexcept
  {
    return useEPC;
  }

  /// @brief Sets the optimizer for W. Call before Compile().
  void SetOptimizer(OptimizerType o) noexcept
  {
    opt = o;
  }
  /// @brief Sets the optimizer for Psi. Call before Compile().
  void SetPsiOptimizer(OptimizerType o) noexcept
  {
    optPsi = o;
  }
  /// @brief Sets the learning rate, propagating it to the optimizer state.
  void SetLearningRate(float learningRate) noexcept;
  /// @brief Sets the inference rate.
  void SetInferenceRate(float inferenceRate) noexcept
  {
    ir = inferenceRate;
  }
  /// @brief Sets the feedback rate, propagating it to the optimizer state.
  void SetFeedbackRate(float feedbackRate) noexcept;
  /// @brief Sets the weight decay coefficient.
  void SetLambda(float lmbda) noexcept
  {
    this->lmbda = lmbda;
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
  /// @brief Returns the direct feedback (DFA) weight matrix Psi, shape [size, terminalSize].
  const float* GetDirectFeedbackWeights() const noexcept
  {
    return Psi;
  }
  /// @brief Returns the bias vector, length nextSize.
  const float* GetBiases() const noexcept
  {
    return b;
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
};
} // namespace Deep
