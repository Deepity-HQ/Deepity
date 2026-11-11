#pragma once

#include <cstdlib>
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
 * @file SimplePCLayer.h
 * @brief Predictive coding layer with plain Gaussian energy (no
 * precision weighting): E = 0.5*sum(e^2), dz_dt = -e.
 *
 * Routed through an IComputeBackend, so it runs on either CPUBackend or
 * CUDABackend. `backend` defaults to nullptr, in which case the layer
 * constructs and owns its own CPUBackend.
 *
 * @note Optionally uses Adam instead of plain SGD, selected via
 * SetOptimizer().
 *
 * @note Optionally caches a clamped layer's outgoing prediction (mu)
 * across settling steps via SetMuCacheThreshold(). Default off.
 *
 * @note Activation convention: mu = W\@phi(z) + b -- the nonlinearity
 * applies to this layer's own belief before projecting up. mu itself
 * carries no final activation, so a terminal layer's mu is unbounded
 * logits; pair with SetCrossEntropy() when the terminal comparison
 * needs to stay bounded (e.g. classification against a one-hot target).
 * @version 1.1
 * @date 2026-09-05
 * @author Jack Rose
 */

namespace Deep
{
class SimplePCNDiagnostics;

/// @brief Predictive Coding layer without precision weighting: the
/// simpler, unweighted-energy counterpart to DiscriminativePCLayer.
class SimplePCLayer : public Layer
{
public:
  /// @brief Constructor for a SimplePCLayer.
  /// @param size Size of this layer's own belief (z)
  /// @param nextSize Size of the layer above's belief (this layer's
  ///        outgoing prediction target); 0 marks a terminal layer
  /// @param batchSize Batch size
  /// @param learningRate Learning rate for weight updates
  /// @param inferenceRate Inference rate (Euler integration step size)
  /// @param lmbda Weight decay (L2 regularization) coefficient
  /// @param act Activation function
  /// @param dAct Derivative of activation function
  /// @param backend Compute backend to route all math through.
  ///        Defaults to nullptr, in which case this layer
  ///        constructs and owns its own CPUBackend internally,
  ///        existing callers don't need to change anything. Pass a
  ///        real backend (owned elsewhere, e.g. by the network) to
  ///        run this layer on GPU.
  SimplePCLayer(size_t size, size_t nextSize, size_t batchSize = 1, float learningRate = 1e-6f,
                float inferenceRate = 0.1f, float lmbda = 1e-2f, void (*act)(float*, size_t) = relu,
                void (*dAct)(float*, size_t, bool) = dRelu, IComputeBackend* backend = nullptr);

  /// @brief Constructor for a SimplePCLayer, using a named
  /// ActivationType instead of raw function pointers.
  /// @param size Size of this layer's own belief (z)
  /// @param nextSize Size of the layer above's belief (this layer's
  ///        outgoing prediction target); 0 marks a terminal layer
  /// @param batchSize Batch size
  /// @param learningRate Learning rate for weight updates
  /// @param inferenceRate Inference rate (Euler integration step size)
  /// @param lmbda Weight decay (L2 regularization) coefficient
  /// @param aType Activation type
  /// @param dType Activation derivative type
  /// @param backend Compute backend to route all math through.
  ///        See the other constructor's doc for the default-nullptr
  ///        behavior.
  SimplePCLayer(size_t size, size_t nextSize, size_t batchSize = 1, float learningRate = 1e-6f,
                float inferenceRate = 0.1f, float lmbda = 1e-2f,
                ActivationType aType = ActivationType::RELU,
                ActivationType dType = ActivationType::dRELU, IComputeBackend* backend = nullptr);

  /// @brief Calculates the total network energy state, and this
  /// layer's outgoing prediction (mu).
  ///
  /// \f[
  /// E = \sum_l 1/2 ||z^{(l)} - \mu^{(l)}||^2
  /// \f]
  /// When SetCrossEntropy() is enabled, this layer's own term in E uses
  /// softmax cross-entropy against layerBelow's mu instead.
  /// @param needEnergy Asks for energy
  /// @return This layer's energy contribution at the current state, if asked for.
  float CalculateState(bool needEnergy = true) noexcept;

  float CalculateState() noexcept override
  {
    return CalculateState(true);
  }

  /// @brief Computes the state derivatives for inference.
  ///
  /// \f[
  /// \frac{dz^{(l)}}{dt} = -e^{(l)} + \sigma'(z^{(l)}) \odot (W^{(l-1)})^T e^{(l-1)}
  /// \f]
  void UpdateState() noexcept override;

  /// @brief Computes weight updates via gradient descent, with L2 weight decay.
  void UpdateWeights() noexcept override;
  /// @brief Subtracts each row's mean from W (see IComputeBackend::CenterRows()).
  void CenterWeights() noexcept override
  {
    if (W != nullptr)
      backend->CenterRows(W, nextSize, size);
  }

  /// @brief No-op; exists for Layer interface conformance.
  void Flush() noexcept override {}

  /// @brief Clamps this layer's beliefs to externally-provided data,
  /// preventing them from being updated by UpdateState().
  /// @param inputData Flattened input data of length `size`.
  void ClampState(const std::vector<float>& inputData) noexcept;
  /// @brief Releases a previous ClampState() call, allowing this
  /// layer's beliefs to update normally again.
  void UnclampState() noexcept;

  /// @brief Whether ClampState() is currently active on this layer.
  bool IsClamped() const noexcept
  {
    return isClamped;
  }

  /// @brief Forces the next ComputeMuOnly() call to recompute mu
  /// instead of reusing a cached value, without otherwise touching
  /// clamp state. Call after a CUDA-graph (re)capture.
  void InvalidateMuCache() noexcept
  {
    muCacheValid = false;
  }

  /// @brief Returns this layer's belief buffer.
  /// @return Pointer to this layer's `size`-length beliefs.
  float* GetBeliefs() noexcept override
  {
    return z;
  }

  /// @brief Returns this layer's outgoing prediction buffer (mu).
  /// nullptr for a terminal layer (nextSize=0).
  /// @return Outgoing prediction buffer
  const float* GetMu() const noexcept
  {
    return mu;
  }

  /// @brief Returns this layer's prediction-error buffer.
  /// @return Pointer to this layer's `size`-length errors.
  const float* GetErrors() const noexcept override
  {
    return e;
  }
  /// @brief Returns this layer's own belief size.
  size_t GetInputSize() const noexcept override
  {
    return size;
  }
  /// @brief Returns this layer's outgoing prediction size (0 for a
  /// terminal layer).
  size_t GetOutputSize() const noexcept override
  {
    return nextSize;
  }
  /// @brief Returns the batch size this layer was constructed with.
  size_t GetBatchSize() const noexcept override
  {
    return batchSize;
  }

  /// @brief Returns this layer's weight buffer.
  const float* GetWeights() const noexcept
  {
    return W;
  }
  /// @brief Returns this layer's weight buffer.
  float* GetWeights() noexcept
  {
    return W;
  }
  /// @brief Returns this layer's bias buffer.
  const float* GetBiases() const noexcept
  {
    return b;
  }
  /// @brief Returns this layer's bias buffer.
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

  /// @brief Returns the learning rate used for weight updates.
  float GetLearningRate() const noexcept
  {
    return lr;
  }
  /// @brief Returns the inference rate (Euler integration step size).
  float GetInferenceRate() const noexcept
  {
    return ir;
  }
  /// @brief Returns the weight-decay (L2 regularization) coefficient.
  float GetLambda() const noexcept
  {
    return lmbda;
  }

  /// @brief Sets the learning rate used for weight updates.
  /// @param lr The new learning rate.
  void SetLearningRate(float lr) noexcept;
  /// @brief Sets the inference rate (Euler integration step size).
  /// @param ir The new inference rate.
  void SetInferenceRate(float ir) noexcept
  {
    this->ir = ir;
  }
  /// @brief Sets the weight-decay (L2 regularization) coefficient.
  /// @param l The new lambda value.
  void SetLambda(float l) noexcept
  {
    this->lmbda = l;
  }

  /// @brief Selects the optimizer used for weight updates (SGD or Adam).
  /// @param o The optimizer type to use.
  void SetOptimizer(const OptimizerType o) noexcept
  {
    opt = o;
  }

  /// @brief Sets the mu-cache staleness threshold: mu is recomputed
  /// only when ||z - z_at_last_recompute|| / (||z_at_last_recompute||
  /// + eps) exceeds this value. -1 disables caching (default). 0 is
  /// exact for clamped layers only (z never changes while clamped).
  /// >0 extends caching to unclamped layers too, as an approximation.
  void SetMuCacheThreshold(float threshold) noexcept
  {
    muCacheThreshold = threshold;
  }
  /// @brief Returns the current mu-cache staleness threshold; see
  /// SetMuCacheThreshold() above.
  float GetMuCacheThreshold() const noexcept
  {
    return muCacheThreshold;
  }

  /// @brief Computes only mu (forward prediction), skipping error/energy.
  /// @param zFAlreadyComputed Pass true if phi(z) was already computed
  /// by the caller this step; false recomputes it here.
  void ComputeMuOnly(bool zFAlreadyComputed = false) noexcept;

  /// @brief Enables softmax cross-entropy energy for this layer's error
  /// against layerBelow's mu, instead of the default Gaussian energy.
  /// Only meaningful on the terminal layer.
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

  /// @brief Sets the layer immediately above this one in the network.
  /// @param above Pointer to the layer above; may be nullptr for a
  /// terminal layer.
  void SetLayerAbove(SimplePCLayer* above) noexcept
  {
    layerAbove = above;
  }
  /// @brief Sets the layer immediately below this one in the network.
  /// @param below Pointer to the layer below; may be nullptr for the
  /// input layer.
  void SetLayerBelow(SimplePCLayer* below) noexcept
  {
    layerBelow = below;
  }

  /// @brief Resets this layer's beliefs/errors back to their initial
  /// values, without touching learned weights.
  void ResetState() noexcept;

  /// @brief Returns the layer immediately above this one.
  /// @warning Dereferences layerAbove without a null check; only
  /// valid if SetLayerAbove() was previously called with a non-null
  /// pointer.
  const SimplePCLayer& GetLayerAbove() const noexcept
  {
    return *layerAbove;
  }
  /// @brief Returns the layer immediately below this one.
  /// @warning Dereferences layerBelow without a null check; only
  /// valid if SetLayerBelow() was previously called with a non-null
  /// pointer.
  const SimplePCLayer& GetLayerBelow() const noexcept
  {
    return *layerBelow;
  }

  /// @brief Randomizes this layer's weights (and biases) in place.
  /// @param twister The classic Mersenne Twister
  /// @param distribution The distribution of the randomization: (normal, uniform)
  void RandomizeWeights(std::mt19937& twister, const char* distribution = "normal") noexcept;

  /// @brief Returns this layer's configured activation type.
  ActivationType GetActivationType() const noexcept
  {
    return To_AType(activation);
  }
  /// @brief Returns this layer's configured activation-derivative type.
  ActivationType GetDerivativeType() const noexcept
  {
    return To_AType(activationDerivative);
  }

  /// @brief Computes the total number of floats this layer requires
  /// from a MemoryArena (weights, biases, beliefs, errors, scratch
  /// buffers, and Adam's moment buffers).
  /// @return The required float count.
  size_t GetRequiredFloats() const noexcept;
  /// @brief Binds this layer's weight/state/scratch buffers into the
  /// supplied arena. Must be called before any other operation.
  /// Templated so either MemoryArena (CPU) or DeviceMemoryArena
  /// (GPU) can be bound, resolved entirely at compile time.
  /// @param arena The arena to bind into.
  template <typename ArenaT> void BindMemory(ArenaT& arena);

private:
  std::unique_ptr<MemoryArena> localArena;

  /// @brief The compute backend this layer routes all math through.
  using BackendDeleter = void (*)(IComputeBackend*);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend;

  float* W;
  float* b;
  float* e;
  float* z;

  int batchSize;

  float* mu;
  float* cachedMu;        // Cached mu, valid when muCacheValid is true.
  float* zF;              // phi(z): this layer's activated belief.
  float* feedbackScratch; // own_state_size scratch for the feedback
                          // GEMM's output, before f'(z) is applied.
  float* feedbackDerivScratch; // own_state_size scratch holding f'(z).

  bool useCrossEntropy = false; // See SetCrossEntropy().
  float* rowEnergies = nullptr; // Per-row energy scratch for the
                                // cross-entropy path, length batchSize.

  float lr;
  float ir;
  float lmbda;
  bool isClamped = false;

  int* t_device = nullptr;
  float* lr_device = nullptr;

  float muCacheThreshold = -1.0f; // See SetMuCacheThreshold().
  bool muCacheValid = false;      // Whether cachedMu is up to date.

  SimplePCLayer* layerAbove;
  SimplePCLayer* layerBelow;
  ActivationFn activation;
  DerivativeFn activationDerivative;
  DerivativeFn2 activationDerivativeInto; // Fused two-buffer derivative (dst, src, n).
  ActivationType activationType;
  OptimizerType opt = OptimizerType::SGD;
  // Optional Adam state.
  float* grad_W = nullptr;
  float* grad_b = nullptr;
  float* m_W = nullptr;
  float* v_W = nullptr;
  float* m_b = nullptr;
  float* v_b = nullptr;

  friend class SimplePCNDiagnostics;
};

} // namespace Deep