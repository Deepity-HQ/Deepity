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
 * @file GaussSeidelPCLayerT.h
 * @brief Templated counterpart to GaussSeidelPCLayer, over a storage
 * type `T` and an accumulator type `AccumT`. See GaussSeidelPCLayer.h's
 * own class comment for the required three-sweep (UpdateState/
 * ComputePrediction/ComputeError) calling order, unchanged here. Does
 * not inherit from Layer.
 */

namespace Deep
{
/// @brief Templated counterpart to GaussSeidelPCLayer.
template <typename T, typename AccumT = T> class GaussSeidelPCLayerT
{
protected:
  size_t size;      ///< This layer's own belief (z) size.
  size_t nextSize;  ///< Size of the layer above (0 marks a terminal layer).
  size_t batchSize; ///< Batch size.

  AccumT lr;    ///< Learning rate for weight updates.
  AccumT ir;    ///< Inference rate (Euler integration step size).
  AccumT lmbda; ///< Weight decay (L2 regularization) coefficient.

  bool isClamped = false; ///< Whether ClampState() is currently active.

  GaussSeidelPCLayerT* layerAbove = nullptr; ///< The next layer up, or nullptr if terminal.
  GaussSeidelPCLayerT* layerBelow = nullptr; ///< The next layer down, or nullptr if the input layer.

  ActivationType activationType; ///< This layer's activation type.
  ActivationType derivativeType; ///< This layer's activation-derivative type.
  OptimizerType opt = OptimizerType::SGD; ///< Optimizer for W/b.

  using BackendDeleter = void (*)(IComputeBackendT<T, AccumT>*);
  std::unique_ptr<IComputeBackendT<T, AccumT>, BackendDeleter> backend; ///< This layer's compute backend.

  T* W = nullptr; ///< Weights.
  T* b = nullptr; ///< Biases.
  T* e = nullptr; ///< Errors.
  T* z = nullptr; ///< Internal state (belief).

  /// @brief This layer's own outgoing prediction. Stays clean and
  /// activated at all times, never mutated into a derivative.
  T* mu = nullptr;
  /// @brief mu's pre-activation value, held over across timesteps the
  /// same way mu itself is.
  T* preact = nullptr;
  /// @brief Feedback-alignment matrix. Separate from W, same shape,
  /// randomly initialized once, never updated again.
  T* E = nullptr;

  std::vector<AccumT> derivScratch;  ///< out_state_size scratch: f'(preact), see UpdateState().
  std::vector<AccumT> feedbackError; ///< out_state_size scratch: e_above * f'(preact).
  std::vector<AccumT> dz_dt;         ///< own_state_size scratch: feedbackError projected through E.

  std::vector<AccumT> grad_W; ///< W's gradient scratch buffer; also SGD's row-sum scratch for grad_b.
  std::vector<AccumT> grad_b; ///< b's gradient scratch buffer; also SGD's row-sum scratch.
  std::vector<AccumT> m_W;    ///< Adam/AdamW first-moment buffer for W.
  std::vector<AccumT> v_W;    ///< Adam/AdamW second-moment buffer for W.
  std::vector<AccumT> m_b;    ///< Adam/AdamW first-moment buffer for b.
  std::vector<AccumT> v_b;    ///< Adam/AdamW second-moment buffer for b.

  int t_device = 0;             ///< Adam/AdamW step count for W/b.
  AccumT lr_device = AccumT(0); ///< Mirror of `lr`, passed by address to the backend.

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
  /// @param dType Activation-derivative type
  /// @param backend Compute backend to run on; defaults to a new
  /// CPUBackendT<T, AccumT> if nullptr.
  GaussSeidelPCLayerT(size_t size, size_t nextSize, size_t batchSize, AccumT learningRate,
                      AccumT inferenceRate, AccumT lmbda, ActivationType aType,
                      ActivationType dType, IComputeBackendT<T, AccumT>* backend = nullptr);

  ~GaussSeidelPCLayerT() = default;

  /// @brief Step 1 of a Gauss-Seidel timestep: updates z using
  /// mu/e_above held over from the end of the previous timestep.
  /// Does NOT mutate mu.
  void UpdateState() noexcept;

  /// @brief Step 2: recomputes mu fresh from the z just updated by
  /// UpdateState() in this same timestep.
  void ComputePrediction() noexcept;

  /// @brief Step 3: recomputes e fresh, using this layer's own
  /// (just-updated) z as the target and layerBelow's fresh mu as the
  /// prediction.
  /// @return This layer's energy contribution, or AccumT(0) if
  /// needEnergy is false.
  AccumT ComputeError(bool needEnergy = true) noexcept;

  /// @brief Computes weight updates. Called once after the full
  /// settling loop completes.
  void UpdateWeights() noexcept;
  /// @brief Subtracts each row's mean from W (see IComputeBackend::CenterRows()).
  void CenterWeights() noexcept
  {
    if (W != nullptr)
      backend->CenterRows(W, nextSize, size);
  }

  /// @brief Clamps this layer's beliefs to externally-provided data.
  void ClampState(const std::vector<T>& inputData) noexcept;
  /// @brief Releases a previous ClampState() call.
  void UnclampState() noexcept;
  bool IsClamped() const noexcept
  {
    return isClamped;
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
  const T* GetFeedbackWeights() const noexcept
  {
    return E;
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

  void SetLayerAbove(GaussSeidelPCLayerT* above) noexcept
  {
    layerAbove = above;
  }
  void SetLayerBelow(GaussSeidelPCLayerT* below) noexcept
  {
    layerBelow = below;
  }

  /// @brief Resets beliefs/errors/predictions back to their initial
  /// values, without touching learned weights.
  void ResetState() noexcept;

  const GaussSeidelPCLayerT& GetLayerAbove() const noexcept
  {
    return *layerAbove;
  }
  const GaussSeidelPCLayerT& GetLayerBelow() const noexcept
  {
    return *layerBelow;
  }

  /// @brief Randomizes this layer's weights. @p distribution applies
  /// to W only; E always uses its own fixed uniform(-0.3,0.3) init.
  void RandomizeWeights(std::mt19937& twister, const char* distribution = "normal") noexcept;

  ActivationType GetActivationType() const noexcept
  {
    return activationType;
  }
  ActivationType GetDerivativeType() const noexcept
  {
    return derivativeType;
  }

  size_t GetRequiredElements() const noexcept;
  template <typename ArenaT> void BindMemory(ArenaT& arena);

  std::map<std::string, TensorDescriptorT<T>> GetStateDict() const;
};
} // namespace Deep
