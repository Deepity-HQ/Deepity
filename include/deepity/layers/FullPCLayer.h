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
 * @brief New, standalone PC layer (not derived from DirectKPPCLayer)
 * meant to eventually carry every technique from the paper-combination
 * work -- muPC scaling, optional residual connections, DKP direct
 * feedback, AdamW, muP-style per-layer LR scaling, optional iPC,
 * cross-entropy terminal loss -- each one OFF by default, opt-in per
 * layer/network, so a plain DKP-PC-equivalent network is just this
 * class with every extra left at its default.
 *
 * THIS PASS: only muPC scaling (`a`) and residual connections
 * (`useResidual`) are added, on top of DirectKPPCLayer's proven
 * structure (DKP feedback, Adam/AdamW, graph-capture-compatible
 * device-pointer t/lr). Everything else stays for a later pass.
 *
 * `a` and `useResidual` are set by the owning network at Compile()
 * time (via SetMuPCScale/SetResidual), not computed here -- this
 * layer, at construction time, doesn't know the full network's shape
 * (width N, depth L, input dim) that muPC's own aL formula needs.
 * Defaults (a=1.0, useResidual=false) reproduce plain, unscaled PC
 * exactly, matching the "every extra opt-in, off by default" design.
 */

namespace Deep
{
class FullPCLayer : public Layer
{
protected:
  size_t size;
  size_t nextSize;
  size_t terminalSize;
  size_t batchSize;

  float lr;
  float ir;
  float fl;
  float lmbda;

  // muPC scaling: mu = a * (W @ phi(z)) + b [+ z if useResidual].
  // a=1.0, useResidual=false reproduces plain PC exactly.
  float a = 1.0f;
  bool useResidual = false;

  bool isClamped = false;
  bool muCacheValid = false;

  FullPCLayer* layerAbove = nullptr;
  FullPCLayer* layerBelow = nullptr;
  FullPCLayer* terminalLayer = nullptr;

  ActivationType activationType;

  OptimizerType opt = OptimizerType::SGD;
  OptimizerType optPsi = OptimizerType::SGD;

  using BackendDeleter = void (*)(IComputeBackend*);
  std::unique_ptr<IComputeBackend, BackendDeleter> backend;

  float* z = nullptr;
  float* e = nullptr;

  float* W = nullptr;
  float* b = nullptr;
  float* mu = nullptr;
  float* cachedMu = nullptr;
  float* Psi = nullptr;
  float* proj = nullptr;

  float* grad_W = nullptr;
  float* grad_b = nullptr;
  float* m_W = nullptr;
  float* v_W = nullptr;
  float* m_b = nullptr;
  float* v_b = nullptr;

  float* grad_Psi = nullptr;
  float* m_Psi = nullptr;
  float* v_Psi = nullptr;

  int* t_device = nullptr;
  float* lr_device = nullptr;
  int* tPsi_device = nullptr;
  float* fl_device = nullptr;

  float* zF = nullptr;
  float* zFDeriv = nullptr;
  float* feedbackScratch = nullptr;

  float* v = nullptr;
  bool useMomentum = false;
  float momentumBeta = 0.9f;

  bool useCrossEntropy = false;
  float* rowEnergies = nullptr;

  /// @brief nextSize-length scratch for SumRows' output in
  /// UpdateWeights()'s SGD branch (accumulate, which SumRows
  /// alone can't express). Same reasoning as DirectKPPCLayer's
  /// identical buffer.
  float* biasGradScratch = nullptr;

  std::unique_ptr<MemoryArena> localArena;

public:
  FullPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
              float learningRate, float inferenceRate, float feedback, float lmbda,
              ActivationType aType, ActivationType dType, IComputeBackend* backend = nullptr);

  ~FullPCLayer() override = default;

  template <typename ArenaT> void BindMemory(ArenaT& arena);
  size_t GetRequiredFloats() const noexcept;
  void RandomizeWeights(std::mt19937& seedGenerator) noexcept;

  std::map<std::string, TensorDescriptor> GetStateDict() const override;

  void SetLayerAbove(FullPCLayer* l) noexcept
  {
    layerAbove = l;
  }
  void SetLayerBelow(FullPCLayer* l) noexcept
  {
    layerBelow = l;
  }
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
  float GetMuPCScale() const noexcept
  {
    return a;
  }

  /// @brief Set by the owning network at Compile() time. Requires
  /// size == nextSize (checked at the point it's actually used,
  /// not here, since nextSize may not be finalized yet when this
  /// is called). false (the default) reproduces plain PC.
  void SetResidual(bool enabled) noexcept
  {
    useResidual = enabled;
  }
  bool GetResidual() const noexcept
  {
    return useResidual;
  }

  /// @brief Enables softmax cross-entropy energy for THIS layer's error
  /// against layerBelow's mu (i.e. this only makes sense set on the
  /// terminal layer, against the second-to-last layer's logits -- NOT
  /// looped over every layer the way SetMuPCScale/SetResidual/
  /// SetMomentum's network-level setters are). OFF by default (plain
  /// Gaussian energy, matching DirectKPPCLayer exactly).
  void SetCrossEntropy(bool enabled) noexcept
  {
    useCrossEntropy = enabled;
  }
  bool GetCrossEntropy() const noexcept
  {
    return useCrossEntropy;
  }

  float CalculateState() noexcept override
  {
    return CalculateState(true);
  }
  float CalculateState(bool needEnergy) noexcept;

  void ComputeMuOnly() noexcept;
  void UpdateState() noexcept override;
  void UpdateWeights() noexcept override;
  void DirectFeedbackUpdate() noexcept;

  void ClampState(const std::vector<float>& inputData) noexcept;
  void UnclampState() noexcept;
  void ResetState() noexcept;

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
  /// that time (same reasoning as biasGradScratch -- cheap, and avoids
  /// an ordering hazard if this is called after Compile()). Reset to
  /// zero at the start of every TrainStep()/Predict() call, same as z.
  void SetMomentum(bool enabled, float beta = 0.9f) noexcept
  {
    useMomentum = enabled;
    momentumBeta = beta;
  }
  bool GetMomentum() const noexcept
  {
    return useMomentum;
  }
  float GetMomentumBeta() const noexcept
  {
    return momentumBeta;
  }

  void SetOptimizer(OptimizerType o) noexcept
  {
    opt = o;
  }
  void SetPsiOptimizer(OptimizerType o) noexcept
  {
    optPsi = o;
  }
  void SetLearningRate(float learningRate) noexcept;
  void SetInferenceRate(float inferenceRate) noexcept
  {
    ir = inferenceRate;
  }
  void SetFeedbackRate(float feedbackRate) noexcept;
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
  const float* GetMu() const noexcept
  {
    return mu;
  }
  const float* GetWeights() const noexcept
  {
    return W;
  }
  const float* GetDirectFeedbackWeights() const noexcept
  {
    return Psi;
  }
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
  size_t GetTerminalSize() const noexcept
  {
    return terminalSize;
  }
};
} // namespace Deep
