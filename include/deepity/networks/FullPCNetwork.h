#pragma once
#include <deepity/backend/Backend.h>
#include <deepity/layers/FullPCLayer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file FullPCNetwork.h
 * @brief Orchestrates FullPCLayer's phases, matching
 * DirectKPPCNetwork's exact four-phase DKP-PC structure (see that
 * class's own docs for the full phase breakdown and citation), plus
 * network-level toggles for muPC scaling and residual connections --
 * both OFF by default, so a plain FullPCNetwork with every toggle left
 * alone reproduces DirectKPPCNetwork exactly (confirmed at the layer
 * level: FullPCLayer with a=1.0, useResidual=false is a bit-identical
 * match to DirectKPPCLayer).
 *
 * THIS PASS: muPC scaling and residual connections only. iPC, muP-style
 * per-layer LR scaling, momentum settling, and cross-entropy terminal
 * loss are not yet added -- each is meant to become its own, separate
 * toggle later, following the same pattern.
 */

namespace Deep
{
class FullPCNetwork
{
public:
  explicit FullPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept;
  ~FullPCNetwork() = default;

  FullPCNetwork(const FullPCNetwork&) = delete;
  FullPCNetwork& operator=(const FullPCNetwork&) = delete;

  void AddLayer(size_t size, size_t nextSize, size_t terminalSize, float lr, float ir, float fl,
                float lmbda, ActivationType aType, ActivationType dType);

  /// @brief Enables muPC-style per-layer forward scaling (Table 1
  /// of the muPC paper). OFF by default. Must be called before
  /// Compile() -- Compile() is what actually computes and applies
  /// each layer's `a`, once the full architecture (every AddLayer
  /// call) is known.
  void SetUseMuPCScaling(bool enabled) noexcept
  {
    useMuPCScaling = enabled;
  }

  /// @brief Enables residual/skip connections on middle hidden
  /// layers (muPC's "1-skip" ResNet formulation -- see that
  /// paper's A.2.4). OFF by default. Requires every middle hidden
  /// layer to have the SAME width as its neighbor; Compile()
  /// throws std::invalid_argument if that's violated, rather than
  /// silently skipping the mismatched connection (a silent skip
  /// risks masking a real configuration mistake).
  void SetUseResidualConnections(bool enabled) noexcept
  {
    useResidualConnections = enabled;
  }

  void RandomizeWeights(std::mt19937& rng);
  void ResetState() noexcept;
  void Clamp(const std::vector<float>& input);

  void ProjectForward() noexcept;
  float CalculateTerminalError() noexcept;
  void DirectFeedbackUpdate() noexcept;
  float Step(bool computeEnergy = true) noexcept;
  void UpdateWeights() noexcept;

  void SetOptimizer(OptimizerType o) noexcept
  {
    for (auto& layer : layers)
      layer->SetOptimizer(o);
  }
  void SetPsiOptimizer(OptimizerType o) noexcept
  {
    for (auto& layer : layers)
      layer->SetPsiOptimizer(o);
  }
  void SetLearningRate(float lr) noexcept
  {
    for (auto& layer : layers)
      layer->SetLearningRate(lr);
  }
  void SetFeedbackRate(float fl) noexcept
  {
    for (auto& layer : layers)
      layer->SetFeedbackRate(fl);
  }

  FullPCLayer* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  std::vector<std::unique_ptr<FullPCLayer>>& GetLayers() noexcept
  {
    return layers;
  }
  const std::vector<std::unique_ptr<FullPCLayer>>& GetLayers() const noexcept
  {
    return layers;
  }
  int GetBatchSize() const noexcept
  {
    return batchSize;
  }
  DeviceType GetDevice() const noexcept
  {
    return device;
  }

  float TrainStep(const std::vector<float>& x, const std::vector<float>& y, int inferenceSteps = 1);

  std::vector<float> Predict(const std::vector<float>& x, int inferenceSteps);

  /// @brief Loads all layers into one contiguous block of memory,
  /// wires layerAbove/layerBelow/terminalLayer across every
  /// layer -- same as DirectKPPCNetwork::Compile() -- PLUS: if
  /// useMuPCScaling, computes and applies each layer's `a` (Table
  /// 1) from the full, now-known architecture; if
  /// useResidualConnections, applies SetResidual(true) to every
  /// middle hidden layer after checking width match (throws
  /// std::invalid_argument on mismatch).
  void Compile();

  /// @brief Enables iPC: UpdateWeights() runs every settling
  /// step instead of once after settling completes. OFF by default --
  /// TrainStep() matches DirectKPPCNetwork's standard, two-phase
  /// behavior exactly when this is false.
  /// @cite Salvatori et al., "Incremental Predictive Coding", arXiv:2212.00720
  void SetUseIPC(bool enabled) noexcept
  {
    useIPC = enabled;
  }

private:
  std::vector<std::unique_ptr<FullPCLayer>> layers;

  std::unique_ptr<IComputeBackend> backend;
  DeviceType device;

  std::unique_ptr<MemoryArena> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif

  int batchSize;
  bool useIPC = false;

  bool useMuPCScaling = false;
  bool useResidualConnections = false;

  bool graphCaptured = false;
  int capturedInferenceSteps = -1;
};
} // namespace Deep
