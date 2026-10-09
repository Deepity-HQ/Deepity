#pragma once
#include <deepity/backend/Backend.h>
#include <deepity/layers/FullConvPCLayer.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file FullConvPCNetwork.h
 * @brief Convolutional analog of FullPCNetwork -- see FullConvPCLayer.h
 * for the activation-placement convention this class's math follows
 * (conv activates `mu` directly, unlike the dense layer's raw-mu/
 * activated-zF split).
 */

namespace Deep
{
/// @brief Convolutional Predictive Coding network combining every PC
/// variant FullPCNetwork has, adapted for conv layers.
class FullConvPCNetwork
{
public:
  explicit FullConvPCNetwork(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept;
  ~FullConvPCNetwork() = default;

  FullConvPCNetwork(const FullConvPCNetwork&) = delete;
  FullConvPCNetwork& operator=(const FullConvPCNetwork&) = delete;

  /// @param outChannels 0 marks a terminal layer.
  /// @param terminalSize the network's final output size (flat channel
  /// count), required on every AddLayer() call, validated against the
  /// actual terminal layer's inChannels at Compile() time.
  void AddLayer(int inChannels, int outChannels, int inHeight, int inWidth, int kernelH,
                int kernelW, int strideH, int strideW, int padH, int padW, int terminalSize,
                float lr, float ir, float fl, float lmbda, ActivationType aType,
                ActivationType dType, int poolH = 1, int poolW = 1, int poolStrideH = 1,
                int poolStrideW = 1);

  void SetUseMuPCScaling(bool enabled) noexcept { useMuPCScaling = enabled; }
  void SetUseResidualConnections(bool enabled) noexcept { useResidualConnections = enabled; }

  void RandomizeWeights(std::mt19937& rng);
  void ResetState() noexcept;
  void Clamp(const std::vector<float>& input);
  void ProjectForward() noexcept;
  float CalculateTerminalError() noexcept;
  void DirectFeedbackUpdate() noexcept;
  float Step(bool computeEnergy = true) noexcept;
  /// @brief ePC's settling step: full forward sweep (bottom to top) then
  /// full backward sweep (top to bottom), see FullPCNetwork::EPCStep()
  /// for the identical dense-side structure this mirrors.
  void EPCStep() noexcept;
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
  void SetInferenceRate(float ir) noexcept
  {
    for (auto& layer : layers)
      layer->SetInferenceRate(ir);
  }
  void SetAdamEpsilon(float eps) noexcept
  {
    for (auto& layer : layers)
      layer->SetAdamEpsilon(eps);
  }

  FullConvPCLayer* GetTerminalLayer() noexcept { return layers.back().get(); }
  std::vector<std::unique_ptr<FullConvPCLayer>>& GetLayers() noexcept { return layers; }
  const std::vector<std::unique_ptr<FullConvPCLayer>>& GetLayers() const noexcept
  {
    return layers;
  }
  int GetBatchSize() const noexcept { return batchSize; }
  DeviceType GetDevice() const noexcept { return device; }

  /// @brief Runs one training step. @p computeEnergy=false skips the
  /// per-layer energy readout (a cublasSdot + cudaStreamSynchronize EACH,
  /// on GPU -- a real cost even though the computation itself is cheap)
  /// and returns the last value that WAS computed instead, for callers
  /// that only log/check energy every few steps.
  float TrainStep(const std::vector<float>& x, const std::vector<float>& y,
                  int inferenceSteps = 1, bool computeEnergy = true);
  std::vector<float> Predict(const std::vector<float>& x, int inferenceSteps);

  /// @brief Each layer's FullConvPCLayer::GetDiagnosticStats(), flat and
  /// row-major [layer][stat] (4 stats/layer: error mean, error RMS,
  /// weight norm, dead-mu fraction). Call right after TrainStep() to see
  /// what the state that just drove a real weight update looked like.
  std::vector<float> GetAllLayerDiagnostics() const noexcept;

  void Compile();

  /// @see Salvatori et al., "Incremental Predictive Coding", https://arxiv.org/abs/2212.00720
  void SetUseIPC(bool enabled) noexcept { useIPC = enabled; }
  void SetUseMomentum(bool enabled, float beta = 0.9f) noexcept
  {
    for (auto& layer : layers)
      layer->SetMomentum(enabled, beta);
  }
  void SetUseCrossEntropy(bool enabled) noexcept { GetTerminalLayer()->SetCrossEntropy(enabled); }
  /// @see Goemaere et al., "ePC: Fast and Deep Predictive Coding in
  /// Digital Simulation", https://arxiv.org/abs/2505.20137
  void SetUseEPC(bool enabled) noexcept
  {
    useEPC = enabled;
    for (size_t i = 1; i + 1 < layers.size(); ++i)
      layers[i]->SetUseEPC(enabled);
  }

private:
  std::vector<std::unique_ptr<FullConvPCLayer>> layers;

  std::unique_ptr<IComputeBackend> backend;
  DeviceType device;

  std::unique_ptr<MemoryArena> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArena> gpuArena;
#endif

  int batchSize;
  bool useIPC = false;
  bool useEPC = false;

  bool useMuPCScaling = false;
  bool useResidualConnections = false;

  bool graphCaptured = false;
  int capturedInferenceSteps = -1;

  /// @brief Last energy TrainStep() actually computed; returned as-is
  /// when a call skips the readout via computeEnergy=false.
  float lastEnergy = 0.0f;
};
} // namespace Deep
