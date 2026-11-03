#pragma once
#include <deepity/backend/BackendT.h>
#include <deepity/layers/ConvPCLayerT.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file ConvPCNetworkT.h
 * @brief Templated counterpart to ConvPCNetwork, over the same
 * `T`/`AccumT` pair as ConvPCLayerT -- see FullPCNetworkT's own doc
 * comment for the full rationale. Produced by mechanically retyping
 * ConvPCNetwork's own `float` members/parameters to `T`/`AccumT`.
 *
 * CUDA graph capture is dropped entirely, same reasoning as every
 * other templated network: device==DEVICE_GPU already throws inside
 * CreateBackendT() at construction (no CUDABackendT yet).
 */

namespace Deep
{
/// @brief Templated counterpart to ConvPCNetwork, see this file's own
/// comment for the full rationale.
template <typename T, typename AccumT = T> class ConvPCNetworkT
{
public:
  using ElementType = T;

  explicit ConvPCNetworkT(int batchSize, DeviceType device = DeviceType::DEVICE_CPU)
      : device(device)
      , batchSize(batchSize)
  {
    backend = CreateBackendT<T, AccumT>(device);
  }
  ~ConvPCNetworkT() = default;

  ConvPCNetworkT(const ConvPCNetworkT&) = delete;
  ConvPCNetworkT& operator=(const ConvPCNetworkT&) = delete;

  /// @brief Adds a convolutional layer. Pass outChannels=0 to mark a
  /// terminal layer (no outgoing prediction).
  void AddLayer(int inChannels, int outChannels, int inHeight, int inWidth, int kernelH,
               int kernelW, int strideH = 1, int strideW = 1, int padH = 0, int padW = 0,
               AccumT lr = AccumT(1e-6), AccumT ir = AccumT(0.1), AccumT pr = AccumT(0.01),
               AccumT lmbda = AccumT(1e-2), ActivationType aType = ActivationType::RELU,
               ActivationType dType = ActivationType::dRELU)
  {
    auto l = std::make_unique<ConvPCLayerT<T, AccumT>>(inChannels, outChannels, inHeight, inWidth,
                                                       kernelH, kernelW, strideH, strideW, padH,
                                                       padW, batchSize, lr, ir, pr, lmbda, aType,
                                                       dType, backend.get());

    if (!layers.empty())
    {
      layers.back()->SetLayerAbove(l.get());
      l->SetLayerBelow(layers.back().get());
    }
    layers.push_back(std::move(l));
  }

  void SetOptimizer(OptimizerType opt) noexcept
  {
    pendingOpt = opt;
  }

  void Compile()
  {
    for (auto& l : layers)
      l->SetOptimizer(pendingOpt);

    size_t total = 0;
    for (auto& l : layers)
      total += l->GetRequiredElements();

    if (device == DeviceType::DEVICE_CPU)
    {
      cpuArena = std::make_unique<MemoryArenaT<T>>(total);
      for (auto& l : layers)
        l->BindMemory(*cpuArena);
    }
#if defined(DEEPITY_USE_CUDA)
    else
    {
      backend->PrepareForBatchSize(batchSize);
      gpuArena = std::make_unique<DeviceMemoryArenaT<T, AccumT>>(backend.get(), total);
      for (auto& l : layers)
        l->BindMemory(*gpuArena);
      for (auto& l : layers)
        l->PrepareConvWorkspace();
    }
#endif
  }

  void RandomizeWeights(std::mt19937& rng, const char* distribution = "normal") noexcept
  {
    for (auto& l : layers)
      l->RandomizeWeights(rng, distribution);
  }
  void ResetState() noexcept
  {
    for (auto& l : layers)
      l->ResetState();
  }

  void Clamp(const std::vector<T>& input) noexcept
  {
    layers.front()->ClampState(input);
  }

  AccumT CalculateState(bool needEnergy = true) noexcept
  {
    AccumT e(0);
    for (auto& l : layers)
      e += l->CalculateState(needEnergy);
    return needEnergy ? e : AccumT(0);
  }
  void UpdateState() noexcept
  {
    for (auto& l : layers)
      l->UpdateState();
  }
  void UpdateWeights() noexcept
  {
    for (size_t i = 0; i + 1 < layers.size(); ++i)
      layers[i]->UpdateWeights();
  }
  /// @brief Applies precision updates to every non-terminal layer.
  /// Not called automatically by TrainStep(), same as
  /// ConvPCNetwork::UpdatePrecision()'s own convention -- the caller
  /// invokes this explicitly when desired.
  void UpdatePrecision() noexcept
  {
    for (size_t i = 0; i + 1 < layers.size(); ++i)
      layers[i]->UpdatePrecision();
  }

  ConvPCLayerT<T, AccumT>* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  const std::vector<std::unique_ptr<ConvPCLayerT<T, AccumT>>>& GetLayers() const noexcept
  {
    return layers;
  }
  int GetBatchSize() const noexcept
  {
    return batchSize;
  }

  AccumT TrainStep(const std::vector<T>& x, const std::vector<T>& y, int inferenceSteps)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    for (int t = 0; t < inferenceSteps; ++t)
    {
      CalculateState(false);
      UpdateState();
    }
    UpdateWeights();

    AccumT finalEnergy = CalculateState();
    GetTerminalLayer()->UnclampState();

    return finalEnergy;
  }

  std::vector<T> Predict(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    for (int t = 0; t < inferenceSteps; ++t)
    {
      CalculateState(false);
      UpdateState();
    }

    ConvPCLayerT<T, AccumT>* terminal = GetTerminalLayer();
    const T* beliefs = terminal->GetBeliefs();
    size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

    std::vector<T> result(count);
    backend->CopyToHost(result.data(), beliefs, count);
    return result;
  }

  void ProjectForward() noexcept
  {
    for (size_t i = 0; i + 1 < layers.size(); ++i)
    {
      layers[i]->ComputeMuOnly();

      if (layers[i + 1]->IsClamped())
        continue;

      const T* mu = layers[i]->GetMu();
      T* nextZ = layers[i + 1]->GetBeliefs();
      size_t n = layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

      backend->Copy(nextZ, mu, n);
    }
  }

  AccumT TrainStepWithProjection(const std::vector<T>& x, const std::vector<T>& y,
                                int inferenceSteps)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    ProjectForward();
    for (int t = 0; t < inferenceSteps; ++t)
    {
      CalculateState(false);
      UpdateState();
    }
    UpdateWeights();

    AccumT finalEnergy = CalculateState();
    GetTerminalLayer()->UnclampState();

    return finalEnergy;
  }

  std::vector<T> PredictWithProjection(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    ProjectForward();
    for (int t = 0; t < inferenceSteps; ++t)
    {
      CalculateState(false);
      UpdateState();
    }

    ConvPCLayerT<T, AccumT>* terminal = GetTerminalLayer();
    const T* beliefs = terminal->GetBeliefs();
    size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

    std::vector<T> result(count);
    backend->CopyToHost(result.data(), beliefs, count);
    return result;
  }

private:
  std::vector<std::unique_ptr<ConvPCLayerT<T, AccumT>>> layers;
  std::unique_ptr<IComputeBackendT<T, AccumT>> backend;
  DeviceType device;
  std::unique_ptr<MemoryArenaT<T>> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArenaT<T, AccumT>> gpuArena;
#endif
  int batchSize;
  OptimizerType pendingOpt = OptimizerType::SGD;
};
} // namespace Deep
