#pragma once
#include <deepity/backend/BackendT.h>
#include <deepity/layers/GaussSeidelPCLayerT.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <vector>

/**
 * @file GaussSeidelPCNetworkT.h
 * @brief Templated counterpart to GaussSeidelPCNetwork, over the same
 * `T`/`AccumT` pair as GaussSeidelPCLayerT. Orchestrates
 * GaussSeidelPCLayerT's three-phase settling step, in order:
 *
 *   1. UpdateState() on every layer       (uses previous timestep's mu/e)
 *   2. ComputePrediction() on every layer (uses this timestep's fresh z)
 *   3. ComputeError() on every layer      (uses this timestep's fresh
 *                                          z and layerBelow's fresh mu)
 *
 * Has no CUDA graph capture path.
 */

namespace Deep
{
/// @brief Templated counterpart to GaussSeidelPCNetwork.
template <typename T, typename AccumT = T> class GaussSeidelPCNetworkT
{
public:
  /// @brief This network's storage element type.
  using ElementType = T;

  /// @brief Constructs an empty network; add layers via AddLayer(),
  /// then Compile() before use.
  explicit GaussSeidelPCNetworkT(int batchSize, DeviceType device = DeviceType::DEVICE_CPU)
      : device(device)
      , batchSize(batchSize)
  {
    backend = CreateBackendT<T, AccumT>(device);
  }
  ~GaussSeidelPCNetworkT() = default;

  GaussSeidelPCNetworkT(const GaussSeidelPCNetworkT&) = delete;
  GaussSeidelPCNetworkT& operator=(const GaussSeidelPCNetworkT&) = delete;

  /// @brief Adds a layer to the network. See GaussSeidelPCLayerT's
  /// own constructor doc comment for each parameter's meaning.
  void AddLayer(size_t size, size_t nextSize, AccumT lr, AccumT ir, AccumT lmbda,
               ActivationType aType, ActivationType dType)
  {
    auto l = std::make_unique<GaussSeidelPCLayerT<T, AccumT>>(size, nextSize, batchSize, lr, ir,
                                                              lmbda, aType, dType, backend.get());

    if (!layers.empty())
    {
      layers.back()->SetLayerAbove(l.get());
      l->SetLayerBelow(layers.back().get());
    }
    layers.push_back(std::move(l));
  }

  void SetOptimizer(OptimizerType opt) noexcept
  {
    for (auto& layer : layers)
      layer->SetOptimizer(opt);
  }

  void RandomizeWeights(std::mt19937& rng, const char* distribution = "normal")
  {
    for (auto& l : layers)
      l->RandomizeWeights(rng, distribution);
  }

  void ResetState() noexcept
  {
    for (auto& l : layers)
      l->ResetState();
  }

  void Clamp(const std::vector<T>& input)
  {
    layers.front()->ClampState(input);
  }

  /// @brief Runs one full Gauss-Seidel settling step (all three
  /// sweeps, in order) across every layer.
  /// @return Total energy, summed from every layer's
  /// ComputeError(), or AccumT(0) if needEnergy is false.
  AccumT Step(bool needEnergy = true) noexcept
  {
    for (auto& l : layers)
      l->UpdateState();

    for (auto& l : layers)
      l->ComputePrediction();

    AccumT totalEnergy(0);
    for (auto& l : layers)
      totalEnergy += l->ComputeError(needEnergy);

    return needEnergy ? totalEnergy : AccumT(0);
  }

  void UpdateWeights() noexcept
  {
    for (size_t i = 0; i + 1 < layers.size(); i++)
      layers[i]->UpdateWeights();
  }

  /// @brief Seeds every hidden layer's z from a genuine forward pass
  /// through current weights, instead of zero-init.
  void ProjectForward() noexcept
  {
    for (size_t i = 0; i + 1 < layers.size(); ++i)
    {
      layers[i]->ComputePrediction();

      if (layers[i + 1]->IsClamped())
        continue;

      const T* mu = layers[i]->GetMu();
      T* nextZ = layers[i + 1]->GetBeliefs();
      size_t n = layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

      backend->Copy(nextZ, mu, n);
    }
  }

  GaussSeidelPCLayerT<T, AccumT>* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  std::vector<std::unique_ptr<GaussSeidelPCLayerT<T, AccumT>>>& GetLayers() noexcept
  {
    return layers;
  }
  const std::vector<std::unique_ptr<GaussSeidelPCLayerT<T, AccumT>>>& GetLayers() const noexcept
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

  void SetLearningRate(AccumT lr) noexcept
  {
    for (auto& layer : layers)
      layer->SetLearningRate(lr);
  }

  /// @brief Full train step: reset, clamp input+target, settle for
  /// inferenceSteps, update weights once, unclamp.
  AccumT TrainStep(const std::vector<T>& x, const std::vector<T>& y, int inferenceSteps)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    for (int t = 0; t < inferenceSteps; ++t)
      Step(false);
    UpdateWeights();

    // Energy readout: sweeps 2+3 only, from the current z, without an
    // extra sweep-1 UpdateState() beyond inferenceSteps.
    AccumT finalEnergy(0);
    for (auto& l : layers)
      l->ComputePrediction();
    for (auto& l : layers)
      finalEnergy += l->ComputeError();

    GetTerminalLayer()->UnclampState();

    return finalEnergy;
  }

  /// @brief Same as TrainStep(), but with ProjectForward() called once
  /// after clamping the input, before the settling loop.
  AccumT TrainStepWithProjection(const std::vector<T>& x, const std::vector<T>& y,
                                 int inferenceSteps)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    ProjectForward();
    GetTerminalLayer()->ComputeError(false);
    for (int t = 0; t < inferenceSteps; ++t)
      Step(false);
    UpdateWeights();

    AccumT finalEnergy(0);
    for (auto& l : layers)
      l->ComputePrediction();
    for (auto& l : layers)
      finalEnergy += l->ComputeError();

    GetTerminalLayer()->UnclampState();

    return finalEnergy;
  }

  /// @brief Forward prediction: clamp input, settle (no target
  /// clamped), read the terminal's beliefs.
  std::vector<T> Predict(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    // Seed the hidden states with the forward pass. Without this,
    // inference starts from z=0 and fails to reach the target.
    ProjectForward();
    for (int t = 0; t < inferenceSteps; ++t)
      Step();

    GaussSeidelPCLayerT<T, AccumT>* terminal = GetTerminalLayer();
    const T* beliefs = terminal->GetBeliefs();
    size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

    std::vector<T> result(count);
    backend->CopyToHost(result.data(), beliefs, count);
    return result;
  }

  /// @brief Loads all layers into one contiguous block of memory.
  void Compile()
  {
    size_t total_elements_needed = 0;
    for (auto& layer : layers)
      total_elements_needed += layer->GetRequiredElements();

    backend->PrepareForBatchSize(batchSize);

    if (device == DeviceType::DEVICE_CPU)
    {
      cpuArena = std::make_unique<MemoryArenaT<T>>(total_elements_needed, true);
      for (auto& layer : layers)
        layer->BindMemory(*cpuArena);
    }
#if defined(DEEPITY_USE_CUDA)
    else
    {
      gpuArena =
          std::make_unique<DeviceMemoryArenaT<T, AccumT>>(backend.get(), total_elements_needed);
      for (auto& layer : layers)
        layer->BindMemory(*gpuArena);
    }
#endif
  }

private:
  std::vector<std::unique_ptr<GaussSeidelPCLayerT<T, AccumT>>> layers;

  std::unique_ptr<IComputeBackendT<T, AccumT>> backend;
  DeviceType device;

  std::unique_ptr<MemoryArenaT<T>> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArenaT<T, AccumT>> gpuArena;
#endif

  int batchSize;
};
} // namespace Deep
