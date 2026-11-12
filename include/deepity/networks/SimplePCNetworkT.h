#pragma once
#include <deepity/backend/BackendT.h>
#include <deepity/layers/SimplePCLayerT.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <pmmintrin.h>
#include <random>
#include <vector>
#include <xmmintrin.h>

/**
 * @file SimplePCNetworkT.h
 * @brief Templated counterpart to SimplePCNetwork, over the same `T`/
 * `AccumT` pair as SimplePCLayerT. Has no CUDA graph capture path.
 */

namespace Deep
{
/// @brief Templated counterpart to SimplePCNetwork.
template <typename T, typename AccumT = T> class SimplePCNetworkT
{
public:
  /// @brief This network's storage element type.
  using ElementType = T;

  /// @brief Constructs an empty network; add layers via AddLayer(),
  /// then Compile() before use.
  explicit SimplePCNetworkT(int batchSize, DeviceType device = DeviceType::DEVICE_CPU)
      : device(device)
      , batchSize(batchSize)
  {
    backend = CreateBackendT<T, AccumT>(device);
  }
  ~SimplePCNetworkT() = default;

  SimplePCNetworkT(const SimplePCNetworkT&) = delete;
  SimplePCNetworkT& operator=(const SimplePCNetworkT&) = delete;

  /// @brief Adds a layer to the network. See SimplePCLayerT's own
  /// constructor doc comment for each parameter's meaning.
  void AddLayer(size_t size, size_t nextSize, AccumT lr, AccumT ir, AccumT lmbda,
               ActivationType aType, ActivationType dType)
  {
    auto l = std::make_unique<SimplePCLayerT<T, AccumT>>(size, nextSize, batchSize, lr, ir, lmbda,
                                                         aType, dType, backend.get());

    if (!layers.empty())
    {
      layers.back()->SetLayerAbove(l.get());
      l->SetLayerBelow(layers.back().get());
    }
    layers.push_back(std::move(l));
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

  /// @brief Re-centers every non-terminal layer's W rows after each UpdateWeights().
  /// Off by default.
  void SetCenterWeights(bool enabled) noexcept
  {
    if (enabled == centerWeights)
      return;
    centerWeights = enabled;
  }

  void UpdateWeights() noexcept
  {
    for (size_t i = 0; i + 1 < layers.size(); i++)
      layers[i]->UpdateWeights();
    if (centerWeights)
      for (size_t i = 0; i + 1 < layers.size(); ++i)
        layers[i]->CenterWeights();
  }

  SimplePCLayerT<T, AccumT>* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  std::vector<std::unique_ptr<SimplePCLayerT<T, AccumT>>>& GetLayers() noexcept
  {
    return layers;
  }
  const std::vector<std::unique_ptr<SimplePCLayerT<T, AccumT>>>& GetLayers() const noexcept
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

  void SetOptimizer(OptimizerType o) noexcept
  {
    for (auto& layer : layers)
      layer->SetOptimizer(o);
  }

  /// @brief Runs a complete training step (clamp, settle, update, unclamp).
  AccumT TrainStep(const std::vector<T>& x, const std::vector<T>& y, int inferenceSteps)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    for (int t = 0; t < inferenceSteps; t++)
    {
      CalculateState(false);
      UpdateState();
    }

    AccumT finalEnergy = CalculateState(true);

    UpdateWeights();
    GetTerminalLayer()->UnclampState();

    return finalEnergy;
  }

  /// @brief Forward prediction: clamp input, settle, read terminal beliefs.
  std::vector<T> Predict(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    for (int t = 0; t < inferenceSteps; t++)
    {
      CalculateState(false);
      UpdateState();
    }

    SimplePCLayerT<T, AccumT>* terminal = GetTerminalLayer();
    const T* beliefs = terminal->GetBeliefs();
    size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

    std::vector<T> result(count);
    backend->CopyToHost(result.data(), beliefs, count);
    return result;
  }

  /// @brief Seeds hidden layers from a genuine forward pass through
  /// current weights, instead of zero-init.
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

  /// @brief Full train step with forward-projection initialization.
  AccumT TrainStepWithProjection(const std::vector<T>& x, const std::vector<T>& y,
                                int inferenceSteps, bool computeEnergy = true)
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

    AccumT finalEnergy = CalculateState(computeEnergy);
    GetTerminalLayer()->UnclampState();

    return finalEnergy;
  }

  /// @brief Inference-only counterpart to TrainStepWithProjection().
  std::vector<T> PredictWithProjection(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    ProjectForward();
    for (int t = 0; t < inferenceSteps; t++)
    {
      CalculateState(false);
      UpdateState();
    }

    SimplePCLayerT<T, AccumT>* terminal = GetTerminalLayer();
    const T* beliefs = terminal->GetBeliefs();
    size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

    std::vector<T> result(count);
    backend->CopyToHost(result.data(), beliefs, count);
    return result;
  }

  /// @brief Sets mu-cache threshold on every layer. Safe to call any
  /// time after Compile().
  void SetMuCacheThreshold(AccumT threshold) noexcept
  {
    for (auto& l : layers)
      l->SetMuCacheThreshold(threshold);
  }

  /// @brief Loads all layers into one contiguous block of memory.
  void Compile()
  {
#pragma omp parallel
    {
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
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
  bool centerWeights = false;
  std::vector<std::unique_ptr<SimplePCLayerT<T, AccumT>>> layers;

  std::unique_ptr<IComputeBackendT<T, AccumT>> backend;
  DeviceType device;

  std::unique_ptr<MemoryArenaT<T>> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArenaT<T, AccumT>> gpuArena;
#endif

  int batchSize;
};
} // namespace Deep
