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
 * `T`/`AccumT` pair as GaussSeidelPCLayerT -- see FullPCNetworkT's own
 * doc comment for the full rationale. Produced by mechanically
 * retyping GaussSeidelPCNetwork's own `float` members/parameters to
 * `T`/`AccumT`. Orchestrates GaussSeidelPCLayerT's three-phase
 * settling step in the same fixed order as GaussSeidelPCNetwork's own
 * class comment requires:
 *
 *   1. UpdateState() on EVERY layer       (uses PREVIOUS timestep's mu/e)
 *   2. ComputePrediction() on EVERY layer (uses THIS timestep's fresh z)
 *   3. ComputeError() on EVERY layer      (uses THIS timestep's fresh
 *                                          z and layerBelow's fresh mu)
 *
 * CUDA graph capture (BeginGraphCapture/ReplayGraph and the
 * graphCaptured-style flags GaussSeidelPCNetwork carries) is dropped
 * entirely, same reasoning as FullPCNetworkT/SimplePCNetworkT:
 * device==DEVICE_GPU already throws inside CreateBackendT() at
 * construction (no CUDABackendT yet), so there's no graph path to
 * reach here, and no runtime toggle that could leave a stale captured
 * graph behind in the first place -- switching precision means
 * constructing a brand-new GaussSeidelPCNetworkT<OtherT>.
 */

namespace Deep
{
/// @brief Templated counterpart to GaussSeidelPCNetwork, see this
/// file's own comment for the full rationale.
template <typename T, typename AccumT = T> class GaussSeidelPCNetworkT
{
public:
  /// @brief This network's storage element type, see
  /// FullPCNetworkT::ElementType's own doc comment.
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

  /// @brief Runs ONE full Gauss-Seidel settling step (all three
  /// sweeps, in order) across every layer. See this file's own class
  /// comment for the required order.
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
  /// through current weights. See GaussSeidelPCNetwork::ProjectForward()'s
  /// own doc comment.
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
  /// inferenceSteps, update weights once, unclamp. See
  /// GaussSeidelPCNetwork::TrainStep()'s own doc comment; no
  /// CUDA-graph path here, see this file's own class comment.
  AccumT TrainStep(const std::vector<T>& x, const std::vector<T>& y, int inferenceSteps)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    for (int t = 0; t < inferenceSteps; ++t)
      Step(false);
    UpdateWeights();

    // Fresh, uncaptured energy readout: sweeps 2+3 only (recompute
    // predictions/errors from the CURRENT z), deliberately without
    // sweep 1's UpdateState() -- calling the full Step() again here
    // would apply an extra, unintended settling step beyond
    // inferenceSteps, same reasoning as GaussSeidelPCNetwork's own
    // identical comment.
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
