#pragma once
#include <deepity/backend/BackendT.h>
#include <deepity/layers/DirectKPPCLayerT.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file DirectKPPCNetworkT.h
 * @brief Templated counterpart to DirectKPPCNetwork, over the same
 * `T`/`AccumT` pair as DirectKPPCLayerT. Orchestrates DirectKPPCLayerT's
 * four DKP-PC phases in order:
 *
 *   0. Forward initialization: clamp input, ProjectForward(), clamp
 *      target, CalculateTerminalError().
 *   1. DirectFeedbackUpdate() on every non-terminal layer.
 *   2. Ordinary PC settling (Step()) for inferenceSteps steps.
 *   3. UpdateWeights() on every non-terminal layer.
 *
 * Has no CUDA graph capture path.
 *
 * @warning Not yet gradient-checked at the network level.
 */

namespace Deep
{
/// @brief Templated counterpart to DirectKPPCNetwork.
template <typename T, typename AccumT = T> class DirectKPPCNetworkT
{
public:
  /// @brief This network's storage element type.
  using ElementType = T;

  /// @brief Constructs an empty network; add layers via AddLayer(),
  /// then Compile() before use.
  explicit DirectKPPCNetworkT(int batchSize, DeviceType device = DeviceType::DEVICE_CPU)
      : device(device)
      , batchSize(batchSize)
  {
    backend = CreateBackendT<T, AccumT>(device);
  }
  ~DirectKPPCNetworkT() = default;

  DirectKPPCNetworkT(const DirectKPPCNetworkT&) = delete;
  DirectKPPCNetworkT& operator=(const DirectKPPCNetworkT&) = delete;

  /// @brief Adds a layer to the network. See DirectKPPCLayerT's own
  /// constructor doc comment for each parameter's meaning.
  void AddLayer(size_t size, size_t nextSize, size_t terminalSize, AccumT lr, AccumT ir, AccumT fl,
               AccumT lmbda, ActivationType aType, ActivationType dType)
  {
    auto l = std::make_unique<DirectKPPCLayerT<T, AccumT>>(size, nextSize, terminalSize, batchSize,
                                                           lr, ir, fl, lmbda, aType, dType,
                                                           backend.get());

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

  /// @brief Seeds every hidden layer's z from a genuine forward pass
  /// through current weights.
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

  /// @brief Phase 1: runs DirectFeedbackUpdate() on every
  /// non-terminal layer. Order among layers does not matter.
  void DirectFeedbackUpdate() noexcept
  {
    for (size_t i = 0; i < layers.size() - 1; ++i)
      layers[i]->DirectFeedbackUpdate();
  }

  /// @brief Phase 0's final step: computes the terminal layer's error
  /// alone, producing a real, non-zero epsilon_L.
  AccumT CalculateTerminalError() noexcept
  {
    return GetTerminalLayer()->CalculateState(false);
  }

  /// @brief Phase 2: runs one ordinary PC settling step
  /// (CalculateState() + UpdateState()) across every layer.
  AccumT Step(bool needEnergy = true) noexcept
  {
    AccumT e(0);
    for (auto& l : layers)
      e += l->CalculateState(needEnergy);
    for (auto& l : layers)
      l->UpdateState();
    return needEnergy ? e : AccumT(0);
  }

  /// @brief Phase 3: updates every non-terminal layer's W and Psi.
  void UpdateWeights() noexcept
  {
    // Refreshes preact against the current z before computing gradients.
    for (size_t i = 0; i + 1 < layers.size(); ++i)
      layers[i]->ComputeMuOnly();

    for (size_t i = 0; i + 1 < layers.size(); i++)
      layers[i]->UpdateWeights();
  }

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
  void SetLearningRate(AccumT lr) noexcept
  {
    for (auto& layer : layers)
      layer->SetLearningRate(lr);
  }
  void SetFeedbackRate(AccumT fl) noexcept
  {
    for (auto& layer : layers)
      layer->SetFeedbackRate(fl);
  }

  DirectKPPCLayerT<T, AccumT>* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  std::vector<std::unique_ptr<DirectKPPCLayerT<T, AccumT>>>& GetLayers() noexcept
  {
    return layers;
  }
  const std::vector<std::unique_ptr<DirectKPPCLayerT<T, AccumT>>>& GetLayers() const noexcept
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

  /// @brief Full train step: reset, clamp input+target, run all four
  /// DKP-PC phases in order, unclamp.
  /// @param inferenceSteps Number of settling steps for phase 2.
  /// Defaults to 1.
  AccumT TrainStep(const std::vector<T>& x, const std::vector<T>& y, int inferenceSteps = 1)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    ProjectForward();
    CalculateTerminalError();
    DirectFeedbackUpdate();
    for (int t = 0; t < inferenceSteps; t++)
      Step(false);
    UpdateWeights();

    AccumT finalEnergy(0);
    for (auto& l : layers)
      finalEnergy += l->CalculateState(true);

    GetTerminalLayer()->UnclampState();
    return finalEnergy;
  }

  /// @brief Forward prediction: clamp input, settle with no DFA
  /// perturbation and no target clamped, read the terminal's beliefs.
  std::vector<T> Predict(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    ProjectForward();
    for (int t = 0; t < inferenceSteps; t++)
      Step();

    DirectKPPCLayerT<T, AccumT>* terminal = GetTerminalLayer();
    const T* beliefs = terminal->GetBeliefs();
    size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

    std::vector<T> result(count);
    backend->CopyToHost(result.data(), beliefs, count);
    return result;
  }

  /// @brief Loads all layers into one contiguous block of memory,
  /// and wires layerAbove/layerBelow/terminalLayer across every
  /// layer.
  void Compile()
  {
    if (!layers.empty() && layers.back()->GetInputSize() != layers.back()->GetTerminalSize())
    {
      throw std::invalid_argument(
          "DirectKPPCNetworkT::Compile(): terminalSize (" +
          std::to_string(layers.back()->GetTerminalSize()) +
          ") passed to AddLayer() doesn't match the actual terminal layer's size (" +
          std::to_string(layers.back()->GetInputSize()) + ").");
    }

    size_t total_elements_needed = 0;
    for (auto& layer : layers)
      total_elements_needed += layer->GetRequiredElements();

    backend->PrepareForBatchSize(batchSize);

    if (device == DeviceType::DEVICE_CPU)
    {
      cpuArena = std::make_unique<MemoryArenaT<T>>(total_elements_needed, false);
      for (auto& layer : layers)
      {
        layer->BindMemory(*cpuArena);
        layer->SetTerminalLayer(layers.back().get());
      }
    }
#if defined(DEEPITY_USE_CUDA)
    else
    {
      gpuArena =
          std::make_unique<DeviceMemoryArenaT<T, AccumT>>(backend.get(), total_elements_needed);
      for (auto& layer : layers)
      {
        layer->BindMemory(*gpuArena);
        layer->SetTerminalLayer(layers.back().get());
      }
    }
#endif
  }

private:
  std::vector<std::unique_ptr<DirectKPPCLayerT<T, AccumT>>> layers;

  std::unique_ptr<IComputeBackendT<T, AccumT>> backend;
  DeviceType device;

  std::unique_ptr<MemoryArenaT<T>> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArenaT<T, AccumT>> gpuArena;
#endif

  int batchSize;
};
} // namespace Deep
