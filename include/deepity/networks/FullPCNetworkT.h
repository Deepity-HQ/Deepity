#pragma once
#include <algorithm>
#include <cmath>
#include <deepity/backend/BackendT.h>
#include <deepity/layers/FullPCLayerT.h>
#include <deepity/utils/DeviceMemoryArena.h>
#include <deepity/utils/MemoryArena.h>
#include <memory>
#include <pmmintrin.h>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <xmmintrin.h>

/**
 * @file FullPCNetworkT.h
 * @brief Templated counterpart to FullPCNetwork, over the same `T`
 * (storage) / `AccumT` (optimizer accumulator) pair as FullPCLayerT --
 * see that class's own doc comment. Produced by mechanically retyping
 * FullPCNetwork's own `float` members/parameters to `T`/`AccumT`, NOT a
 * reimplementation: every method here must stay line-for-line
 * equivalent to FullPCNetwork's own method bodies, so that
 * FullPCNetworkT<float, float> reproduces FullPCNetwork bit-exactly.
 */

namespace Deep
{
/// @brief Templated counterpart to FullPCNetwork, see this file's own
/// comment for the full rationale.
template <typename T, typename AccumT = T>
class FullPCNetworkT
{
public:
  /// @brief This network's storage element type, for ModelIO's dtype
  /// derivation/checking (see DTypeTag<T>). ModelIO itself actually
  /// reads the type off each tensor's own GetStateDict() pointer
  /// rather than this typedef, but it documents the network's own
  /// element type at the type level, for anyone reading this class.
  using ElementType = T;

  /// @brief Constructs an empty network; add layers via AddLayer(),
  /// then Compile() before use.
  /// @param batchSize Fixed batch size for every layer.
  /// @param device Which device this network's layers run on.
  explicit FullPCNetworkT(int batchSize, DeviceType device = DeviceType::DEVICE_CPU) noexcept
      : device(device)
      , batchSize(batchSize)
  {
    backend = CreateBackendT<T, AccumT>(device);
  }
  ~FullPCNetworkT() = default;

  FullPCNetworkT(const FullPCNetworkT&) = delete;
  FullPCNetworkT& operator=(const FullPCNetworkT&) = delete;

  /// @brief Adds a layer to the network. See FullPCLayerT's own
  /// constructor doc comment for each parameter's meaning.
  void AddLayer(size_t size, size_t nextSize, size_t terminalSize, AccumT lr, AccumT ir, AccumT fl,
               AccumT lmbda, ActivationType aType, ActivationType dType)
  {
    auto l = std::make_unique<FullPCLayerT<T, AccumT>>(size, nextSize, terminalSize, batchSize, lr,
                                                       ir, fl, lmbda, aType, dType, backend.get());

    if (!layers.empty())
    {
      layers.back()->SetLayerAbove(l.get());
      l->SetLayerBelow(layers.back().get());
    }

    layers.push_back(std::move(l));
  }

  /// @brief Enables muPC's full Table 1 parameterization, see
  /// FullPCNetwork::SetUseMuPCScaling()'s own doc comment.
  void SetUseMuPCScaling(bool enabled) noexcept
  {
    useMuPCScaling = enabled;
  }

  /// @brief Enables residual/skip connections, see
  /// FullPCNetwork::SetUseResidualConnections()'s own doc comment.
  void SetUseResidualConnections(bool enabled) noexcept
  {
    useResidualConnections = enabled;
  }

  /// @brief Randomizes the weights (W and Psi) of each layer.
  void RandomizeWeights(std::mt19937& rng, const char* distribution = "normal")
  {
    for (auto& l : layers)
      l->RandomizeWeights(rng, distribution);
  }

  /// @brief Resets each layer's state without touching learned weights.
  void ResetState() noexcept
  {
    for (auto& l : layers)
      l->ResetState();
  }

  /// @brief Clamps the input to the first layer.
  void Clamp(const std::vector<T>& input)
  {
    layers.front()->ClampState(input);
  }

  /// @brief Seeds hidden layers from a genuine forward pass through
  /// current weights, see FullPCNetwork::ProjectForward()'s own doc comment.
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

  /// @brief Runs the DFA phase on every non-terminal layer.
  void DirectFeedbackUpdate() noexcept
  {
    for (size_t i = 0; i < layers.size() - 1; ++i)
      layers[i]->DirectFeedbackUpdate();
  }

  /// @brief Computes the terminal layer's error/energy against its
  /// clamped target.
  AccumT CalculateTerminalError() noexcept
  {
    return GetTerminalLayer()->CalculateState(false);
  }

  /// @brief Runs one settling step on every layer.
  AccumT Step(bool needEnergy = true) noexcept
  {
    AccumT e(0);
    for (auto& l : layers)
      e += l->CalculateState(needEnergy);
    for (auto& l : layers)
      l->UpdateState();
    return needEnergy ? e : AccumT(0);
  }

  /// @brief ePC's settling step, see FullPCNetwork::EPCStep()'s own doc comment.
  void EPCStep() noexcept
  {
    layers[0]->ComputeMuOnly();
    for (size_t i = 1; i + 1 < layers.size(); ++i)
    {
      layers[i]->ReconstructBelief();
      layers[i]->ComputeMuOnly();
    }
    GetTerminalLayer()->CalculateState(false);

    const T* adjointAbove = GetTerminalLayer()->GetErrors();
    bool firstHop = true;
    for (size_t i = layers.size() - 2; i >= 1; --i)
    {
      layers[i]->ComputeAdjoint(adjointAbove, firstHop ? AccumT(-1) : AccumT(1));
      firstHop = false;
      layers[i]->UpdateErrorEPC();
      adjointAbove = layers[i]->GetAdjoint();
    }
  }

  /// @brief Applies weight updates to every layer.
  void UpdateWeights() noexcept
  {
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

  FullPCLayerT<T, AccumT>* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  std::vector<std::unique_ptr<FullPCLayerT<T, AccumT>>>& GetLayers() noexcept
  {
    return layers;
  }
  const std::vector<std::unique_ptr<FullPCLayerT<T, AccumT>>>& GetLayers() const noexcept
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
  /// DKP-PC phases in order, unclamp. See FullPCNetwork::TrainStep()'s
  /// own doc comment. CUDA graph capture/replay is not reachable here:
  /// device==DEVICE_GPU already throws inside CreateBackendT() at
  /// construction (no CUDABackendT yet), so this just runs the
  /// non-graph path unconditionally.
  AccumT TrainStep(const std::vector<T>& x, const std::vector<T>& y, int inferenceSteps = 1)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    auto settleStep = [this]()
    {
      if (useEPC)
        EPCStep();
      else
        Step(false);
      if (useIPC)
      {
        UpdateWeights();
        for (auto& l : layers)
          l->InvalidateMuCache();
      }
    };

    ProjectForward();
    CalculateTerminalError();
    DirectFeedbackUpdate();
    for (int t = 0; t < inferenceSteps; t++)
      settleStep();
    if (!useIPC)
      UpdateWeights();

    AccumT finalEnergy(0);
    for (auto& l : layers)
      finalEnergy += l->CalculateState(true);

    GetTerminalLayer()->UnclampState();
    return finalEnergy;
  }

  /// @brief Forward prediction: clamp input, settle with no DFA
  /// perturbation and no target clamped, read the terminal's beliefs.
  /// See FullPCNetwork::Predict()'s own doc comment; same CUDA-graph
  /// caveat as TrainStep() above.
  std::vector<T> Predict(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    auto settleStep = [this]()
    {
      if (useEPC)
        EPCStep();
      else
        Step();
    };

    FullPCLayerT<T, AccumT>* terminal = GetTerminalLayer();

    ProjectForward();
    for (int t = 0; t < inferenceSteps; t++)
      settleStep();
    if (terminal->GetCrossEntropy())
    {
      FullPCLayerT<T, AccumT>* below = layers[layers.size() - 2].get();
      below->ComputeMuOnly();
      size_t n = below->GetBatchSize() * below->GetOutputSize();
      backend->Copy(terminal->GetBeliefs(), below->GetMu(), n);
    }

    const T* beliefs = terminal->GetBeliefs();
    size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

    std::vector<T> result(count);
    backend->CopyToHost(result.data(), beliefs, count);
    return result;
  }

  /// @brief Loads all layers into one contiguous block of memory,
  /// see FullPCNetwork::Compile()'s own doc comment.
  void Compile()
  {
    if (!layers.empty() && layers.back()->GetInputSize() != layers.back()->GetTerminalSize())
    {
      throw std::invalid_argument(
          "FullPCNetworkT::Compile(): terminalSize (" +
          std::to_string(layers.back()->GetTerminalSize()) +
          ") passed to AddLayer() doesn't match the actual terminal layer's size (" +
          std::to_string(layers.back()->GetInputSize()) + ").");
    }

#pragma omp parallel
    {
      _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
      _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
    }
    size_t total_elements_needed = 0;
    for (auto& layer : layers)
      total_elements_needed += layer->GetRequiredElements();

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
      backend->PrepareForBatchSize(batchSize);
      gpuArena = std::make_unique<DeviceMemoryArenaT<T, AccumT>>(backend.get(),
                                                                 total_elements_needed);
      for (auto& layer : layers)
      {
        layer->BindMemory(*gpuArena);
        layer->SetTerminalLayer(layers.back().get());
      }
    }
#endif

    if (!useMuPCScaling && !useResidualConnections)
      return;

    const int L = (int)layers.size() - 1;
    const int H = L - 1;

    if (L < 1)
      return;

    if (useMuPCScaling)
    {
      const AccumT d_in = AccumT(layers[0]->GetInputSize());

      AccumT N;
      if (H > 0)
      {
        N = AccumT(0);
        for (int i = 0; i < L - 1; ++i)
          N = (std::max)(N, AccumT(layers[i]->GetOutputSize()));
      }
      else
      {
        N = AccumT(layers[L - 1]->GetOutputSize());
      }

      for (int l = 1; l <= L; ++l)
      {
        AccumT a;
        if (l == 1)
          a = std::pow(d_in, AccumT(-0.5));
        else if (l == L)
          a = AccumT(1) / N;
        else
          a = std::pow(N * AccumT(L), AccumT(-0.5));

        layers[l - 1]->SetMuPCScale(a);
        layers[l - 1]->SetMuPCInit(true);
      }
    }

    if (useResidualConnections)
    {
      for (int l = 2; l <= H; ++l)
      {
        FullPCLayerT<T, AccumT>* layer = layers[l - 1].get();
        if (layer->GetInputSize() != layer->GetOutputSize())
        {
          throw std::invalid_argument(
              "FullPCNetworkT::Compile(): residual connections require "
              "matching width, layer index " +
              std::to_string(l - 1) + " has input size " + std::to_string(layer->GetInputSize()) +
              " but output size " + std::to_string(layer->GetOutputSize()) + ".");
        }
        layer->SetResidual(true);
      }
    }
  }

  void SetUseIPC(bool enabled) noexcept
  {
    useIPC = enabled;
  }

  void SetUseMomentum(bool enabled, AccumT beta = AccumT(0.9)) noexcept
  {
    for (auto& layer : layers)
      layer->SetMomentum(enabled, beta);
  }

  void SetUseCrossEntropy(bool enabled) noexcept
  {
    GetTerminalLayer()->SetCrossEntropy(enabled);
  }

  void SetUseEPC(bool enabled) noexcept
  {
    useEPC = enabled;
    for (size_t i = 1; i + 1 < layers.size(); ++i)
      layers[i]->SetUseEPC(enabled);
  }

private:
  std::vector<std::unique_ptr<FullPCLayerT<T, AccumT>>> layers;

  std::unique_ptr<IComputeBackendT<T, AccumT>> backend;
  DeviceType device;

  std::unique_ptr<MemoryArenaT<T>> cpuArena;
#if defined(DEEPITY_USE_CUDA)
  std::unique_ptr<DeviceMemoryArenaT<T, AccumT>> gpuArena;
#endif

  int batchSize;
  bool useIPC = false;
  bool useEPC = false;

  bool useMuPCScaling = false;
  bool useResidualConnections = false;
};
} // namespace Deep
