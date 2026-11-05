#pragma once
#include <algorithm>
#include <cmath>
#include <deepity/backend/BackendT.h>
#include <deepity/layers/FullConvPCLayerT.h>
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
 * @file FullConvPCNetworkT.h
 * @brief Templated counterpart to FullConvPCNetwork, over the same
 * `T`/`AccumT` pair as FullConvPCLayerT. Has no CUDA graph capture
 * path, and does not implement Save()/Load() or
 * GetAllLayerDiagnostics().
 */

namespace Deep
{
/// @brief Templated counterpart to FullConvPCNetwork.
template <typename T, typename AccumT = T> class FullConvPCNetworkT
{
public:
  using ElementType = T;

  explicit FullConvPCNetworkT(int batchSize, DeviceType device = DeviceType::DEVICE_CPU)
      : device(device)
      , batchSize(batchSize)
  {
    backend = CreateBackendT<T, AccumT>(device);
  }
  ~FullConvPCNetworkT() = default;

  FullConvPCNetworkT(const FullConvPCNetworkT&) = delete;
  FullConvPCNetworkT& operator=(const FullConvPCNetworkT&) = delete;

  /// @param outChannels 0 marks a terminal layer.
  /// @param terminalSize the network's final output size (flat channel
  /// count), required on every AddLayer() call, validated against the
  /// actual terminal layer's inChannels at Compile() time.
  void AddLayer(int inChannels, int outChannels, int inHeight, int inWidth, int kernelH,
               int kernelW, int strideH, int strideW, int padH, int padW, int terminalSize,
               AccumT lr, AccumT ir, AccumT fl, AccumT lmbda, ActivationType aType,
               ActivationType dType, int poolH = 1, int poolW = 1, int poolStrideH = 1,
               int poolStrideW = 1)
  {
    auto l = std::make_unique<FullConvPCLayerT<T, AccumT>>(
        inChannels, outChannels, inHeight, inWidth, kernelH, kernelW, strideH, strideW, padH,
        padW, terminalSize, batchSize, lr, ir, fl, lmbda, aType, dType, poolH, poolW, poolStrideH,
        poolStrideW, backend.get());

    if (!layers.empty())
    {
      layers.back()->SetLayerAbove(l.get());
      l->SetLayerBelow(layers.back().get());
    }

    layers.push_back(std::move(l));
  }

  void SetUseMuPCScaling(bool enabled) noexcept
  {
    useMuPCScaling = enabled;
  }
  void SetUseResidualConnections(bool enabled) noexcept
  {
    useResidualConnections = enabled;
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

  void DirectFeedbackUpdate() noexcept
  {
    for (size_t i = 0; i < layers.size() - 1; ++i)
      layers[i]->DirectFeedbackUpdate();
  }

  AccumT CalculateTerminalError() noexcept
  {
    return GetTerminalLayer()->CalculateState(false);
  }

  AccumT Step(bool needEnergy = true) noexcept
  {
    AccumT e(0);
    for (auto& l : layers)
      e += l->CalculateState(needEnergy);
    for (auto& l : layers)
      l->UpdateState();
    return needEnergy ? e : AccumT(0);
  }

  /// @brief ePC's settling step: full forward sweep (bottom to top)
  /// then full backward sweep (top to bottom).
  void EPCStep() noexcept
  {
    layers[0]->ComputeMuOnly();
    for (size_t i = 1; i + 1 < layers.size(); ++i)
    {
      layers[i]->ReconstructBelief();
      layers[i]->ComputeMuOnly();
    }

    GetTerminalLayer()->CalculateState(false);

    for (size_t i = 0; i + 1 < layers.size(); ++i)
      if (layers[i]->IsClamped())
        layers[i]->EnsureMuHoldsDerivative();

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

  void UpdateWeights() noexcept
  {
    for (size_t i = 0; i + 1 < layers.size(); ++i)
    {
      layers[i]->ComputeMuOnly();
      layers[i]->EnsureMuHoldsDerivative();
    }
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
  void SetInferenceRate(AccumT ir) noexcept
  {
    for (auto& layer : layers)
      layer->SetInferenceRate(ir);
  }
  void SetAdamEpsilon(AccumT eps) noexcept
  {
    for (auto& layer : layers)
      layer->SetAdamEpsilon(eps);
  }
  void SetUsePCXInit(bool enabled) noexcept
  {
    for (auto& layer : layers)
      layer->SetPCXInit(enabled);
  }

  FullConvPCLayerT<T, AccumT>* GetTerminalLayer() noexcept
  {
    return layers.back().get();
  }
  std::vector<std::unique_ptr<FullConvPCLayerT<T, AccumT>>>& GetLayers() noexcept
  {
    return layers;
  }
  const std::vector<std::unique_ptr<FullConvPCLayerT<T, AccumT>>>& GetLayers() const noexcept
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

  AccumT TrainStep(const std::vector<T>& x, const std::vector<T>& y, int inferenceSteps = 1,
                   bool computeEnergy = true)
  {
    ResetState();
    Clamp(x);
    GetTerminalLayer()->ClampState(y);

    bool skipRedundantProjection = useEPC;
    if (skipRedundantProjection)
    {
      for (size_t i = 0; i + 1 < layers.size(); ++i)
      {
        if (!layers[i]->IsFlZero())
        {
          skipRedundantProjection = false;
          break;
        }
      }
    }

    auto projectAndClampTerminal = [this, skipRedundantProjection]()
    {
      if (skipRedundantProjection)
        return;
      ProjectForward();
      CalculateTerminalError();
    };

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

    auto resyncErrorForClassicSettling = [this]()
    {
      if (useEPC)
        return;
      for (auto& l : layers)
        l->CalculateState(false);
      for (auto& l : layers)
        l->EnsureMuHoldsDerivative();
    };

    projectAndClampTerminal();
    DirectFeedbackUpdate();
    for (int t = 0; t < inferenceSteps; t++)
      settleStep();
    if (!useIPC)
    {
      resyncErrorForClassicSettling();
      UpdateWeights();
    }

    if (computeEnergy)
    {
      lastEnergy = AccumT(0);
      for (auto& l : layers)
        lastEnergy += l->CalculateState(true);
    }

    GetTerminalLayer()->UnclampState();
    return lastEnergy;
  }

  std::vector<T> Predict(const std::vector<T>& x, int inferenceSteps)
  {
    ResetState();
    Clamp(x);

    FullConvPCLayerT<T, AccumT>* terminal = GetTerminalLayer();

    ProjectForward();
    for (int t = 0; t < inferenceSteps; t++)
    {
      if (useEPC)
        EPCStep();
      else
        Step();
    }
    if (terminal->GetCrossEntropy())
    {
      FullConvPCLayerT<T, AccumT>* below = layers[layers.size() - 2].get();
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

  void Compile()
  {
    if (!layers.empty() &&
        (size_t)layers.back()->GetInChannels() != (size_t)layers.back()->GetTerminalSize())
    {
      throw std::invalid_argument(
          "FullConvPCNetworkT::Compile(): terminalSize (" +
          std::to_string(layers.back()->GetTerminalSize()) +
          ") passed to AddLayer() doesn't match the actual terminal layer's inChannels (" +
          std::to_string(layers.back()->GetInChannels()) + ").");
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
      for (auto& layer : layers)
        layer->PrepareConvWorkspace();
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
      const AccumT d_in =
          AccumT(layers[0]->GetInChannels()) * AccumT(layers[0]->GetKernelH()) *
          AccumT(layers[0]->GetKernelW());

      AccumT N;
      AccumT midFanIn;
      if (H > 0)
      {
        N = AccumT(0);
        midFanIn = AccumT(1);
        for (int i = 0; i < L - 1; ++i)
        {
          AccumT outCh = AccumT(layers[i]->GetOutChannels());
          if (outCh > N)
          {
            N = outCh;
            midFanIn = outCh * AccumT(layers[i]->GetKernelH()) * AccumT(layers[i]->GetKernelW());
          }
        }
      }
      else
      {
        N = AccumT(layers[L - 1]->GetOutChannels());
        midFanIn =
            N * AccumT(layers[L - 1]->GetKernelH()) * AccumT(layers[L - 1]->GetKernelW());
      }

      for (int l = 1; l <= L; ++l)
      {
        AccumT a;
        if (l == 1)
          a = std::pow(d_in, AccumT(-0.5));
        else if (l == L)
          a = AccumT(1) / N;
        else
          a = useResidualConnections ? std::pow(midFanIn * AccumT(L), AccumT(-0.5))
                                     : std::pow(midFanIn, AccumT(-0.5));

        layers[l - 1]->SetMuPCScale(a);
        layers[l - 1]->SetMuPCInit(true);
      }
    }

    if (useResidualConnections)
    {
      for (int l = 2; l <= H; ++l)
      {
        FullConvPCLayerT<T, AccumT>* layer = layers[l - 1].get();
        bool matches = layer->GetInChannels() == layer->GetOutChannels() &&
                       layer->GetInHeight() == layer->GetOutHeight() &&
                       layer->GetInWidth() == layer->GetOutWidth();
        if (!matches)
        {
          throw std::invalid_argument(
              "FullConvPCNetworkT::Compile(): residual connections require matching "
              "channels/spatial dims, layer index " +
              std::to_string(l - 1) + " has input shape [" +
              std::to_string(layer->GetInChannels()) + "," + std::to_string(layer->GetInHeight()) +
              "," + std::to_string(layer->GetInWidth()) + "] but output shape [" +
              std::to_string(layer->GetOutChannels()) + "," +
              std::to_string(layer->GetOutHeight()) + "," + std::to_string(layer->GetOutWidth()) +
              "].");
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
  std::vector<std::unique_ptr<FullConvPCLayerT<T, AccumT>>> layers;

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

  AccumT lastEnergy = AccumT(0);
};
} // namespace Deep
