#include <algorithm>
#include <cmath>
#include <deepity/networks/FullConvPCNetwork.h>
#include <iostream>
#include <pmmintrin.h>
#include <stdexcept>
#include <xmmintrin.h>

namespace Deep
{
FullConvPCNetwork::FullConvPCNetwork(int batchSize, DeviceType device) noexcept
    : device(device)
    , batchSize(batchSize)
{
  backend = CreateBackend(device);
}

void FullConvPCNetwork::AddLayer(int inChannels, int outChannels, int inHeight, int inWidth,
                                 int kernelH, int kernelW, int strideH, int strideW, int padH,
                                 int padW, int terminalSize, float lr, float ir, float fl,
                                 float lmbda, ActivationType aType, ActivationType dType)
{
  auto l = std::make_unique<FullConvPCLayer>(inChannels, outChannels, inHeight, inWidth, kernelH,
                                             kernelW, strideH, strideW, padH, padW, terminalSize,
                                             batchSize, lr, ir, fl, lmbda, aType, dType,
                                             backend.get());

  if (!layers.empty())
  {
    layers.back()->SetLayerAbove(l.get());
    l->SetLayerBelow(layers.back().get());
  }

  layers.push_back(std::move(l));
}

void FullConvPCNetwork::RandomizeWeights(std::mt19937& rng)
{
  for (auto& l : layers)
    l->RandomizeWeights(rng);
}

void FullConvPCNetwork::ResetState() noexcept
{
  for (auto& l : layers)
    l->ResetState();
}

void FullConvPCNetwork::Clamp(const std::vector<float>& input)
{
  layers.front()->ClampState(input);
}

void FullConvPCNetwork::ProjectForward() noexcept
{
  for (size_t i = 0; i + 1 < layers.size(); ++i)
  {
    layers[i]->ComputeMuOnly();

    if (layers[i + 1]->IsClamped())
      continue;

    const float* mu = layers[i]->GetMu();
    float* nextZ = layers[i + 1]->GetBeliefs();
    size_t n = layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

    backend->Copy(nextZ, mu, n);
  }
}

void FullConvPCNetwork::DirectFeedbackUpdate() noexcept
{
  for (size_t i = 0; i < layers.size() - 1; ++i)
    layers[i]->DirectFeedbackUpdate();
}

float FullConvPCNetwork::CalculateTerminalError() noexcept
{
  return GetTerminalLayer()->CalculateState(false);
}

float FullConvPCNetwork::Step(bool needEnergy) noexcept
{
  float e = 0.0f;
  for (auto& l : layers)
    e += l->CalculateState(needEnergy);
  for (auto& l : layers)
    l->UpdateState();
  return needEnergy ? e : 0.0f;
}

void FullConvPCNetwork::EPCStep() noexcept
{
  layers[0]->ComputeMuOnly();
  for (size_t i = 1; i + 1 < layers.size(); ++i)
  {
    layers[i]->ReconstructBelief();
    layers[i]->ComputeMuOnly();
  }

  // Any clamped weight-bearing layer (the input layer) is never touched
  // by ComputeAdjoint() below -- it has no free error to settle -- but
  // its own UpdateWeights() still needs mu holding the derivative, same
  // as every other layer. Unclamped hidden layers get this from
  // ComputeAdjoint() itself and must NOT be touched here first.
  for (size_t i = 0; i + 1 < layers.size(); ++i)
    if (layers[i]->IsClamped())
      layers[i]->EnsureMuHoldsDerivative();

  GetTerminalLayer()->CalculateState(false);

  const float* adjointAbove = GetTerminalLayer()->GetErrors();
  bool firstHop = true;
  for (size_t i = layers.size() - 2; i >= 1; --i)
  {
    layers[i]->ComputeAdjoint(adjointAbove, firstHop ? -1.0f : 1.0f);
    firstHop = false;
    layers[i]->UpdateErrorEPC();
    adjointAbove = layers[i]->GetAdjoint();
  }
}

void FullConvPCNetwork::UpdateWeights() noexcept
{
  for (size_t i = 0; i + 1 < layers.size(); i++)
    layers[i]->UpdateWeights();
}

float FullConvPCNetwork::TrainStep(const std::vector<float>& x, const std::vector<float>& y,
                                   int inferenceSteps)
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

  if (device == DeviceType::DEVICE_GPU)
  {
    if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
    {
      backend->BeginGraphCapture();
      ProjectForward();
      CalculateTerminalError();
      DirectFeedbackUpdate();
      for (int t = 0; t < inferenceSteps; t++)
        settleStep();
      if (!useIPC)
        UpdateWeights();
      bool captureOk = backend->EndGraphCapture();

      if (captureOk)
      {
        graphCaptured = true;
        capturedInferenceSteps = inferenceSteps;
      }
      else
      {
        std::cerr << "Graph capture failed, falling back to non-graph execution for this call.\n";
      }
    }

    if (graphCaptured)
    {
      backend->ReplayGraph();
    }
    else
    {
      ProjectForward();
      CalculateTerminalError();
      DirectFeedbackUpdate();
      for (int t = 0; t < inferenceSteps; t++)
        settleStep();
      if (!useIPC)
        UpdateWeights();
    }
  }
  else
  {
    ProjectForward();
    CalculateTerminalError();
    DirectFeedbackUpdate();
    for (int t = 0; t < inferenceSteps; t++)
      settleStep();
    if (!useIPC)
      UpdateWeights();
  }

  float finalEnergy = 0.0f;
  for (auto& l : layers)
    finalEnergy += l->CalculateState(true);

  GetTerminalLayer()->UnclampState();
  return finalEnergy;
}

std::vector<float> FullConvPCNetwork::Predict(const std::vector<float>& x, int inferenceSteps)
{
  ResetState();
  Clamp(x);
  ProjectForward();

  for (int t = 0; t < inferenceSteps; t++)
  {
    if (useEPC)
      EPCStep();
    else
      Step();
  }

  FullConvPCLayer* terminal = GetTerminalLayer();

  if (terminal->GetCrossEntropy())
  {
    FullConvPCLayer* below = layers[layers.size() - 2].get();
    below->ComputeMuOnly();
    size_t n = below->GetBatchSize() * below->GetOutputSize();
    backend->Copy(terminal->GetBeliefs(), below->GetMu(), n);
  }

  const float* beliefs = terminal->GetBeliefs();
  size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

  std::vector<float> result(count);
  backend->CopyToHost(result.data(), beliefs, count);
  return result;
}

void FullConvPCNetwork::Compile()
{
  if (!layers.empty() &&
      (size_t)layers.back()->GetInChannels() != (size_t)layers.back()->GetTerminalSize())
  {
    throw std::invalid_argument(
        "FullConvPCNetwork::Compile(): terminalSize (" +
        std::to_string(layers.back()->GetTerminalSize()) +
        ") passed to AddLayer() doesn't match the actual terminal layer's inChannels (" +
        std::to_string(layers.back()->GetInChannels()) + ").");
  }

#pragma omp parallel
  {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
  size_t total_floats_needed = 0;
  for (auto& layer : layers)
    total_floats_needed += layer->GetRequiredFloats();

  if (device == DeviceType::DEVICE_CPU)
  {
    cpuArena = std::make_unique<MemoryArena>(total_floats_needed, false);
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
    gpuArena = std::make_unique<DeviceMemoryArena>(backend.get(), total_floats_needed);
    for (auto& layer : layers)
    {
      layer->BindMemory(*gpuArena);
      layer->SetTerminalLayer(layers.back().get());
    }
  }
#endif

  if (!useMuPCScaling && !useResidualConnections)
    return;

  // L = weight-bearing layers (excludes the terminal, outChannels==0).
  // H = L-1 hidden layers. d_in is the true fan-in of the FIRST layer
  // (inChannels*kernelH*kernelW, the conv analog of dense's input
  // dimension); N is the widest hidden layer's CHANNEL count (conv's
  // analog of dense's hidden width -- spatial extent isn't "capacity"
  // the same way channel count is).
  const int L = (int)layers.size() - 1;
  const int H = L - 1;

  if (L < 1)
    return;

  if (useMuPCScaling)
  {
    const float d_in =
        (float)layers[0]->GetInChannels() * layers[0]->GetKernelH() * layers[0]->GetKernelW();

    float N;
    if (H > 0)
    {
      N = 0.0f;
      for (int i = 0; i < L - 1; ++i)
        N = std::max(N, (float)layers[i]->GetOutChannels());
    }
    else
    {
      N = (float)layers[L - 1]->GetOutChannels();
    }

    for (int l = 1; l <= L; ++l)
    {
      float a;
      if (l == 1)
        a = std::pow(d_in, -0.5f);
      else if (l == L)
        a = 1.0f / N;
      else
        a = std::pow(N * (float)L, -0.5f);

      layers[l - 1]->SetMuPCScale(a);
      layers[l - 1]->SetMuPCInit(true);
    }
  }

  if (useResidualConnections)
  {
    for (int l = 2; l <= H; ++l)
    {
      FullConvPCLayer* layer = layers[l - 1].get();
      bool matches = layer->GetInChannels() == layer->GetOutChannels() &&
                     layer->GetInHeight() == layer->GetOutHeight() &&
                     layer->GetInWidth() == layer->GetOutWidth();
      if (!matches)
      {
        throw std::invalid_argument(
            "FullConvPCNetwork::Compile(): residual connections require matching "
            "channels/spatial dims, layer index " +
            std::to_string(l - 1) + " has input shape [" +
            std::to_string(layer->GetInChannels()) + "," + std::to_string(layer->GetInHeight()) +
            "," + std::to_string(layer->GetInWidth()) + "] but output shape [" +
            std::to_string(layer->GetOutChannels()) + "," + std::to_string(layer->GetOutHeight()) +
            "," + std::to_string(layer->GetOutWidth()) + "].");
      }
      layer->SetResidual(true);
    }
  }
}
} // namespace Deep
