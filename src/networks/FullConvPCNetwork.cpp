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
    // ir has no device-resident buffer (unlike lr/fl, see irDirty's own
    // comment in FullConvPCLayer.h), so a captured graph keeps replaying
    // whatever ir was at capture time even after SetInferenceRate()
    // changes it. Force a recapture whenever any layer's ir changed since
    // the last successful capture, the same way capturedInferenceSteps
    // already forces one when the step count changes.
    bool irChanged = false;
    for (auto& l : layers)
    {
      if (l->IsIrDirty())
      {
        irChanged = true;
        break;
      }
    }

    if (!graphCaptured || capturedInferenceSteps != inferenceSteps || irChanged)
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

        // settleStep()'s own InvalidateMuCache() calls above only run
        // under useIPC; without it, nothing clears a clamped layer's
        // mu-cache after this recording pass, so the fresh CalculateState()
        // read below would wrongly reuse mu from before this call's own
        // weight update (see ConvPCNetwork::TrainStep()'s identical bug).
        for (auto& l : layers)
        {
          l->InvalidateMuCache();
          l->ClearIrDirty();
        }
      }
      else
      {
        std::cerr << "Graph capture failed, falling back to non-graph execution for this call.\n";
      }
    }

    if (graphCaptured)
    {
      backend->ReplayGraph();
      backend->Synchronize();
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
  // dimension); N is the widest hidden layer's CHANNEL count, and
  // midFanIn is that SAME layer's true fan-in (N*kernelH*kernelW).
  //
  // Middle layers need midFanIn, not N alone: a 3x3 conv's true fan-in is
  // 9x its channel count, so using raw N there understated fan-in by 9x,
  // making `a` too large by sqrt(9)=3 -- confirmed empirically (SHALLOW
  // blew up faster than the full 10-layer network, the opposite of what
  // "too many layers" predicts, but exactly what "a itself is too large"
  // predicts) and mathematically (a flat MUPC_DAMPING of ~0.3 in
  // imagenet.py was accidentally undoing almost exactly this 1/3 factor).
  // The first layer's d_in formula above already got this right; this
  // brings the middle-layer formula in line with it.
  //
  // The extra *L factor below is ONLY correct alongside residual
  // connections: with W ~ N(0,1) (RandomizeWeights() uses limit=1.0f
  // under muPC init -- true unit variance, not Kaiming-scaled), a middle
  // layer's effective gain is a*sqrt(trueFanIn), and *L divides that down
  // to 1/sqrt(L) per layer specifically so that SUMMING L such
  // once-per-layer residual contributions into a shared stream keeps the
  // stream's total variance bounded as L grows. With residual OFF (this
  // architecture's current setting), there's no summed stream to bound --
  // each layer's whole output IS the next layer's whole input, so it
  // should be variance-preserving on its own (gain=1, i.e. a=1/sqrt
  // (trueFanIn), no *L). Leaving *L in anyway, after fixing the fan-in
  // term above, took effective gain from ~0.95/layer (old formula's two
  // errors -- missing kernel-area term AND this same *L -- coincidentally
  // offsetting at L=10) to 1/sqrt(10)~=0.32/layer, compounding to ~6e-5
  // over 9 middle-layer hops -- exactly the near-zero output variance
  // seen in both diagnostics after the fan-in-only fix landed.
  const int L = (int)layers.size() - 1;
  const int H = L - 1;

  if (L < 1)
    return;

  if (useMuPCScaling)
  {
    const float d_in =
        (float)layers[0]->GetInChannels() * layers[0]->GetKernelH() * layers[0]->GetKernelW();

    float N;
    float midFanIn;
    if (H > 0)
    {
      N = 0.0f;
      midFanIn = 1.0f;
      for (int i = 0; i < L - 1; ++i)
      {
        float outCh = (float)layers[i]->GetOutChannels();
        if (outCh > N)
        {
          N = outCh;
          midFanIn = outCh * (float)layers[i]->GetKernelH() * (float)layers[i]->GetKernelW();
        }
      }
    }
    else
    {
      N = (float)layers[L - 1]->GetOutChannels();
      midFanIn = N * (float)layers[L - 1]->GetKernelH() * (float)layers[L - 1]->GetKernelW();
    }

    for (int l = 1; l <= L; ++l)
    {
      float a;
      if (l == 1)
        a = std::pow(d_in, -0.5f);
      else if (l == L)
        a = 1.0f / N;
      else
        a = useResidualConnections ? std::pow(midFanIn * (float)L, -0.5f)
                                    : std::pow(midFanIn, -0.5f);

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
