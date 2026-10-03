#include <algorithm>
#include <cmath>
#include <deepity/networks/FullPCNetwork.h>
#include <iostream>
#include <pmmintrin.h>
#include <stdexcept>
#include <xmmintrin.h>

namespace Deep
{
FullPCNetwork::FullPCNetwork(int batchSize, DeviceType device) noexcept
    : device(device)
    , batchSize(batchSize)
{
  backend = CreateBackend(device);
}

void FullPCNetwork::AddLayer(size_t size, size_t nextSize, size_t terminalSize, float lr, float ir,
                             float fl, float lmbda, ActivationType aType, ActivationType dType)
{
  std::unique_ptr<FullPCLayer> l = std::make_unique<FullPCLayer>(
      size, nextSize, terminalSize, batchSize, lr, ir, fl, lmbda, aType, dType, backend.get());

  if (!layers.empty())
  {
    layers.back()->SetLayerAbove(l.get());
    l->SetLayerBelow(layers.back().get());
  }

  layers.push_back(std::move(l));
}

void FullPCNetwork::RandomizeWeights(std::mt19937& rng)
{
  for (auto& l : layers)
    l->RandomizeWeights(rng);
}

void FullPCNetwork::ResetState() noexcept
{
  for (auto& l : layers)
    l->ResetState();
}

void FullPCNetwork::Clamp(const std::vector<float>& input)
{
  layers.front()->ClampState(input);
}

void FullPCNetwork::ProjectForward() noexcept
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

void FullPCNetwork::DirectFeedbackUpdate() noexcept
{
  for (size_t i = 0; i < layers.size() - 1; ++i)
    layers[i]->DirectFeedbackUpdate();
}

float FullPCNetwork::CalculateTerminalError() noexcept
{
  return GetTerminalLayer()->CalculateState(false);
}

float FullPCNetwork::Step(bool needEnergy) noexcept
{
  float e = 0.0f;
  for (auto& l : layers)
    e += l->CalculateState(needEnergy);
  for (auto& l : layers)
    l->UpdateState();
  return needEnergy ? e : 0.0f;
}

void FullPCNetwork::EPCStep() noexcept
{
  // Forward sweep, bottom to top: reconstruct each hidden layer's belief
  // from the layer below's just-computed prediction (z_i := mu_{i-1}+e_i,
  // the paper's s_i := ŝ_i+ε_i), then compute this layer's own mu for the
  // layer above. The input layer (index 0) is clamped, its mu is already
  // valid from ClampState()/InvalidateMuCache(); the terminal's error
  // (against the clamped target, Gaussian or cross-entropy) IS the loss
  // gradient the backward sweep starts from, unchanged from every other
  // settling mode.
  layers[0]->ComputeMuOnly();
  for (size_t i = 1; i + 1 < layers.size(); ++i)
  {
    layers[i]->ReconstructBelief();
    layers[i]->ComputeMuOnly();
  }
  GetTerminalLayer()->CalculateState(false);

  // Backward sweep, top to bottom: the terminal's error is the topmost
  // hidden layer's incoming signal; each layer below then reads the one
  // above's freshly computed `adjoint`, chaining the FULL gradient
  // through every layer in one pass (not the one-hop message every
  // other settling mode uses), then updates its own error in place.
  const float* adjointAbove = GetTerminalLayer()->GetErrors();
  bool firstHop = true; // see ComputeAdjoint()'s adjointAboveScale docs
  for (size_t i = layers.size() - 2; i >= 1; --i)
  {
    layers[i]->ComputeAdjoint(adjointAbove, firstHop ? -1.0f : 1.0f);
    firstHop = false;
    layers[i]->UpdateErrorEPC();
    adjointAbove = layers[i]->GetAdjoint();
  }
}

void FullPCNetwork::UpdateWeights() noexcept
{
  for (size_t i = 0; i + 1 < layers.size(); i++)
    layers[i]->UpdateWeights();
}

float FullPCNetwork::TrainStep(const std::vector<float>& x, const std::vector<float>& y,
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

        // settleStep()'s own InvalidateMuCache() calls above only run
        // under useIPC; without it, nothing clears a clamped layer's
        // mu-cache after this recording pass, so the fresh CalculateState()
        // read below would wrongly reuse mu from before this call's own
        // weight update (see ConvPCNetwork::TrainStep()'s identical bug).
        for (auto& l : layers)
          l->InvalidateMuCache();
      }
      else
        std::cerr << "Graph capture failed, falling back to non-graph execution for this call.\n";
    }

    if (graphCaptured)
    {
      backend->ReplayGraph();
      // cudaGraphLaunch() only enqueues the replay; see
      // ConvPCNetwork::TrainStep()'s identical comment for why this
      // matters even though same-stream ordering should already cover it.
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

std::vector<float> FullPCNetwork::Predict(const std::vector<float>& x, int inferenceSteps)
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

  FullPCLayer* terminal = GetTerminalLayer();

  if (terminal->GetCrossEntropy())
  {
    FullPCLayer* below = layers[layers.size() - 2].get();
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

void FullPCNetwork::Compile()
{
  // terminalSize is supplied on every AddLayer() call rather than
  // inferred, since the true terminal layer isn't known until the whole
  // network is assembled, validate it now against the actual last
  // layer's size. A mismatch here previously went uncaught: the DFA
  // feedback GEMM in DirectFeedbackUpdate() would read past the real
  // (smaller) terminal error buffer using the wrong terminalSize as its
  // GEMM dimension, corrupting W with garbage/inf on the very first
  // TrainStep(), confirmed by reproducing exactly this with a
  // deliberately wrong terminalSize.
  if (!layers.empty() && layers.back()->GetInputSize() != layers.back()->GetTerminalSize())
  {
    throw std::invalid_argument(
        "FullPCNetwork::Compile(): terminalSize (" +
        std::to_string(layers.back()->GetTerminalSize()) +
        ") passed to AddLayer() doesn't match the actual terminal layer's size (" +
        std::to_string(layers.back()->GetInputSize()) + ").");
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

  // muPC scaling / residual connections: OFF by default (a=1.0,
  // useResidual=false on every layer already, from FullPCLayer's
  // own constructor defaults), only touch anything if the
  // corresponding network-level toggle was actually set.
  if (!useMuPCScaling && !useResidualConnections)
    return;

  // L = number of real, weight-bearing layers (excludes the
  // terminal-only layer, layers.back(), whose nextSize == 0).
  // H = number of hidden z's (L-1). Mirrors the muPC prototype's
  // own L/H definitions exactly, verified by hand against its
  // [784,256,256,10] example before writing this.
  const int L = (int)layers.size() - 1;
  const int H = L - 1;

  if (L < 1)
    return;

  if (useMuPCScaling)
  {
    const float d_in = (float)layers[0]->GetInputSize();

    float N;
    if (H > 0)
    {
      N = 0.0f;
      for (int i = 0; i < L - 1; ++i)
        N = std::max(N, (float)layers[i]->GetOutputSize());
    }
    else
    {
      N = (float)layers[L - 1]->GetOutputSize();
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
      layers[l - 1]->SetMuPCInit(true); // Table 1's other change: b_l = 1
    }
  }

  if (useResidualConnections)
  {
    for (int l = 2; l <= H; ++l)
    {
      FullPCLayer* layer = layers[l - 1].get();
      if (layer->GetInputSize() != layer->GetOutputSize())
      {
        throw std::invalid_argument("FullPCNetwork::Compile(): residual connections require "
                                    "matching width, layer index " +
                                    std::to_string(l - 1) + " has input size " +
                                    std::to_string(layer->GetInputSize()) + " but output size " +
                                    std::to_string(layer->GetOutputSize()) + ".");
      }
      layer->SetResidual(true);
    }
  }
}
} // namespace Deep
