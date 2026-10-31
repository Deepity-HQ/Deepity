#include <chrono>
#include <deepity/backend/CPUBackend.h>
#include <deepity/networks/SimplePCNetwork.h>
#include <iostream>
#include <omp.h>
#include <pmmintrin.h>
#include <xmmintrin.h>

namespace Deep
{
SimplePCNetwork::SimplePCNetwork(int batchSize, DeviceType device)
    : device(device)
    , batchSize(batchSize)
{
  backend = CreateBackend(device);
}

void SimplePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                               void (*act)(float*, size_t), void (*dAct)(float*, size_t, bool))
{
  std::unique_ptr<SimplePCLayer> l = std::make_unique<SimplePCLayer>(
      size, nextSize, batchSize, lr, ir, lmbda, act, dAct, backend.get());

  if (!layers.empty())
  {
    layers.back()->SetLayerAbove(l.get());
    l->SetLayerBelow(layers.back().get());
  }
  layers.push_back(std::move(l));
}

void SimplePCNetwork::AddLayer(int size, int nextSize, float lr, float ir, float lmbda,
                               ActivationType aType, ActivationType dType)
{
  std::unique_ptr<SimplePCLayer> l = std::make_unique<SimplePCLayer>(
      size, nextSize, batchSize, lr, ir, lmbda, aType, dType, backend.get());

  if (!layers.empty())
  {
    layers.back()->SetLayerAbove(l.get());
    l->SetLayerBelow(layers.back().get());
  }
  layers.push_back(std::move(l));
}

void SimplePCNetwork::RandomizeWeights(std::mt19937& rng, const char* distribution)
{
  for (auto& l : layers)
    l->RandomizeWeights(rng, distribution);
}

void SimplePCNetwork::ResetState() noexcept
{
  for (auto& l : layers)
    l->ResetState();
}

void SimplePCNetwork::Clamp(const std::vector<float>& input)
{
  layers.front()->ClampState(input);
}

float SimplePCNetwork::CalculateState(bool needEnergy)
{
  float e = 0.0f;
  for (size_t i = 0; i < layers.size(); i++)
    e += layers[i]->CalculateState(needEnergy);
  return needEnergy ? e : 0.0f;
}

void SimplePCNetwork::UpdateState()
{
  for (auto& l : layers)
    l->UpdateState();
}

void SimplePCNetwork::UpdateWeights()
{
  for (size_t i = 0; i + 1 < layers.size(); i++)
    layers[i]->UpdateWeights();
}

float SimplePCNetwork::TrainStep(const std::vector<float>& x, const std::vector<float>& y,
                                 int inferenceSteps)
{
  ResetState();
  Clamp(x);
  GetTerminalLayer()->ClampState(y);

  for (int t = 0; t < inferenceSteps; t++)
  {
    CalculateState(false);
    UpdateState();
  }

  float finalEnergy = CalculateState(true);

  UpdateWeights();
  GetTerminalLayer()->UnclampState();

  return finalEnergy;
}

std::vector<float> SimplePCNetwork::Predict(const std::vector<float>& x, int inferenceSteps)
{
  ResetState();
  Clamp(x);

  for (int t = 0; t < inferenceSteps; t++)
  {
    CalculateState();
    UpdateState();
  }

  SimplePCLayer* terminal = GetTerminalLayer();
  const float* beliefs = terminal->GetBeliefs();
  size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

  std::vector<float> result(count);
  backend->CopyToHost(result.data(), beliefs, count);
  return result;
}

void SimplePCNetwork::ProjectForward() noexcept
{
  for (size_t i = 0; i + 1 < layers.size(); ++i)
  {
    layers[i]->ComputeMuOnly();

    if (layers[i + 1]->IsClamped())
      continue; // never overwrite a clamped layer's real target with a forward guess

    const float* mu = layers[i]->GetMu();
    float* nextZ = layers[i + 1]->GetBeliefs();
    size_t n = layers[i]->GetBatchSize() * layers[i]->GetOutputSize();

    backend->Copy(nextZ, mu, n);
  }
}

float SimplePCNetwork::TrainStepWithProjection(const std::vector<float>& x,
                                               const std::vector<float>& y, int inferenceSteps,
                                               bool computeEnergy)
{
  ResetState();
  Clamp(x);
  GetTerminalLayer()->ClampState(y);

  if (device == DeviceType::DEVICE_GPU)
  {
    if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
    {
      backend->BeginGraphCapture();
      ProjectForward();
      for (int t = 0; t < inferenceSteps; ++t)
      {
        CalculateState(false);
        UpdateState();
      }
      UpdateWeights();
      bool captureOk = backend->EndGraphCapture();

      if (captureOk)
      {
        graphCaptured = true;
        capturedInferenceSteps = inferenceSteps;

        // The recording pass above just ran this settling loop
        // as real C++, leaving muCacheValid=true behind for any
        // clamped layer it touched last; nothing resets that
        // before the fresh energy read below, which would
        // otherwise wrongly reuse mu from before this call's
        // own weight update (see ConvPCNetwork::TrainStep()'s
        // identical bug).
        for (auto& l : layers)
          l->InvalidateMuCache();
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
      for (int t = 0; t < inferenceSteps; ++t)
      {
        CalculateState(false);
        UpdateState();
      }
      UpdateWeights();
    }
  }
  else
  {
    // TEMPORARY profiling instrumentation -- see CPUBackend.h's
    // DebugGetGemmSeconds() declaration for the GEMM-side half of this.
    // Remove this whole block (and that declaration/accumulator) once
    // the wall-clock breakdown investigation is done.
    static double sProjectForwardSec = 0.0, sSettlingSec = 0.0, sUpdateWeightsSec = 0.0;
    static long sCalls = 0;

    auto t0 = std::chrono::steady_clock::now();
    ProjectForward();
    auto t1 = std::chrono::steady_clock::now();
    for (int t = 0; t < inferenceSteps; ++t)
    {
      CalculateState(false);
      UpdateState();
    }
    auto t2 = std::chrono::steady_clock::now();
    UpdateWeights();
    auto t3 = std::chrono::steady_clock::now();

    sProjectForwardSec += std::chrono::duration<double>(t1 - t0).count();
    sSettlingSec += std::chrono::duration<double>(t2 - t1).count();
    sUpdateWeightsSec += std::chrono::duration<double>(t3 - t2).count();
    ++sCalls;

    if (sCalls % 240 == 0)
    {
      double total = sProjectForwardSec + sSettlingSec + sUpdateWeightsSec;
      double gemmSec = DebugGetGemmSeconds();
      double muGemmSec = DebugGetMuGemmSeconds();
      double feedbackGemmSec = DebugGetFeedbackGemmSeconds();
      double weightGradGemmSec = DebugGetWeightGradGemmSeconds();
      double errSec = DebugGetErrorSeconds();
      double muSec = DebugGetMuOnlyElementwiseSeconds();
      double updSec = DebugGetUpdateStateElementwiseSeconds();
      double allGemmSec = gemmSec + muGemmSec + feedbackGemmSec + weightGradGemmSec;
      std::cerr << "[PROFILE] calls=" << sCalls << " total=" << total << "s"
                << " ProjectForward=" << sProjectForwardSec << "s ("
                << (100.0 * sProjectForwardSec / total) << "%)"
                << " Settling(T=" << inferenceSteps << ")=" << sSettlingSec << "s ("
                << (100.0 * sSettlingSec / total) << "%)"
                << " UpdateWeights=" << sUpdateWeightsSec << "s ("
                << (100.0 * sUpdateWeightsSec / total) << "%)"
                << " | GEMM(cblas, remaining)=" << gemmSec << "s (" << (100.0 * gemmSec / total)
                << "% of total, " << DebugGetGemmCalls() << " calls)"
                << " GEMM(oneDNN mu)=" << muGemmSec << "s (" << (100.0 * muGemmSec / total) << "%)"
                << " GEMM(oneDNN feedback)=" << feedbackGemmSec << "s ("
                << (100.0 * feedbackGemmSec / total) << "%)"
                << " GEMM(oneDNN weightGrad)=" << weightGradGemmSec << "s ("
                << (100.0 * weightGradGemmSec / total) << "%)"
                << " allGEMM=" << allGemmSec << "s (" << (100.0 * allGemmSec / total) << "%)"
                << " non-GEMM=" << (total - allGemmSec) << "s ("
                << (100.0 * (total - allGemmSec) / total) << "%)"
                << " [[error=" << errSec << "s (" << (100.0 * errSec / total) << "%)"
                << " muOnlyElementwise(activation+cache)=" << muSec << "s ("
                << (100.0 * muSec / total) << "%)"
                << " updateStateElementwise=" << updSec << "s (" << (100.0 * updSec / total)
                << "%) sum=" << (errSec + muSec + updSec) << "s ("
                << (100.0 * (errSec + muSec + updSec) / total) << "%)]]\n";
    }
  }

  float finalEnergy = CalculateState(computeEnergy);
  GetTerminalLayer()->UnclampState();

  return finalEnergy;
}

std::vector<float> SimplePCNetwork::PredictWithProjection(const std::vector<float>& x,
                                                          int inferenceSteps)
{
  ResetState();
  Clamp(x);

  auto settleAndProject = [&]()
  {
    ProjectForward();
    for (int t = 0; t < inferenceSteps; t++)
    {
      CalculateState(false);
      UpdateState();
    }
  };

  if (device == DeviceType::DEVICE_GPU)
  {
    if (!predictGraphCaptured || capturedPredictInferenceSteps != inferenceSteps)
    {
      backend->BeginGraphCapture(kPredictGraphSlot);
      settleAndProject();
      bool captureOk = backend->EndGraphCapture(kPredictGraphSlot);

      if (captureOk)
      {
        predictGraphCaptured = true;
        capturedPredictInferenceSteps = inferenceSteps;
      }
      else
        std::cerr << "PredictWithProjection() graph capture failed, falling back to non-graph "
                     "execution for this call.\n";
    }

    if (predictGraphCaptured)
    {
      backend->ReplayGraph(kPredictGraphSlot);
      // cudaGraphLaunch() only enqueues the replay -- the CopyToHost()
      // below needs it to have actually finished.
      backend->Synchronize();
    }
    else
    {
      settleAndProject();
    }
  }
  else
  {
    settleAndProject();
  }

  SimplePCLayer* terminal = GetTerminalLayer();
  const float* beliefs = terminal->GetBeliefs();
  size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

  std::vector<float> result(count);
  backend->CopyToHost(result.data(), beliefs, count);
  return result;
}

void SimplePCNetwork::SetMuCacheThreshold(float threshold) noexcept
{
  for (auto& l : layers)
    l->SetMuCacheThreshold(threshold);
}

void SimplePCNetwork::Compile()
{
#pragma omp parallel
  {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }
  size_t total_floats_needed = 0;
  for (auto& layer : layers)
    total_floats_needed += layer->GetRequiredFloats();

  backend->PrepareForBatchSize(batchSize);

  if (device == DeviceType::DEVICE_CPU)
  {
    cpuArena = std::make_unique<MemoryArena>(total_floats_needed, true);
    for (auto& layer : layers)
      layer->BindMemory(*cpuArena);
  }
#if defined(DEEPITY_USE_CUDA)
  else
  {
    gpuArena = std::make_unique<DeviceMemoryArena>(backend.get(), total_floats_needed);
    for (auto& layer : layers)
      layer->BindMemory(*gpuArena);
  }
#endif
}
} // namespace Deep