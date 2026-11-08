#include <deepity/networks/DirectKPPCNetwork.h>
#include <iostream>
#include <pmmintrin.h>
#include <stdexcept>
#include <xmmintrin.h>

namespace Deep
{
DirectKPPCNetwork::DirectKPPCNetwork(int batchSize, DeviceType device)
    : device(device)
    , batchSize(batchSize)
{
  backend = CreateBackend(device);
}

void DirectKPPCNetwork::AddLayer(size_t size, size_t nextSize, size_t terminalSize, float lr,
                                 float ir, float fl, float lmbda, ActivationType aType,
                                 ActivationType dType)
{
  std::unique_ptr<DirectKPPCLayer> l = std::make_unique<DirectKPPCLayer>(
      size, nextSize, terminalSize, batchSize, lr, ir, fl, lmbda, aType, dType, backend.get());

  if (!layers.empty())
  {
    layers.back()->SetLayerAbove(l.get());
    l->SetLayerBelow(layers.back().get());
  }

  layers.push_back(std::move(l));
}

void DirectKPPCNetwork::RandomizeWeights(std::mt19937& rng, const char* distribution)
{
  for (auto& l : layers)
    l->RandomizeWeights(rng, distribution);
}

void DirectKPPCNetwork::ResetState() noexcept
{
  if (!UseBatchedPath())
  {
    for (auto& l : layers)
      l->ResetState();
    return;
  }

  layers.front()->ResetState();
  // Unlike FinalEnergyBatchedRun(), zeroing has no cross-layer
  // dependency, so all L run layers (including batchedRunStart's own)
  // go in this one call instead of needing the run's first layer
  // handled separately.
  const size_t own = (size_t)batchSize * batchedRunWidth;
  backend->Zero(flatZ, own * batchedRunCount);
  for (size_t i = batchedRunStart + batchedRunCount; i < layers.size(); ++i)
    layers[i]->ResetState();
}

void DirectKPPCNetwork::Clamp(const std::vector<float>& input)
{
  layers.front()->ClampState(input);
}

bool DirectKPPCNetwork::UseBatchedPath() const noexcept
{
  if (batchedRunCount == 0)
    return false;

  // Structural eligibility (width/activation/terminalSize) was already
  // checked once in Compile(), since none of those can change after
  // construction. lr/fl/lmbda/ir and the optimizer selection DO have
  // per-layer setters callable any time (an LR schedule, say), so they
  // get checked fresh here, every call.
  DirectKPPCLayer* first = layers[batchedRunStart].get();
  OptimizerType opt0 = first->GetOptimizer();
  OptimizerType optPsi0 = first->GetPsiOptimizer();
  float lr0 = first->GetLearningRate();
  float fl0 = first->GetFeedbackRate();
  float lmbda0 = first->GetLambda();
  float ir0 = first->GetInferenceRate();
  if (opt0 != OptimizerType::SGD || optPsi0 != OptimizerType::SGD)
    return false;

  for (size_t l = 1; l < batchedRunCount; ++l)
  {
    DirectKPPCLayer* layer = layers[batchedRunStart + l].get();
    if (layer->GetOptimizer() != opt0 || layer->GetPsiOptimizer() != optPsi0 ||
        layer->GetLearningRate() != lr0 || layer->GetFeedbackRate() != fl0 ||
        layer->GetLambda() != lmbda0 || layer->GetInferenceRate() != ir0)
      return false;
  }
  return true;
}

void DirectKPPCNetwork::ComputeMuBatchedRun() noexcept
{
  const size_t L = batchedRunCount;
  const size_t H = batchedRunWidth;
  const size_t own = (size_t)batchSize * H;
  const size_t w = H * H;

  // mu[l] = Activation(z[l] @ W[l]^T + b[l]) for every run layer -- a
  // pure function of z/W/b, each fixed since the last UpdateState()
  // call, so this has no ordering dependency on anything else computed
  // in the same phase (see the header's own comment on this).
  for (size_t l = 0; l < L; ++l)
  {
    batchedA[l] = flatZ + l * own;
    batchedB[l] = flatW + l * w;
    batchedC[l] = flatMu + l * own;
  }
  backend->MatMulBatched(false, true, (int)batchSize, (int)H, (int)H, 1.0f, batchedA.data(), (int)H,
                         batchedB.data(), (int)H, 0.0f, batchedC.data(), (int)H, (int)L);

  for (size_t l = 0; l < L; ++l)
    backend->AddBiasBroadcast(flatMu + l * own, flatB + l * H, (size_t)batchSize, H);

  backend->Copy(flatPreact, flatMu, own * L);
  backend->Activation(layers[batchedRunStart]->GetActivationType(), flatMu, own * L);
}

void DirectKPPCNetwork::ProjectForwardBatchedRun() noexcept
{
  const size_t L = batchedRunCount;
  const size_t own = (size_t)batchSize * batchedRunWidth;

  ComputeMuBatchedRun();

  // z[l+1] = mu[l] for every INTERIOR pair (both endpoints inside the
  // run); one shifted flat copy covers all L-1 of them at once, since
  // both ranges are contiguous multiples of `own`. The two pairs that
  // cross the run's boundary (layers.front()->run, run->layers.back())
  // are handled by the caller, via the unchanged per-layer copy.
  if (L > 1)
    backend->Copy(flatZ + own, flatMu, own * (L - 1));
}

void DirectKPPCNetwork::DirectFeedbackUpdateBatchedRun() noexcept
{
  const size_t L = batchedRunCount;
  const size_t H = batchedRunWidth;
  const size_t own = (size_t)batchSize * H;
  const size_t w = H * H;
  const size_t PS = layers[batchedRunStart]->GetPsiStride();
  const size_t direct = H * PS;

  if (L < 2)
    return; // nothing left once the run's own last layer is excluded below

  // The run's own last layer is excluded here -- its layerAbove is
  // outside the run (the terminal layer directly, or a trailing
  // non-run middle layer), handled by the caller's own per-layer call
  // instead, which correctly no-ops for a null Psi (terminal case, see
  // BindMemory()'s nextSize>0 guard) or does a real update otherwise.
  const size_t M = L - 1;

  // When Compile() padded Psi (see its own comment), K/lda/ldb need
  // the padded width PS, not terminalSize, to match Psi's real row
  // stride -- and the A operand becomes the zero-padded copy of
  // terminalE, refreshed fresh since terminalE's value differs at
  // every TrainStep() call site that reads it. A no-op, using
  // terminalE directly, when Compile() didn't pad (PS==terminalSize).
  RefreshPaddedTerminalE();
  const float* terminalE = layers.back()->GetErrors();
  const float* A_terminalE = (flatTerminalEPadded != nullptr) ? flatTerminalEPadded : terminalE;

  // proj[l] = terminalLayer->GetErrors() @ layerAbove->Psi^T, where
  // layerAbove for run layer l is run layer l+1 -- Psi shifted by one
  // layer, same flat-buffer-plus-offset trick as everywhere else here.
  for (size_t l = 0; l < M; ++l)
  {
    batchedA[l] = A_terminalE;
    batchedB[l] = flatPsi + (l + 1) * direct;
    batchedC[l] = flatProj + l * own;
  }
  backend->MatMulBatched(false, true, (int)batchSize, (int)H, (int)PS, 1.0f, batchedA.data(), (int)PS,
                         batchedB.data(), (int)PS, 0.0f, batchedC.data(), (int)H, (int)M);

  backend->ActivationDerivativeInto(ToDerivativeType(layers[batchedRunStart]->GetActivationType()),
                                    flatDerivScratch, flatPreact, own * M);
  backend->MultiplyInto(flatProj, flatProj, flatDerivScratch, own * M);

  const float fl = layers[batchedRunStart]->GetFeedbackRate();
  for (size_t l = 0; l < M; ++l)
  {
    batchedA[l] = flatProj + l * own;
    batchedB[l] = flatZ + l * own;
    batchedC[l] = flatW + l * w;
  }
  backend->MatMulBatched(true, false, (int)H, (int)H, (int)batchSize, fl / (float)batchSize,
                         batchedA.data(), (int)H, batchedB.data(), (int)H, 1.0f, batchedC.data(),
                         (int)H, (int)M);
}

void DirectKPPCNetwork::StepBatchedRunCalculateState() noexcept
{
  const size_t L = batchedRunCount;
  const size_t own = (size_t)batchSize * batchedRunWidth;

  ComputeMuBatchedRun();

  // e[l] = z[l] - mu_below[l]. mu_below[0] is layers.front()'s own mu
  // (outside the flat buffer; the caller refreshes it just before this
  // call), mu_below[1..L-1] is flatMu[0..L-2] -- the same flat buffer,
  // shifted by one layer, same shape as ProjectForwardBatchedRun()'s
  // own shifted copy above.
  backend->ComputeError(flatE, flatZ, layers.front()->GetMu(), own);
  if (L > 1)
    backend->ComputeError(flatE + own, flatZ + own, flatMu, own * (L - 1));
}

void DirectKPPCNetwork::StepBatchedRunUpdateState() noexcept
{
  const size_t L = batchedRunCount;
  const size_t H = batchedRunWidth;
  const size_t own = (size_t)batchSize * H;
  const size_t w = H * H;

  if (L < 2)
    return; // the run's own last layer is handled by the caller's own unchanged call

  const size_t M = L - 1;

  backend->ActivationDerivativeInto(ToDerivativeType(layers[batchedRunStart]->GetActivationType()),
                                    flatDerivScratch, flatPreact, own * M);
  // feedbackError[l] = e_above[l] * derivScratch[l]; e_above[l] is run
  // layer (l+1)'s error -- the same flat buffer shifted by one layer.
  backend->MultiplyInto(flatFeedbackError, flatE + own, flatDerivScratch, own * M);

  for (size_t l = 0; l < M; ++l)
  {
    batchedA[l] = flatFeedbackError + l * own;
    batchedB[l] = flatW + l * w;
    batchedC[l] = flatFeedbackScratch + l * own;
  }
  backend->MatMulBatched(false, false, (int)batchSize, (int)H, (int)H, 1.0f, batchedA.data(), (int)H,
                         batchedB.data(), (int)H, 0.0f, batchedC.data(), (int)H, (int)M);

  backend->AxpyInto(flatFeedbackScratch, flatE, own * M, -1.0f);
  backend->AxpyInto(flatZ, flatFeedbackScratch, own * M, layers[batchedRunStart]->GetInferenceRate());
}

void DirectKPPCNetwork::UpdateWeightsBatchedRun() noexcept
{
  const size_t L = batchedRunCount;
  const size_t H = batchedRunWidth;
  const size_t own = (size_t)batchSize * H;
  const size_t w = H * H;
  const size_t PS = layers[batchedRunStart]->GetPsiStride();
  const size_t direct = H * PS;

  if (L < 2)
    return; // the run's own last layer is handled by the caller's own unchanged call

  const size_t M = L - 1;
  const float lr = layers[batchedRunStart]->GetLearningRate();
  const float lmbda = layers[batchedRunStart]->GetLambda();
  const float fl = layers[batchedRunStart]->GetFeedbackRate();

  // local_grad[l] = e_above[l] * ActivationDerivative(preact[l]);
  // e_above[l] is run layer (l+1)'s error, shifted by one layer.
  backend->ActivationDerivativeInto(ToDerivativeType(layers[batchedRunStart]->GetActivationType()),
                                    flatDerivScratch, flatPreact, own * M);
  backend->MultiplyInto(flatFeedbackError, flatE + own, flatDerivScratch, own * M);

  if (lmbda > 0.0f)
    backend->Scale(flatW, w * M, 1.0f - lmbda);

  for (size_t l = 0; l < M; ++l)
  {
    batchedA[l] = flatFeedbackError + l * own;
    batchedB[l] = flatZ + l * own;
    batchedC[l] = flatW + l * w;
  }
  backend->MatMulBatched(true, false, (int)H, (int)H, (int)batchSize, lr / (float)batchSize,
                         batchedA.data(), (int)H, batchedB.data(), (int)H, 1.0f, batchedC.data(),
                         (int)H, (int)M);

  // SumRows() reduces over `batchSize` rows per call; each run layer's
  // M rows are its OWN batch, so this needs M separate reductions, not
  // one flat call. biasGradScratch is small (H floats per layer) and
  // this is O(M) cheap dispatch either way.
  for (size_t l = 0; l < M; ++l)
    backend->SumRows(flatBiasGradScratch + l * H, flatFeedbackError + l * own, (size_t)batchSize, H);
  backend->AxpyInto(flatB, flatBiasGradScratch, H * M, lr / (float)batchSize);

  // Psi += fl/batchSize * z^T @ terminalLayer->GetErrors() -- the SAME
  // terminal errors for every run layer, so A is the same pointer
  // repeated across the whole batch (MatMulBatched doesn't require
  // distinct pointers). Scoped to M (not L): the run's own last layer
  // gets this same accumulate-onto-Psi update from its own individual
  // UpdateWeights() call in the caller, so including it here too would
  // double its Psi gradient every TrainStep() call.
  //
  // N/ldb/ldc use PS (Psi's real, possibly-padded row width), not
  // terminalSize -- B needs the zero-padded copy of terminalE so the
  // GEMM's extra output columns see zero inputs (writing zero into
  // Psi's own padding, same safe-forever argument as every other use
  // of flatTerminalEPadded here); C writes straight into Psi's own
  // buffer at its real stride, no write-back copy needed. A no-op,
  // identical to the unpadded case, when Compile() didn't pad
  // (PS==terminalSize, flatTerminalEPadded null).
  RefreshPaddedTerminalE();
  const float* terminalE = layers.back()->GetErrors();
  const float* B_terminalE = (flatTerminalEPadded != nullptr) ? flatTerminalEPadded : terminalE;
  for (size_t l = 0; l < M; ++l)
  {
    batchedA[l] = flatZ + l * own;
    batchedB[l] = B_terminalE;
    batchedC[l] = flatPsi + l * direct;
  }
  backend->MatMulBatched(true, false, (int)H, (int)PS, (int)batchSize, fl / (float)batchSize,
                         batchedA.data(), (int)H, batchedB.data(), (int)PS, 1.0f, batchedC.data(),
                         (int)PS, (int)M);
}

float DirectKPPCNetwork::FinalEnergyBatchedRun() noexcept
{
  const size_t L = batchedRunCount;
  const size_t own = (size_t)batchSize * batchedRunWidth;

  if (L < 2)
    return 0.0f; // the run's own single layer is handled by the caller's own unchanged call

  // mu[l] = Activation(z[l]@W[l]+b[l]) for every run layer, from z's
  // already-final value (CalculateState() never writes z) -- safe to
  // batch for the same reason ProjectForwardBatchedRun() is: nothing
  // computed in this call depends on another layer's freshly-written
  // value from within the SAME call.
  ComputeMuBatchedRun();

  // Run layer (run-local index) l's error needs run layer (l-1)'s mu,
  // for l=1..L-1 -- one shifted ComputeErrorAndEnergy() call covers
  // all L-1 of them at once (z shifted forward by one layer against
  // mu, exactly matching ComputeErrorAndEnergy's own e=z-mu, summed
  // into one scalar). The run's first layer needs layers.front()'s mu
  // (outside the run) and is left to the caller's own per-layer call.
  return backend->ComputeErrorAndEnergy(flatE + own, flatZ + own, flatMu, own * (L - 1));
}

void DirectKPPCNetwork::RefreshPaddedTerminalE() noexcept
{
  if (flatTerminalEPadded == nullptr)
    return; // Compile() didn't pad Psi -- callers use terminalE directly instead

  const size_t TS = layers[batchedRunStart]->GetTerminalSize();
  const size_t PS = layers[batchedRunStart]->GetPsiStride();
  const float* terminalE = layers.back()->GetErrors();
  backend->Copy2D(flatTerminalEPadded, PS, terminalE, TS, (size_t)batchSize, TS);
}

void DirectKPPCNetwork::ProjectForward() noexcept
{
  auto pair = [&](size_t i)
  {
    layers[i]->ComputeMuOnly();

    // Skip a layer that's already clamped (the terminal layer,
    // once ClampState(y) has run), don't overwrite its real
    // target with a forward-projected guess. Requires
    // ClampState(y) to run before this call.
    if (layers[i + 1]->IsClamped())
      return;

    backend->Copy(layers[i + 1]->GetBeliefs(), layers[i]->GetMu(),
                  layers[i]->GetBatchSize() * layers[i]->GetOutputSize());
  };

  if (!UseBatchedPath())
  {
    for (size_t i = 0; i + 1 < layers.size(); ++i)
      pair(i);
    return;
  }

  // The run always starts at index 1 (just past layers.front()), but
  // does NOT necessarily reach all the way to layers.back() -- when the
  // run's own width differs from terminalSize (the common case), the
  // one middle layer right before the terminal has nextSize==
  // terminalSize, not the run's width, and Compile()'s detection stops
  // the run one layer short of it (see its own comment). That layer,
  // and the terminal, are a "suffix" of ordinary non-run layers,
  // handled below exactly like the unbatched loop above -- this is
  // just as correct whether the suffix is one layer (straight to the
  // terminal) or more.
  size_t suffixStart = batchedRunStart + batchedRunCount;

  for (size_t i = 0; i < batchedRunStart; ++i)
    pair(i); // just layers.front(), since the run always starts at index 1

  ProjectForwardBatchedRun();
  pair(suffixStart - 1); // the run's own last layer -- entry already
                          // done above; this computes its mu (redundant
                          // with ProjectForwardBatchedRun()'s own mu
                          // pass, harmless) and copies it into whatever
                          // comes next.

  for (size_t i = suffixStart; i + 1 < layers.size(); ++i)
    pair(i);
}

void DirectKPPCNetwork::DirectFeedbackUpdate() noexcept
{
  if (!UseBatchedPath())
  {
    for (size_t i = 0; i < layers.size() - 1; ++i)
      layers[i]->DirectFeedbackUpdate();
    return;
  }

  size_t suffixStart = batchedRunStart + batchedRunCount;

  layers.front()->DirectFeedbackUpdate(); // reads the run's first layer's Psi, unaffected by batching
  DirectFeedbackUpdateBatchedRun();        // batches the run's layers [0, L-2]

  // The run's own last layer's DirectFeedbackUpdate() reads
  // layerAbove->GetDirectFeedbackWeights(), where layerAbove is
  // layers[suffixStart] -- the terminal layer (null Psi there, so this
  // no-ops, same as DirectFeedbackUpdateBatchedRun()'s own excluded
  // case) or a real trailing non-run middle layer (a real Psi, so this
  // is a genuine per-layer update, not batched).
  layers[suffixStart - 1]->DirectFeedbackUpdate();
  for (size_t i = suffixStart; i < layers.size() - 1; ++i)
    layers[i]->DirectFeedbackUpdate();
}

float DirectKPPCNetwork::CalculateTerminalError() noexcept
{
  return GetTerminalLayer()->CalculateState(false);
}

float DirectKPPCNetwork::Step(bool needEnergy) noexcept
{
  if (needEnergy || !UseBatchedPath())
  {
    float e = 0.0f;
    for (auto& l : layers)
      e += l->CalculateState(needEnergy);
    for (auto& l : layers)
      l->UpdateState();
    return needEnergy ? e : 0.0f;
  }

  size_t suffixStart = batchedRunStart + batchedRunCount;

  // CalculateState phase: prefix (layers.front()), the run (batched,
  // self-contained -- StepBatchedRunCalculateState() computes mu THEN e
  // for every run layer itself, so it needs nothing sequenced around
  // it besides layers.front()'s own mu being fresh first), then the
  // suffix (zero or more trailing non-run middle layers, then the
  // terminal). Every CalculateState() here must finish before any
  // UpdateState() below runs, matching the unbatched branch's own
  // two-separate-loops structure.
  layers.front()->CalculateState(false);
  StepBatchedRunCalculateState();
  for (size_t i = suffixStart; i < layers.size(); ++i)
    layers[i]->CalculateState(false);

  layers.front()->UpdateState(); // always clamped during TrainStep()/Predict(), no-op
  StepBatchedRunUpdateState();   // z-update for the run's layers [0, L-2]
  layers[suffixStart - 1]->UpdateState(); // the run's own last layer
  for (size_t i = suffixStart; i < layers.size(); ++i)
    layers[i]->UpdateState();

  return 0.0f;
}

void DirectKPPCNetwork::UpdateWeights() noexcept
{
  // Step()'s last CalculateState() call left preact one settling
  // iteration behind the z that UpdateState() just produced from it --
  // refresh it here so the gradient below is computed "from the settled
  // state" (see the class doc comment's phase 3), not from the z/preact
  // pairing one step before it.
  if (!UseBatchedPath())
  {
    for (size_t i = 0; i + 1 < layers.size(); ++i)
      layers[i]->ComputeMuOnly();
    for (size_t i = 0; i + 1 < layers.size(); i++)
      layers[i]->UpdateWeights();
    return;
  }

  size_t suffixStart = batchedRunStart + batchedRunCount;

  layers.front()->ComputeMuOnly();
  ComputeMuBatchedRun(); // mu is a pure function of z/W/b -- the three
                         // segments here have no ordering dependency on
                         // each other, unlike CalculateState()'s own e.
  for (size_t i = suffixStart; i + 1 < layers.size(); ++i)
    layers[i]->ComputeMuOnly();

  layers.front()->UpdateWeights();
  UpdateWeightsBatchedRun();               // batches the run's layers [0, L-2]
  layers[suffixStart - 1]->UpdateWeights(); // the run's own last layer
  for (size_t i = suffixStart; i + 1 < layers.size(); ++i)
    layers[i]->UpdateWeights();
}

float DirectKPPCNetwork::TrainStep(const std::vector<float>& x, const std::vector<float>& y,
                                   int inferenceSteps)
{
  ResetState();
  Clamp(x);
  // Moved BEFORE ProjectForward(), required for ProjectForward's
  // new IsClamped() guard to actually protect the terminal layer,
  // and required so ProjectForward() can safely move inside the
  // captured region below.
  GetTerminalLayer()->ClampState(y);

  if (device == DeviceType::DEVICE_GPU)
  {
    if (!graphCaptured || capturedInferenceSteps != inferenceSteps)
    {
      backend->BeginGraphCapture();
      ProjectForward(); // now inside capture, see note above
      CalculateTerminalError();
      DirectFeedbackUpdate();
      for (int t = 0; t < inferenceSteps; t++)
        Step(false);
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
      ProjectForward();
      CalculateTerminalError();
      DirectFeedbackUpdate();
      for (int t = 0; t < inferenceSteps; t++)
        Step(false);
      UpdateWeights();
    }
  }
  else
  {
    ProjectForward();
    CalculateTerminalError();
    DirectFeedbackUpdate();
    for (int t = 0; t < inferenceSteps; t++)
      Step(false);
    UpdateWeights();
  }

  float finalEnergy = 0.0f;
  if (!UseBatchedPath())
  {
    for (auto& l : layers)
      finalEnergy += l->CalculateState(true);
  }
  else
  {
    finalEnergy += layers.front()->CalculateState(true);
    // The run's first layer needs layers.front()'s mu (outside the
    // run), so it's left to its own per-layer call; the rest of the
    // run is covered by FinalEnergyBatchedRun() below.
    finalEnergy += layers[batchedRunStart]->CalculateState(true);
    finalEnergy += FinalEnergyBatchedRun();
    for (size_t i = batchedRunStart + batchedRunCount; i < layers.size(); ++i)
      finalEnergy += layers[i]->CalculateState(true);
  }

  GetTerminalLayer()->UnclampState();
  return finalEnergy;
}

std::vector<float> DirectKPPCNetwork::Predict(const std::vector<float>& x, int inferenceSteps)
{
  ResetState();
  Clamp(x);

  // Pure forward/settling work, reading only the buffers Clamp() just
  // filled -- no DirectFeedbackUpdate(), no UpdateWeights(), no
  // terminal ClampState(), unlike TrainStep(). Captured into its OWN
  // graph slot (kPredictGraphSlot), separate from TrainStep()'s --
  // see FullPCNetwork::Predict()'s identical comment for why.
  auto settleAndProject = [&]()
  {
    ProjectForward();
    for (int t = 0; t < inferenceSteps; t++)
      // false: needEnergy's ComputeErrorAndEnergy() does a synchronous
      // cudaStreamSynchronize() on CUDA, illegal during graph capture
      // -- see TrainStep()'s own settle loop for the same fix.
      Step(false);
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
        std::cerr << "Predict() graph capture failed, falling back to non-graph "
                     "execution for this call.\n";
    }

    // Not an else of the block above: on the call that just captured,
    // this replays it immediately so THIS call's output is real, not
    // stale -- see FullPCNetwork::PredictSettle()'s identical pattern.
    if (predictGraphCaptured)
    {
      backend->ReplayGraph(kPredictGraphSlot);
      // cudaGraphLaunch() only enqueues the replay -- the
      // CopyToHost() below needs it to have actually finished.
      backend->Synchronize();
    }
    else
      settleAndProject();
  }
  else
  {
    settleAndProject();
  }

  DirectKPPCLayer* terminal = GetTerminalLayer();
  const float* beliefs = terminal->GetBeliefs();
  size_t count = terminal->GetBatchSize() * terminal->GetInputSize();

  std::vector<float> result(count);
  backend->CopyToHost(result.data(), beliefs, count);
  return result;
}

void DirectKPPCNetwork::Compile()
{
  // See FullPCNetwork::Compile()'s identical check, terminalSize is
  // supplied on every AddLayer() call rather than inferred, and a
  // mismatch against the real terminal layer's size previously went
  // uncaught, silently corrupting W via an out-of-bounds GEMM in
  // DirectFeedbackUpdate().
  if (!layers.empty() && layers.back()->GetInputSize() != layers.back()->GetTerminalSize())
  {
    throw std::invalid_argument(
        "DirectKPPCNetwork::Compile(): terminalSize (" +
        std::to_string(layers.back()->GetTerminalSize()) +
        ") passed to AddLayer() doesn't match the actual terminal layer's size (" +
        std::to_string(layers.back()->GetInputSize()) + ").");
  }

#pragma omp parallel
  {
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  }

  // Detect the batched-GPU fast path's eligible run -- see its member
  // doc comments in the header for what this means and why it's scoped
  // to GPU only, and why it never includes layers.front().
  batchedRunStart = 0;
  batchedRunCount = 0;
  batchedRunWidth = 0;
  if (device == DeviceType::DEVICE_GPU && layers.size() >= 4)
  {
    size_t runStart = 1;
    size_t H = layers[runStart]->GetInputSize();
    size_t TS0 = layers[runStart]->GetTerminalSize();
    ActivationType act0 = layers[runStart]->GetActivationType();

    // Extend the run forward while every layer still matches layer
    // runStart's own shape/activation/terminalSize, stopping (never
    // including) at the terminal layer itself. This naturally stops
    // one layer short of the terminal in the common case where the
    // hidden width H differs from terminalSize: ProjectForward()'s own
    // shape-consistency requirement forces the LAST middle layer's
    // nextSize to equal terminalSize (Compile()'s own check below
    // requires layers.back()->GetInputSize()==terminalSize, and that
    // size must match whatever fed into it) -- when H != terminalSize,
    // that layer's GetOutputSize()!=H breaks the match below and the
    // loop correctly stops just before it, leaving it as an ordinary
    // (non-run) layer like layers.front()/layers.back().
    size_t runEnd = runStart;
    while (runEnd + 1 < layers.size() && layers[runEnd]->GetInputSize() == H &&
           layers[runEnd]->GetOutputSize() == H && layers[runEnd]->GetTerminalSize() == TS0 &&
           layers[runEnd]->GetActivationType() == act0)
    {
      ++runEnd;
    }

    if (runEnd - runStart >= 2)
    {
      batchedRunStart = runStart;
      batchedRunCount = runEnd - runStart;
      batchedRunWidth = H;
    }
  }

  size_t total_floats_needed = 0;
  for (size_t i = 0; i < layers.size(); ++i)
  {
    if (batchedRunCount > 0 && i >= batchedRunStart && i < batchedRunStart + batchedRunCount)
      continue; // this layer's floats come from the flat buffers below instead
    total_floats_needed += layers[i]->GetRequiredFloats();
  }

  auto pad16 = [](size_t n) { return (n + 15) & ~(size_t)15; };
  const size_t H = batchedRunWidth;
  const size_t L = batchedRunCount;
  const size_t B = (size_t)batchSize;
  const size_t TS = layers.empty() ? 0 : layers.back()->GetTerminalSize();
  const size_t own = B * H;
  const size_t w = H * H;
  // Matches CUDABackend::MatMulBatched()'s own kMinSafeDim: below this,
  // it falls back to a plain sequential MatMul() loop rather than risk
  // cuBLASLt's small-matrix output-overrun bug (see that function's own
  // comment). DFA's proj GEMM and UpdateWeights()' Psi GEMM both use
  // terminalSize as a GEMM dimension, so a small terminalSize (the
  // common case -- 10 for MNIST, say) always tripped that fallback for
  // the whole batched run, turning an O(1)-dispatch call back into
  // batchedRunCount-1 sequential ones. Padding Psi's own row width to
  // kPsiMinSafeWidth (permanently-zero columns beyond terminalSize,
  // never read or written by anything outside the two batched GEMMs
  // below) clears that threshold safely: the extra columns always
  // multiply against an equally-zero-padded copy of the terminal
  // error, so they can never contribute anything but zero to either
  // GEMM's real output. CPU's own MatMulBatched() fallback exists for
  // an unrelated reason (OpenBLAS's cblas_sgemm_batch crash) and never
  // sees this cost in the first place, so padding there would only
  // waste memory for no benefit.
  constexpr size_t kPsiMinSafeWidth = 128;
  const size_t psiStride =
      (device == DeviceType::DEVICE_GPU && TS > 0 && TS < kPsiMinSafeWidth) ? kPsiMinSafeWidth : TS;
  const size_t direct = H * psiStride;

  if (L > 0)
  {
    // z, e, mu, cachedMu, preact, cachedPreact, derivScratch, feedbackError, proj, feedbackScratch
    total_floats_needed += pad16(own * L) * 10;
    total_floats_needed += pad16(w * L) * 4;      // W, grad_W, m_W, v_W
    total_floats_needed += pad16(H * L) * 5;      // b, biasGradScratch, grad_b, m_b, v_b
    total_floats_needed += pad16(direct * L) * 4; // Psi, grad_Psi, m_Psi, v_Psi
    total_floats_needed += pad16(L) * 4;          // t_device, lr_device, tPsi_device, fl_device
    if (psiStride != TS)
      total_floats_needed += pad16(B * psiStride); // flatTerminalEPadded
  }

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
    for (size_t i = 0; i < layers.size(); ++i)
    {
      if (batchedRunCount > 0 && i >= batchedRunStart && i < batchedRunStart + batchedRunCount)
        continue; // bound below instead, into the flat run buffers
      layers[i]->BindMemory(*gpuArena);
      layers[i]->SetTerminalLayer(layers.back().get());
    }

    if (L > 0)
    {
      // Must happen before the first TrainStep()/Predict() ever runs,
      // never lazily inside one: MatMulBatched()'s own scratch growth
      // (cudaMalloc/cudaMallocHost) isn't a capturable operation, and
      // the first real batched GEMM call always happens inside that
      // call's graph capture.
      backend->PrewarmBatchedGemmScratch(L);

      // Same reasoning, for MatMulBatched()'s own CUTLASS Grouped GEMM
      // fast path: its per-item problem-size array and solver workspace
      // are sized by (M,N,K,lda,ldb,ldc), not just batchCount, so every
      // distinct shape used below needs its own prewarm call -- one per
      // (transA,transB,M,N,K,lda,ldb,ldc) combination actually passed to
      // MatMulBatched() in this file (ComputeMuBatchedRun(),
      // DirectFeedbackUpdateBatchedRun(), StepBatchedRunUpdateState(),
      // UpdateWeightsBatchedRun() -- grep MatMulBatched( to recheck if
      // any of those are ever changed). batchCount uses L (not M=L-1,
      // which some of these sites actually pass) since prewarming for
      // the larger count is always sufficient and keeps this list
      // shape-only, not also per-batchCount.
      backend->PrewarmGroupedGemmScratch(false, true, (int)B, (int)H, (int)H, (int64_t)H,
                                         (int64_t)H, (int64_t)H, L);
      backend->PrewarmGroupedGemmScratch(false, true, (int)B, (int)H, (int)psiStride,
                                         (int64_t)psiStride, (int64_t)psiStride, (int64_t)H, L);
      backend->PrewarmGroupedGemmScratch(true, false, (int)H, (int)H, (int)B, (int64_t)H,
                                         (int64_t)H, (int64_t)H, L);
      backend->PrewarmGroupedGemmScratch(false, false, (int)B, (int)H, (int)H, (int64_t)H,
                                         (int64_t)H, (int64_t)H, L);
      backend->PrewarmGroupedGemmScratch(true, false, (int)H, (int)psiStride, (int)B, (int64_t)H,
                                         (int64_t)psiStride, (int64_t)psiStride, L);

      flatZ = gpuArena->AllocateFloats(own * L);
      flatE = gpuArena->AllocateFloats(own * L);
      flatW = gpuArena->AllocateFloats(w * L);
      flatB = gpuArena->AllocateFloats(H * L);
      flatMu = gpuArena->AllocateFloats(own * L);
      flatCachedMu = gpuArena->AllocateFloats(own * L);
      flatPreact = gpuArena->AllocateFloats(own * L);
      flatCachedPreact = gpuArena->AllocateFloats(own * L);
      flatDerivScratch = gpuArena->AllocateFloats(own * L);
      flatFeedbackError = gpuArena->AllocateFloats(own * L);
      flatProj = gpuArena->AllocateFloats(own * L);
      flatFeedbackScratch = gpuArena->AllocateFloats(own * L);
      flatPsi = gpuArena->AllocateFloats(direct * L);
      flatBiasGradScratch = gpuArena->AllocateFloats(H * L);
      flatGradW = gpuArena->AllocateFloats(w * L);
      flatMW = gpuArena->AllocateFloats(w * L);
      flatVW = gpuArena->AllocateFloats(w * L);
      flatGradB = gpuArena->AllocateFloats(H * L);
      flatMB = gpuArena->AllocateFloats(H * L);
      flatVB = gpuArena->AllocateFloats(H * L);
      flatTDevice = reinterpret_cast<int*>(gpuArena->AllocateFloats(L));
      flatLrDevice = gpuArena->AllocateFloats(L);
      flatGradPsi = gpuArena->AllocateFloats(direct * L);
      flatMPsi = gpuArena->AllocateFloats(direct * L);
      flatVPsi = gpuArena->AllocateFloats(direct * L);
      flatTPsiDevice = reinterpret_cast<int*>(gpuArena->AllocateFloats(L));
      flatFlDevice = gpuArena->AllocateFloats(L);
      if (psiStride != TS)
        flatTerminalEPadded = gpuArena->AllocateFloats(B * psiStride);

      backend->Zero(flatZ, own * L);
      backend->Zero(flatE, own * L);
      backend->Zero(flatB, H * L);
      backend->Zero(flatMu, own * L);
      backend->Zero(flatCachedMu, own * L);
      backend->Zero(flatPreact, own * L);
      backend->Zero(flatCachedPreact, own * L);
      backend->Zero(flatDerivScratch, own * L);
      backend->Zero(flatFeedbackError, own * L);
      backend->Zero(flatProj, own * L);
      backend->Zero(flatFeedbackScratch, own * L);
      backend->Zero(flatPsi, direct * L);
      backend->Zero(flatBiasGradScratch, H * L);
      backend->Zero(flatGradW, w * L);
      backend->Zero(flatMW, w * L);
      backend->Zero(flatVW, w * L);
      backend->Zero(flatGradB, H * L);
      backend->Zero(flatMB, H * L);
      backend->Zero(flatVB, H * L);
      backend->Zero(flatGradPsi, direct * L);
      backend->Zero(flatMPsi, direct * L);
      backend->Zero(flatVPsi, direct * L);
      if (flatTerminalEPadded != nullptr)
        backend->Zero(flatTerminalEPadded, B * psiStride);

      std::vector<int> zeros(L, 0);
      backend->CopyFromHost(reinterpret_cast<float*>(flatTDevice),
                            reinterpret_cast<const float*>(zeros.data()), L);
      backend->CopyFromHost(reinterpret_cast<float*>(flatTPsiDevice),
                            reinterpret_cast<const float*>(zeros.data()), L);

      batchedA.assign(L, nullptr);
      batchedB.assign(L, nullptr);
      batchedC.assign(L, nullptr);

      for (size_t l = 0; l < L; ++l)
      {
        DirectKPPCLayer* layer = layers[batchedRunStart + l].get();

        DirectKPPCLayer::ExternalMemory mem;
        mem.z = flatZ + l * own;
        mem.e = flatE + l * own;
        mem.W = flatW + l * w;
        mem.b = flatB + l * H;
        mem.mu = flatMu + l * own;
        mem.cachedMu = flatCachedMu + l * own;
        mem.preact = flatPreact + l * own;
        mem.cachedPreact = flatCachedPreact + l * own;
        mem.derivScratch = flatDerivScratch + l * own;
        mem.feedbackError = flatFeedbackError + l * own;
        mem.proj = flatProj + l * own;
        mem.Psi = flatPsi + l * direct;
        mem.psiStride = psiStride;
        mem.feedbackScratch = flatFeedbackScratch + l * own;
        mem.biasGradScratch = flatBiasGradScratch + l * H;
        mem.grad_W = flatGradW + l * w;
        mem.m_W = flatMW + l * w;
        mem.v_W = flatVW + l * w;
        mem.grad_b = flatGradB + l * H;
        mem.m_b = flatMB + l * H;
        mem.v_b = flatVB + l * H;
        mem.grad_Psi = flatGradPsi + l * direct;
        mem.m_Psi = flatMPsi + l * direct;
        mem.v_Psi = flatVPsi + l * direct;
        mem.t_device = flatTDevice + l;
        mem.lr_device = flatLrDevice + l;
        mem.tPsi_device = flatTPsiDevice + l;
        mem.fl_device = flatFlDevice + l;

        float lrVal = layer->GetLearningRate();
        float flVal = layer->GetFeedbackRate();
        backend->CopyFromHost(mem.lr_device, &lrVal, 1);
        backend->CopyFromHost(mem.fl_device, &flVal, 1);

        layer->BindExternalMemory(mem);
        layer->SetTerminalLayer(layers.back().get());
      }
    }
  }
#endif
}
} // namespace Deep