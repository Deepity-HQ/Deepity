#include "deepity/layers/FullPCLayer.h"
#include "deepity/backend/CPUBackend.h"
#include "deepity/utils/LayerBindHelpers.h"
#include "deepity/utils/Optimize.h"
#include "deepity/utils/RandomizeDistribution.h"
#include <cstring>

namespace Deep
{
namespace
{
void DeleteBackend(IComputeBackend* p)
{
  delete p;
}
void NoOpDeleter(IComputeBackend*) {}
} // namespace

FullPCLayer::FullPCLayer(size_t size, size_t nextSize, size_t terminalSize, size_t batchSize,
                         float learningRate, float inferenceRate, float feedback, float lmbda,
                         ActivationType aType, ActivationType dType, IComputeBackend* backend)
    : size(size)
    , nextSize(nextSize)
    , terminalSize(terminalSize)
    , batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , fl(feedback)
    , lmbda(lmbda)
    , activationType(aType)
    , backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
{
  DynamicThread(batchSize);

  localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
  BindMemory(*localArena);
}

void FullPCLayer::SetLearningRate(float learningRate) noexcept
{
  lr = learningRate;
  if (lr_device)
    backend->CopyFromHost(lr_device, &lr, 1);
}

void FullPCLayer::SetFeedbackRate(float feedbackRate) noexcept
{
  fl = feedbackRate;
  if (fl_device)
    backend->CopyFromHost(fl_device, &fl, 1);
}

template <typename ArenaT> void FullPCLayer::BindMemory(ArenaT& arena)
{
  size_t own_state_size = batchSize * size;
  size_t out_state_size = batchSize * nextSize;
  size_t direct_size = size * terminalSize;

  z = arena.AllocateFloats(own_state_size);
  e = arena.AllocateFloats(own_state_size);
  rowEnergies = arena.AllocateFloats(batchSize);

  backend->Zero(rowEnergies, batchSize);
  backend->Zero(z, own_state_size);
  backend->Zero(e, own_state_size);

  if (nextSize > 0)
  {
    size_t w_size = size * nextSize;

    W = arena.AllocateFloats(w_size);
    b = arena.AllocateFloats(nextSize);
    mu = arena.AllocateFloats(out_state_size);
    cachedMu = arena.AllocateFloats(out_state_size);
    preact = arena.AllocateFloats(out_state_size);
    cachedPreact = arena.AllocateFloats(out_state_size);
    derivScratch = arena.AllocateFloats(out_state_size);
    feedbackError = arena.AllocateFloats(out_state_size);
    proj = arena.AllocateFloats(out_state_size);
    Psi = arena.AllocateFloats(direct_size);
    feedbackScratch = arena.AllocateFloats(own_state_size);
    v = arena.AllocateFloats(own_state_size);
    backend->Zero(v, own_state_size);
    adjoint = arena.AllocateFloats(own_state_size);
    backend->Zero(adjoint, own_state_size);
    biasGradScratch = arena.AllocateFloats(nextSize);

    backend->Zero(b, nextSize);
    backend->Zero(mu, out_state_size);
    backend->Zero(cachedMu, out_state_size);
    backend->Zero(preact, out_state_size);
    backend->Zero(cachedPreact, out_state_size);
    backend->Zero(derivScratch, out_state_size);
    backend->Zero(feedbackError, out_state_size);
    backend->Zero(proj, out_state_size);
    backend->Zero(Psi, direct_size);
    backend->Zero(feedbackScratch, own_state_size);
    backend->Zero(biasGradScratch, nextSize);

    // Always allocated, see GetRequiredFloats().
    BindAdamState(arena, backend.get(), w_size, grad_W, m_W, v_W);
    BindAdamState(arena, backend.get(), nextSize, grad_b, m_b, v_b);
    BindLearningRateState(arena, backend.get(), t_device, lr_device, lr);

    BindAdamState(arena, backend.get(), direct_size, grad_Psi, m_Psi, v_Psi);
    BindLearningRateState(arena, backend.get(), tPsi_device, fl_device, fl);
  }

  ReleaseLocalArenaIfForeign(localArena, arena);
}

size_t FullPCLayer::GetRequiredFloats() const noexcept
{
  auto pad16 = [](size_t n) { return (n + 15) & ~(size_t)15; };

  size_t total = 0;
  size_t own_state_size = batchSize * size;
  size_t direct_size = terminalSize * size;

  total += pad16(own_state_size) * 2;
  total += pad16(batchSize);

  if (nextSize > 0)
  {
    size_t out_state_size = (size_t)batchSize * nextSize;
    size_t w_size = (size_t)size * nextSize;

    total += pad16(w_size);
    total += pad16(nextSize);
    total += pad16(out_state_size) * 3; // mu, cachedMu, proj
    total += pad16(out_state_size) * 4; // preact, cachedPreact, derivScratch, feedbackError
    total += pad16(own_state_size);     // feedbackScratch
    total += pad16(own_state_size) * 2; // v, adjoint
    total += pad16(direct_size);
    total += pad16(nextSize);

    // Always allocated, regardless of optimizer, so switching to Adam/AdamW after Compile() stays
    // safe.
    total += pad16(w_size) * 3;
    total += pad16(nextSize) * 3;
    total += pad16(1) * 2;

    total += pad16(direct_size) * 3;
    total += pad16(1) * 2;
  }

  return total;
}

void FullPCLayer::RandomizeWeights(std::mt19937& seedGenerator, const char* distribution) noexcept
{
  if (nextSize == 0)
    return;

  std::uniform_int_distribution<uint32_t> seedDist;
  size_t Wsz = size * nextSize;
  size_t Psisz = terminalSize * size;

  // muPC's b_l = 1 (unit variance, Table 1): `a` alone carries the
  // width/depth scaling, so W's own init shouldn't ALSO be fan-scaled.
  // Psi is Deepity's own direct-feedback addition, outside the muPC
  // paper's scope, so it keeps the fan-scaled init unconditionally.
  float limit = useMuPCInit ? 1.0f : std::sqrt(2.0f / (size + nextSize));
  float limPsi = std::sqrt(2.0f / (size + terminalSize));

  uint32_t seedW = seedDist(seedGenerator);
  uint32_t seedPsi = seedDist(seedGenerator);

  RandomizeWithDistribution(backend.get(), W, Wsz, distribution, seedW, 0.0f, limit);
  backend->RandomizeNormal(Psi, Psisz, 0.0f, limPsi, seedPsi);
}

std::map<std::string, Deep::TensorDescriptor> Deep::FullPCLayer::GetStateDict() const
{
  return {{"W", {W, std::vector<size_t>{nextSize, size}}},
          {"b", {b, std::vector<size_t>{nextSize}}},
          {"Psi", {Psi, std::vector<size_t>{nextSize, terminalSize}}}};
}

float FullPCLayer::CalculateState(bool needEnergy) noexcept
{
  const size_t N = batchSize * size;

  if (layerBelow == nullptr)
  {
    backend->Zero(e, N);
    if (nextSize > 0)
      ComputeMuOnly();
    return 0.0f;
  }

  float totalEnergy = 0.0f;
  if (useCrossEntropy)
  {
    if (needEnergy)
      totalEnergy = backend->ComputeSoftmaxCrossEntropyErrorAndEnergy(
          e, z, layerBelow->mu, batchSize, size, rowEnergies);
    else
      backend->ComputeSoftmaxCrossEntropyError(e, z, layerBelow->mu, batchSize, size);
  }
  else if (needEnergy)
  {
    totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->mu, N);
  }
  else
  {
    backend->ComputeError(e, z, layerBelow->mu, N);
  }

  if (nextSize > 0)
    ComputeMuOnly();

  return needEnergy ? totalEnergy : 0.0f;
}

void FullPCLayer::ComputeMuOnly() noexcept
{
  if (nextSize == 0)
    return;

  size_t Nout = batchSize * nextSize;
  size_t N = batchSize * size;

  if (isClamped && muCacheValid)
  {
    backend->Copy(mu, cachedMu, Nout);
    backend->Copy(preact, cachedPreact, Nout);
    return;
  }

  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true,
      (int)batchSize,
      (int)nextSize,
      (int)size,
      a,
      z,
      (int)size,
      W,
      (int)size,
      0.0f,
      mu,
      (int)nextSize);

  backend->AddBiasBroadcast(mu, b, batchSize, nextSize);

  // Snapshotted BEFORE Activation() overwrites mu in place, same
  // reasoning as SimplePCLayer/SimpleConvPCLayer's own preact. The
  // residual add below happens AFTER activation, so it's deliberately
  // NOT part of this snapshot.
  backend->Copy(preact, mu, Nout);
  backend->Activation(activationType, mu, Nout);

  // Residual (forward): mu += z (raw, un-activated, the skip bypasses
  // the nonlinearity entirely, matching muPC's "1-skip" formulation).
  // Requires size == nextSize; not checked here (checked once at
  // network Compile() time when SetResidual() is called with full
  // shape info available).
  if (useResidual)
    backend->AxpyInto(mu, z, Nout, 1.0f);

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    backend->Copy(cachedPreact, preact, Nout);
    muCacheValid = true;
  }
}

void FullPCLayer::UpdateState() noexcept
{
  size_t N = (size_t)batchSize * size;

  if (isClamped)
    return;

  if (useCrossEntropy)
    return;

  if (layerAbove != nullptr && nextSize > 0)
  {
    const float* e_above = layerAbove->GetErrors();

    bool topDownMeaningless = layerAbove->GetCrossEntropy() && !layerAbove->IsClamped();

    if (topDownMeaningless)
    {
      backend->AxpyInto(z, e, N, -ir);
    }
    else
    {
      size_t Nout = (size_t)batchSize * nextSize;

      // The derivative applies at the source (mu=phi(a*W@z+b)'s own
      // nonlinearity), to e_above, before projecting it through W --
      // not after, and not via f'(z). Computed into derivScratch,
      // never into mu itself -- see derivScratch's own doc comment
      // for why.
      ActivationType dType = ToDerivativeType(activationType);
      backend->ActivationDerivativeInto(dType, derivScratch, preact, Nout);
      backend->MultiplyInto(feedbackError, e_above, derivScratch, Nout);

      backend->MatMul(
          /*transA=*/false,
          /*transB=*/false,
          (int)batchSize,
          (int)size,
          (int)nextSize,
          a,
          feedbackError,
          (int)nextSize,
          W,
          (int)size,
          0.0f,
          feedbackScratch,
          (int)size);

      if (useMomentum)
      {
        // v = beta*v + (1-beta)*(feedbackScratch - e); z += ir*v.
        // Same formula FusedStateUpdateMomentum() documents, just
        // without ITS inline f'(z)-at-destination multiply, already
        // folded into feedbackScratch above instead.
        backend->AxpyInto(feedbackScratch, e, N, -1.0f);
        backend->Scale(v, N, momentumBeta);
        backend->AxpyInto(v, feedbackScratch, N, 1.0f - momentumBeta);
        backend->AxpyInto(z, v, N, ir);
      }
      else
      {
        backend->AxpyInto(feedbackScratch, e, N, -1.0f);
        backend->AxpyInto(z, feedbackScratch, N, ir);
      }

      if (useResidual)
        backend->AxpyInto(z, e_above, N, ir);
    }
  }
  else
  {
    backend->AxpyInto(z, e, N, -ir);
  }
}

void FullPCLayer::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  size_t Nout = (size_t)batchSize * nextSize;
  const float* e_above = layerAbove->GetErrors();

  // Recomputed fresh from preact, not read back from a previous
  // UpdateState() call -- see UpdateState()'s own comment on why mu
  // itself must never hold this.
  ActivationType dType = ToDerivativeType(activationType);
  backend->ActivationDerivativeInto(dType, derivScratch, preact, Nout);
  backend->MultiplyInto(feedbackError, e_above, derivScratch, Nout);
  const float* local_grad = feedbackError;

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > 0.0f)
      backend->Scale(W, (size_t)nextSize * size, 1.0f - lmbda);

    // d(mu)/dW = a * f'(preact) * z, so the weight-gradient GEMM's
    // scale factor picks up an extra *a (was just lr/batchSize); the
    // f'(preact) factor is already folded into local_grad above.
    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        (int)batchSize,
        a * lr / batchSize,
        local_grad,
        (int)nextSize,
        z,
        (int)size,
        1.0f,
        W,
        (int)size);

    float lr_batch = lr / batchSize;
    backend->SumRows(biasGradScratch, local_grad, batchSize, nextSize);
    backend->AxpyInto(b, biasGradScratch, nextSize, lr_batch);

    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(t_device);

    size_t num_weights = (size_t)nextSize * size;
    float adam_scale = -1.0f * a; // d(mu)/dW picks up *a here too

    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        (int)batchSize,
        adam_scale,
        local_grad,
        (int)nextSize,
        z,
        (int)size,
        0.0f,
        grad_W,
        (int)size);

    backend->SumRows(grad_b, local_grad, batchSize, nextSize);
    backend->Scale(grad_b, nextSize, -1.0f); // bias has no `a` scaling

    if (opt == OptimizerType::ADAMW)
      backend->AdamWStep(W, grad_W, m_W, v_W, num_weights, t_device, lr_device, lmbda);
    else
      backend->AdamStep(W, grad_W, m_W, v_W, num_weights, t_device, lr_device);

    backend->AdamStep(b, grad_b, m_b, v_b, nextSize, t_device, lr_device);
    break;
  }
  }

  switch (optPsi)
  {
  case OptimizerType::SGD:
  {
    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)size,
        (int)terminalSize,
        (int)batchSize,
        fl / batchSize,
        z,
        (int)size,
        terminalLayer->GetErrors(),
        (int)terminalSize,
        1.0f,
        Psi,
        (int)terminalSize);
    break;
  }
  case OptimizerType::ADAMW:
  case OptimizerType::ADAM:
  {
    backend->IncrementCounter(tPsi_device);

    size_t num_weights_psi = size * terminalSize;
    float adam_scale = -1.0f;

    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)size,
        (int)terminalSize,
        (int)batchSize,
        adam_scale,
        z,
        (int)size,
        terminalLayer->GetErrors(),
        (int)terminalSize,
        0.0f,
        grad_Psi,
        (int)terminalSize);

    if (optPsi == OptimizerType::ADAMW)
      backend->AdamWStep(
          Psi, grad_Psi, m_Psi, v_Psi, num_weights_psi, tPsi_device, fl_device, lmbda);
    else
      backend->AdamStep(Psi, grad_Psi, m_Psi, v_Psi, num_weights_psi, tPsi_device, fl_device);

    break;
  }
  }
}

void FullPCLayer::DirectFeedbackUpdate() noexcept
{
  if (layerAbove == nullptr || layerAbove->GetDirectFeedbackWeights() == nullptr)
    return;

  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true,
      (int)batchSize,
      (int)nextSize,
      (int)terminalSize,
      1.0f,
      terminalLayer->GetErrors(),
      (int)terminalSize,
      layerAbove->GetDirectFeedbackWeights(),
      (int)terminalSize,
      0.0f,
      proj,
      (int)nextSize);

  // The derivative applies at the source (mu's own nonlinearity),
  // same convention as UpdateWeights()'s real gradient -- keeps this
  // DFA approximation structurally consistent with it, even though
  // Psi/proj are never exact (see tDirectKPVerify.cpp's file header,
  // same reasoning here).
  {
    size_t Nout = (size_t)batchSize * nextSize;
    ActivationType dType = ToDerivativeType(activationType);
    backend->ActivationDerivativeInto(dType, derivScratch, preact, Nout);
    backend->MultiplyInto(proj, proj, derivScratch, Nout);
  }

  backend->MatMul(
      /*transA=*/true,
      /*transB=*/false,
      (int)nextSize,
      (int)size,
      (int)batchSize,
      fl / batchSize,
      proj,
      (int)nextSize,
      z,
      (int)size,
      1.0f,
      W,
      (int)size);
}

void FullPCLayer::ReconstructBelief() noexcept
{
  if (layerBelow == nullptr)
    return;

  const size_t N = batchSize * size;
  backend->Copy(z, layerBelow->GetMu(), N);
  backend->AxpyInto(z, e, N, 1.0f);
}

void FullPCLayer::ComputeAdjoint(const float* adjointAbove, float adjointAboveScale) noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  const size_t N = batchSize * size;
  const size_t Nout = batchSize * nextSize;

  // The derivative applies at the source (mu=phi(a*W@z+b)'s own
  // nonlinearity), to adjointAbove, before projecting it through W --
  // not after, and not via f'(z). Same feedback-signal shape
  // UpdateState() already uses, but the result is materialized into
  // `adjoint` (not applied inline) so the layer below can read it as
  // ITS incoming signal -- the "global via AD" chain, one hop at a
  // time, rather than a one-hop local message.
  ActivationType dType = ToDerivativeType(activationType);
  backend->ActivationDerivativeInto(dType, derivScratch, preact, Nout);
  backend->MultiplyInto(feedbackError, adjointAbove, derivScratch, Nout);

  backend->MatMul(
      /*transA=*/false,
      /*transB=*/false,
      (int)batchSize,
      (int)size,
      (int)nextSize,
      a * adjointAboveScale,
      feedbackError,
      (int)nextSize,
      W,
      (int)size,
      0.0f,
      adjoint,
      (int)size);

  // Residual (forward): mu += z, UN-activated, bypassing phi' entirely,
  // see ComputeMuOnly()'s own comment. Its Jacobian contributes an extra
  // +I term to d(mu)/dz, so the adjoint needs an extra
  // +adjointAboveScale*adjointAbove on top of the W-path above (no `a`
  // factor: the residual add in ComputeMuOnly() has coefficient 1.0, not
  // `a`, matching UpdateState()'s own un-scaled `AxpyInto(z, e_above, N,
  // ir)` residual term for the one-hop settling path). Requires
  // nextSize == size, already guaranteed wherever SetResidual(true) is
  // actually set (checked once at network Compile() time).
  if (useResidual)
    backend->AxpyInto(adjoint, adjointAbove, N, adjointAboveScale);
}

void FullPCLayer::UpdateErrorEPC() noexcept
{
  if (isClamped)
    return;

  const size_t N = batchSize * size;
  backend->Scale(e, N, 1.0f - ir);
  backend->AxpyInto(e, adjoint, N, -ir);
}

void FullPCLayer::ClampState(const std::vector<float>& inputData) noexcept
{
  size_t copyFloats = (std::min)(inputData.size(), (size_t)(batchSize * size));
  backend->CopyFromHost(z, inputData.data(), copyFloats);
  isClamped = true;
  muCacheValid = false;
}

void FullPCLayer::UnclampState() noexcept
{
  isClamped = false;
}

void FullPCLayer::ResetState() noexcept
{
  size_t N = (size_t)batchSize * size;
  backend->Zero(z, N);
  backend->Zero(e, N);
  if (v)
    backend->Zero(v, N);
}

template void FullPCLayer::BindMemory<MemoryArena>(MemoryArena& arena);
#if defined(DEEPITY_USE_CUDA)
template void FullPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena& arena);
#endif
} // namespace Deep
