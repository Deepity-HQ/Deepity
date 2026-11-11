#include "deepity/layers/DirectKPPCLayer.h"
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

DirectKPPCLayer::DirectKPPCLayer(size_t size, size_t nextSize, size_t terminalSize,
                                 size_t batchSize, float learningRate, float inferenceRate,
                                 float feedback, float lmbda, ActivationType aType,
                                 ActivationType dType, IComputeBackend* backend)
    : size(size)
    , nextSize(nextSize)
    , terminalSize(terminalSize)
    , batchSize(batchSize)
    , psiStride(terminalSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , fl(feedback)
    , lmbda(lmbda)
    , activationType(aType)
    , backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
{
  this->activation = To_Fn(aType);
  this->activationDerivative = To_dFn(dType);
  this->activationDerivativeInto = To_dFn2(dType);
  DynamicThread(batchSize);

  localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
  BindMemory(*localArena);
}

void DirectKPPCLayer::SetLearningRate(float learningRate) noexcept
{
  lr = learningRate;
  if (lr_device)
    backend->CopyFromHost(lr_device, &lr, 1);
}

void DirectKPPCLayer::SetFeedbackRate(float feedbackRate) noexcept
{
  fl = feedbackRate;
  if (fl_device)
    backend->CopyFromHost(fl_device, &fl, 1);
}

template <typename ArenaT> void DirectKPPCLayer::BindMemory(ArenaT& arena)
{
  size_t own_state_size = batchSize * size;
  size_t out_state_size = batchSize * nextSize;
  size_t direct_size = size * terminalSize;

  z = arena.AllocateFloats(own_state_size);
  e = arena.AllocateFloats(own_state_size);

  ArenaZero<ArenaT>(backend.get(), z, own_state_size);
  ArenaZero<ArenaT>(backend.get(), e, own_state_size);

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
    biasGradScratch = arena.AllocateFloats(nextSize);

    ArenaZero<ArenaT>(backend.get(), b, nextSize);
    ArenaZero<ArenaT>(backend.get(), mu, out_state_size);
    ArenaZero<ArenaT>(backend.get(), cachedMu, out_state_size);
    ArenaZero<ArenaT>(backend.get(), preact, out_state_size);
    ArenaZero<ArenaT>(backend.get(), cachedPreact, out_state_size);
    ArenaZero<ArenaT>(backend.get(), derivScratch, out_state_size);
    ArenaZero<ArenaT>(backend.get(), feedbackError, out_state_size);
    ArenaZero<ArenaT>(backend.get(), proj, out_state_size);
    ArenaZero<ArenaT>(backend.get(), Psi, direct_size);
    ArenaZero<ArenaT>(backend.get(), feedbackScratch, own_state_size);
    ArenaZero<ArenaT>(backend.get(), biasGradScratch, nextSize);

    // Always allocated, see GetRequiredFloats().
    BindAdamState(arena, backend.get(), w_size, grad_W, m_W, v_W);
    BindAdamState(arena, backend.get(), nextSize, grad_b, m_b, v_b);
    BindLearningRateState(arena, backend.get(), t_device, lr_device, lr);

    BindAdamState(arena, backend.get(), direct_size, grad_Psi, m_Psi, v_Psi);
    BindLearningRateState(arena, backend.get(), tPsi_device, fl_device, fl);
  }

  ReleaseLocalArenaIfForeign(localArena, arena);
}

void DirectKPPCLayer::BindExternalMemory(const ExternalMemory& mem) noexcept
{
  z = mem.z;
  e = mem.e;
  W = mem.W;
  b = mem.b;
  mu = mem.mu;
  cachedMu = mem.cachedMu;
  preact = mem.preact;
  cachedPreact = mem.cachedPreact;
  derivScratch = mem.derivScratch;
  feedbackError = mem.feedbackError;
  proj = mem.proj;
  Psi = mem.Psi;
  psiStride = mem.psiStride;
  feedbackScratch = mem.feedbackScratch;
  biasGradScratch = mem.biasGradScratch;
  grad_W = mem.grad_W;
  m_W = mem.m_W;
  v_W = mem.v_W;
  grad_b = mem.grad_b;
  m_b = mem.m_b;
  v_b = mem.v_b;
  grad_Psi = mem.grad_Psi;
  m_Psi = mem.m_Psi;
  v_Psi = mem.v_Psi;
  t_device = mem.t_device;
  lr_device = mem.lr_device;
  tPsi_device = mem.tPsi_device;
  fl_device = mem.fl_device;

  // This layer's own private fallback arena (eagerly created and bound
  // in the constructor, before Compile() ever runs) is now entirely
  // unused -- same cleanup BindMemory() does via
  // ReleaseLocalArenaIfForeign() for the ordinary arena-bound case.
  localArena.reset();
}

size_t DirectKPPCLayer::GetRequiredFloats() const noexcept
{
  auto pad16 = [](size_t n) { return (n + 15) & ~(size_t)15; };

  size_t total = 0;
  size_t own_state_size = batchSize * size;
  size_t direct_size = terminalSize * size;

  total += pad16(own_state_size) * 2;

  if (nextSize > 0)
  {
    size_t out_state_size = (size_t)batchSize * nextSize;
    size_t w_size = (size_t)size * nextSize;

    total += pad16(w_size);
    total += pad16(nextSize);
    total += pad16(out_state_size) * 7; // mu, cachedMu, proj, preact, cachedPreact, derivScratch, feedbackError
    total += pad16(own_state_size);     // feedbackScratch
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

void DirectKPPCLayer::RandomizeWeights(std::mt19937& seedGenerator,
                                       const char* distribution) noexcept
{
  if (nextSize == 0)
    return;

  std::uniform_int_distribution<uint32_t> seedDist;
  size_t Wsz = size * nextSize;
  float limit = std::sqrt(2.0f / (size + nextSize));
  float limPsi = std::sqrt(2.0f / (size + terminalSize));

  uint32_t seedW = seedDist(seedGenerator);

  RandomizeWithDistribution(backend.get(), W, Wsz, distribution, seedW, 0.0f, limit);

  if (psiStride == terminalSize)
  {
    uint32_t seedPsi = seedDist(seedGenerator);
    backend->RandomizeNormal(Psi, terminalSize * size, 0.0f, limPsi, seedPsi);
  }
  else
  {
    // Padded wider than terminalSize (see GetPsiStride()) -- build the
    // whole buffer host-side, padding columns left zero, one upload
    // instead of size separate per-row GPU calls.
    std::vector<float> hostPsi(size * psiStride, 0.0f);
    std::normal_distribution<float> dist(0.0f, limPsi);
    std::mt19937 psiRng(seedDist(seedGenerator));
    for (size_t row = 0; row < size; ++row)
      for (size_t col = 0; col < terminalSize; ++col)
        hostPsi[row * psiStride + col] = dist(psiRng);
    backend->CopyFromHost(Psi, hostPsi.data(), hostPsi.size());
  }
}

std::map<std::string, Deep::TensorDescriptor> Deep::DirectKPPCLayer::GetStateDict() const
{
  return {{"W", {W, std::vector<size_t>{nextSize, size}}},
          {"b", {b, std::vector<size_t>{nextSize}}},
          {"Psi", {Psi, std::vector<size_t>{nextSize, terminalSize}, psiStride}}};
}

float DirectKPPCLayer::CalculateState(bool needEnergy) noexcept
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
  if (needEnergy)
    totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->mu, N);
  else
    backend->ComputeError(e, z, layerBelow->mu, N);

  if (nextSize > 0)
    ComputeMuOnly();

  return totalEnergy;
}

void DirectKPPCLayer::ComputeMuOnly() noexcept
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
      1.0f,
      z,
      (int)size,
      W,
      (int)size,
      0.0f,
      mu,
      (int)nextSize);

  backend->AddBiasBroadcast(mu, b, batchSize, nextSize);

  // Snapshotted BEFORE Activation() overwrites mu in place, same
  // reasoning as SimplePCLayer/SimpleConvPCLayer's own preact.
  backend->Copy(preact, mu, Nout);
  backend->Activation(activationType, mu, Nout);

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    backend->Copy(cachedPreact, preact, Nout);
    muCacheValid = true;
  }
}

void DirectKPPCLayer::CenterWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;
  if (backend->TryCenterRows(W, nextSize, size))
    return;
  // Fallback, scratch is free here (runs after UpdateWeights()). Two
  // GEMMs: rowSum = W @ ones, then W -= (1/size) * rowSum @ ones^T.
  float* ones = feedbackScratch;
  backend->Fill(ones, size, 1.0f);
  backend->MatMul(false, false, (int)nextSize, 1, (int)size, 1.0f, W, (int)size, ones, 1, 0.0f,
                  biasGradScratch, 1);
  backend->MatMul(false, false, (int)nextSize, (int)size, 1, -1.0f / (float)size, biasGradScratch,
                  1, ones, (int)size, 1.0f, W, (int)size);
}

void DirectKPPCLayer::UpdateState() noexcept
{
  size_t N = (size_t)batchSize * size;

  if (isClamped)
    return;

  if (layerAbove != nullptr && nextSize > 0)
  {
    size_t Nout = batchSize * nextSize;
    const float* e_above = layerAbove->GetErrors();

    // The derivative applies at the source (mu=phi(W@z+b)'s own
    // nonlinearity), to e_above, before projecting it through W -- not
    // after, and not via f'(z). Computed into derivScratch, never into
    // mu itself -- see derivScratch's own doc comment for why.
    backend->ActivationDerivativeInto(ToDerivativeType(activationType), derivScratch, preact, Nout);
    backend->MultiplyInto(feedbackError, e_above, derivScratch, Nout);

    backend->MatMul(
        /*transA=*/false,
        /*transB=*/false,
        (int)batchSize,
        (int)size,
        (int)nextSize,
        1.0f,
        feedbackError,
        (int)nextSize,
        W,
        (int)size,
        0.0f,
        feedbackScratch,
        (int)size);

    backend->AxpyInto(feedbackScratch, e, N, -1.0f);
    backend->AxpyInto(z, feedbackScratch, N, ir);
  }
  else
  {
    backend->AxpyInto(z, e, N, -ir);
  }
}

void DirectKPPCLayer::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  size_t Nout = (size_t)batchSize * nextSize;
  const float* e_above = layerAbove->GetErrors();

  // Recomputed fresh from preact, not read back from a previous
  // UpdateState() call -- see UpdateState()'s own comment on why mu
  // itself must never hold this.
  backend->ActivationDerivativeInto(ToDerivativeType(activationType), derivScratch, preact, Nout);
  backend->MultiplyInto(feedbackError, e_above, derivScratch, Nout);
  const float* local_grad = feedbackError;

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > 0.0f)
      backend->Scale(W, (size_t)nextSize * size, 1.0f - lmbda);

    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        (int)batchSize,
        lr / batchSize,
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
    float adam_scale = -1.0f;

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
    backend->Scale(grad_b, nextSize, adam_scale);

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
        (int)psiStride);
    break;
  }
  case OptimizerType::ADAMW:
  case OptimizerType::ADAM:
  {
    backend->IncrementCounter(tPsi_device);

    // num_weights_psi spans psiStride (not just terminalSize) so Adam's
    // per-element state stays aligned with Psi's own possibly-padded
    // layout; the padding columns' gradient is always exactly zero (the
    // GEMM below only ever writes the real terminalSize columns of
    // each row), so Adam's moment estimates for them settle at zero
    // too and never move those columns off zero.
    size_t num_weights_psi = size * psiStride;
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
        (int)psiStride);

    if (optPsi == OptimizerType::ADAMW)
      backend->AdamWStep(
          Psi, grad_Psi, m_Psi, v_Psi, num_weights_psi, tPsi_device, fl_device, lmbda);
    else
      backend->AdamStep(Psi, grad_Psi, m_Psi, v_Psi, num_weights_psi, tPsi_device, fl_device);

    break;
  }
  }
}

void DirectKPPCLayer::DirectFeedbackUpdate() noexcept
{
  if (layerAbove == nullptr || layerAbove->GetDirectFeedbackWeights() == nullptr)
    return;

  // proj = terminalLayer->GetErrors() @ layerAbove->GetDirectFeedbackWeights()^T
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
      (int)layerAbove->GetPsiStride(),
      0.0f,
      proj,
      (int)nextSize);

  // The derivative applies at the source (mu's own nonlinearity),
  // same convention as UpdateWeights()'s real gradient -- keeps this
  // DFA approximation structurally consistent with it, even though
  // Psi/proj are never exact (see tDirectKPVerify.cpp's file header).
  {
    size_t Nout = (size_t)batchSize * nextSize;
    backend->ActivationDerivativeInto(ToDerivativeType(activationType), derivScratch, preact, Nout);
    backend->MultiplyInto(proj, proj, derivScratch, Nout);
  }

  // W += fl/batchSize * proj^T @ z
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

void DirectKPPCLayer::ClampState(const std::vector<float>& inputData) noexcept
{
  size_t copyFloats = (std::min)(inputData.size(), (size_t)(batchSize * size));
  backend->CopyFromHost(z, inputData.data(), copyFloats);
  isClamped = true;
  muCacheValid = false;
}

void DirectKPPCLayer::UnclampState() noexcept
{
  isClamped = false;
}

void DirectKPPCLayer::ResetState() noexcept
{
  size_t N = (size_t)batchSize * size;
  backend->Zero(z, N);
}

template void DirectKPPCLayer::BindMemory<MemoryArena>(MemoryArena& arena);
#if defined(DEEPITY_USE_CUDA)
template void DirectKPPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena& arena);
#endif
} // namespace Deep