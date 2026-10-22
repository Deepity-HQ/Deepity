#include <algorithm>
#include <cmath>
#include <deepity/backend/CPUBackend.h>
#include <deepity/layers/DiscriminativePCLayer.h>
#include <deepity/utils/LayerBindHelpers.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/RandomizeDistribution.h>

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

DiscriminativePCLayer::DiscriminativePCLayer(int size, int nextSize, int batchSize,
                                             float learningRate, float inferenceRate,
                                             float precisionRate, float lmbda, ActivationType aType,
                                             ActivationType dType, IComputeBackend* backend)
    : batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , pr(precisionRate)
    , lmbda(lmbda)
    , activationType(aType)
    , derivativeType(dType)
    , backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
{
  this->size = size;
  this->nextSize = nextSize;
  DynamicThread(batchSize);

  localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
  BindMemory(*localArena);
}

void DiscriminativePCLayer::SetLearningRate(float learningRate) noexcept
{
  lr = learningRate;
  if (lr_device)
    backend->CopyFromHost(lr_device, &lr, 1);
}

void DiscriminativePCLayer::RandomizeWeights(std::mt19937& seedGenerator,
                                             const char* distribution) noexcept
{
  if (nextSize == 0)
    return;

  size_t Wsz = (size_t)size * nextSize;
  float limit = std::sqrt(2.0f / (size + nextSize));

  std::uniform_int_distribution<uint32_t> seedDist;
  uint32_t seed = seedDist(seedGenerator);

  RandomizeWithDistribution(backend.get(), W, Wsz, distribution, seed, 0.0f, limit);
}

float DiscriminativePCLayer::CalculateState(bool needEnergy) noexcept
{
  const size_t N = (size_t)batchSize * size;

  // See ConvPCLayer::CalculateState()'s identical comment: this
  // branch indexed log_p[i] from host code unconditionally before,
  // which is illegal once log_p is a GPU device pointer
  // (DeviceMemoryArena) -- fixed by routing through backend->Sum()
  // (a synchronous GPU readback, same constraint as
  // ComputePrecisionWeightedErrorAndEnergy below, so only computed
  // when actually needed).
  if (layerBelow == nullptr)
  {
    backend->Zero(e, N);
    float totalEnergy = 0.0f;
    if (needEnergy)
      totalEnergy = -0.5f * (float)batchSize * backend->Sum(log_p, size);

    if (nextSize > 0)
      ComputeMuOnly();
    return totalEnergy;
  }

  float totalEnergy = 0.0f;
  if (needEnergy)
    totalEnergy =
        backend->ComputePrecisionWeightedErrorAndEnergy(e, z, layerBelow->mu, p, batchSize, size);
  else
    backend->ComputePrecisionWeightedError(e, z, layerBelow->mu, p, batchSize, size);

  if (nextSize > 0)
    ComputeMuOnly();

  return totalEnergy;
}

void DiscriminativePCLayer::ComputeMuOnly() noexcept
{
  if (nextSize == 0)
    return;

  size_t Nout = (size_t)batchSize * nextSize;

  if (isClamped && muCacheValid)
  {
    backend->Copy(mu, cachedMu, Nout);
    backend->Copy(preact, cachedPreact, Nout);
    return;
  }

  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true,
      batchSize,
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

void DiscriminativePCLayer::UpdateState() noexcept
{
  size_t N = (size_t)batchSize * size;
  size_t Nout = (size_t)batchSize * nextSize;

  if (isClamped)
    return;

  if (layerAbove != nullptr && nextSize > 0)
  {
    const float* e_above = layerAbove->GetErrors();
    const float* p_above = layerAbove->GetPrecisions();

    // bottom_up[b,f] = e_above[b,f] * p_above[f] * f'(preact[b,f]).
    // The activation derivative applies at the source (mu's own
    // nonlinearity), before projecting through W, not after via f'(z).
    // Computed into derivScratch, never into mu itself -- see
    // derivScratch's own doc comment for why.
    backend->Zero(bottom_up, Nout);
    backend->AxpyBroadcastInto(bottom_up, e_above, p_above, batchSize, nextSize, 1.0f);
    backend->ActivationDerivativeInto(derivativeType, derivScratch, preact, Nout);
    backend->MultiplyInto(bottom_up, bottom_up, derivScratch, Nout);

    // feedbackScratch = bottom_up @ W (no transpose)
    backend->MatMul(
        /*transA=*/false,
        /*transB=*/false,
        batchSize,
        (int)size,
        (int)nextSize,
        1.0f,
        bottom_up,
        (int)nextSize,
        W,
        (int)size,
        0.0f,
        feedbackScratch,
        (int)size);

    // dz_dt = feedbackScratch, then dz_dt -= p*e
    backend->Copy(dz_dt, feedbackScratch, N);
    backend->AxpyBroadcastInto(dz_dt, e, p, batchSize, size, -1.0f);
  }
  else // Output layer
  {
    backend->Zero(dz_dt, N);
    backend->AxpyBroadcastInto(dz_dt, e, p, batchSize, size, -1.0f);
  }

  backend->AxpyInto(z, dz_dt, N, ir);
}

void DiscriminativePCLayer::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  const float* e_above = layerAbove->GetErrors();
  const float* p_above = layerAbove->GetPrecisions();
  size_t outN = (size_t)batchSize * nextSize;

  // local_grad[b,f] = e_above[b,f] * p_above[f] * f'(preact[b,f]),
  // recomputed fresh from preact -- not read back from a previous
  // UpdateState() call, see UpdateState()'s own comment on why mu
  // itself must never hold this.
  backend->Zero(bottom_up, outN);
  backend->AxpyBroadcastInto(bottom_up, e_above, p_above, batchSize, nextSize, 1.0f);
  backend->ActivationDerivativeInto(derivativeType, derivScratch, preact, outN);
  backend->MultiplyInto(bottom_up, bottom_up, derivScratch, outN);

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > 0.0f)
      backend->Scale(W, (size_t)nextSize * size, 1.0f - lmbda);

    float lr_batch = lr / batchSize;

    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        batchSize,
        lr_batch,
        bottom_up,
        (int)nextSize,
        z,
        (int)size,
        1.0f,
        W,
        (int)size);

    backend->SumRows(biasGradScratch, bottom_up, batchSize, nextSize);
    backend->AxpyInto(b, biasGradScratch, nextSize, lr_batch);
    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(t_device);

    float grad_scale = -1.0f;

    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        batchSize,
        grad_scale,
        bottom_up,
        (int)nextSize,
        z,
        (int)size,
        0.0f,
        grad_W,
        (int)size);

    backend->SumRows(grad_b, bottom_up, batchSize, nextSize);
    backend->Scale(grad_b, nextSize, grad_scale);

    if (opt == OptimizerType::ADAMW)
      backend->AdamWStep(W, grad_W, m_W, v_W, (size_t)nextSize * size, t_device, lr_device, lmbda);
    else
      backend->AdamStep(W, grad_W, m_W, v_W, (size_t)nextSize * size, t_device, lr_device);

    backend->AdamStep(b, grad_b, m_b, v_b, nextSize, t_device, lr_device);
    break;
  }
  }
}

void DiscriminativePCLayer::UpdatePrecision() noexcept
{
  if (layerBelow == nullptr)
    return;

  backend->UpdatePrecisionFromError(p, log_p, e, batchSize, size, pr);
}

void DiscriminativePCLayer::ResetState() noexcept
{
  backend->Zero(z, (size_t)batchSize * size);
}

void DiscriminativePCLayer::ClampState(const std::vector<float>& inputData) noexcept
{
  size_t copyFloats = (std::min)(inputData.size(), (size_t)batchSize * size);
  backend->CopyFromHost(z, inputData.data(), copyFloats);
  isClamped = true;
  muCacheValid = false;
}

void DiscriminativePCLayer::UnclampState() noexcept
{
  isClamped = false;
}

void DiscriminativePCLayer::ResyncLogPrecision() noexcept
{
  // Cold path (checkpoint load only), host round-trip, same
  // reasoning as ConvPCLayer::ResyncLogPrecision().
  std::vector<float> hostP(size);
  backend->CopyToHost(hostP.data(), p, size);

  std::vector<float> hostLogP(size);
  for (size_t i = 0; i < size; ++i)
    hostLogP[i] = std::log(std::max(hostP[i], 1e-8f));

  backend->CopyFromHost(log_p, hostLogP.data(), size);
}

size_t DiscriminativePCLayer::GetRequiredFloats() const noexcept
{
  auto pad16 = [](size_t n) { return (n + 15) & ~(size_t)15; };

  size_t total = 0;
  size_t ownStateSize = (size_t)batchSize * size;

  total += pad16(ownStateSize) * 3; // z, e, dz_dt
  total += pad16(ownStateSize);     // feedbackScratch
  total += pad16(size) * 2;         // p, log_p

  if (nextSize > 0)
  {
    size_t outStateSize = (size_t)batchSize * nextSize;
    size_t wSize = (size_t)size * nextSize;

    total += pad16(wSize);            // W
    total += pad16(nextSize);         // b
    total += pad16(outStateSize) * 6; // mu, bottom_up, cachedMu, preact, cachedPreact, derivScratch
    total += pad16(nextSize);         // biasGradScratch

    // Always allocated, regardless of optimizer, so switching to Adam/AdamW after Compile() stays
    // safe.
    total += pad16(wSize) * 3;    // grad_W, m_W, v_W
    total += pad16(nextSize) * 3; // grad_b, m_b, v_b
    total += pad16(1) * 2;        // t_device, lr_device
  }
  return total;
}

template <typename ArenaT> void DiscriminativePCLayer::BindMemory(ArenaT& arena)
{
  size_t ownStateSize = (size_t)batchSize * size;

  z = arena.AllocateFloats(ownStateSize);
  e = arena.AllocateFloats(ownStateSize);
  dz_dt = arena.AllocateFloats(ownStateSize);
  feedbackScratch = arena.AllocateFloats(ownStateSize);
  p = arena.AllocateFloats(size);
  log_p = arena.AllocateFloats(size);

  backend->Zero(z, ownStateSize);
  backend->Zero(e, ownStateSize);
  backend->Zero(dz_dt, ownStateSize);
  backend->Zero(feedbackScratch, ownStateSize);
  backend->Fill(p, size, 1.0f);
  backend->Zero(log_p, size);

  if (nextSize > 0)
  {
    size_t outStateSize = (size_t)batchSize * nextSize;
    size_t wSize = (size_t)size * nextSize;

    W = arena.AllocateFloats(wSize);
    b = arena.AllocateFloats(nextSize);
    mu = arena.AllocateFloats(outStateSize);
    bottom_up = arena.AllocateFloats(outStateSize);
    cachedMu = arena.AllocateFloats(outStateSize);
    preact = arena.AllocateFloats(outStateSize);
    cachedPreact = arena.AllocateFloats(outStateSize);
    derivScratch = arena.AllocateFloats(outStateSize);
    biasGradScratch = arena.AllocateFloats(nextSize);

    backend->Zero(b, nextSize);
    backend->Zero(mu, outStateSize);
    backend->Zero(bottom_up, outStateSize);
    backend->Zero(cachedMu, outStateSize);
    backend->Zero(preact, outStateSize);
    backend->Zero(cachedPreact, outStateSize);
    backend->Zero(derivScratch, outStateSize);
    backend->Zero(biasGradScratch, nextSize);

    // Always allocated, see GetRequiredFloats().
    BindAdamState(arena, backend.get(), wSize, grad_W, m_W, v_W);
    BindAdamState(arena, backend.get(), nextSize, grad_b, m_b, v_b);
    BindLearningRateState(arena, backend.get(), t_device, lr_device, lr);
  }
  else
  {
    W = nullptr;
    b = nullptr;
    mu = nullptr;
    bottom_up = nullptr;
    cachedMu = nullptr;
    preact = nullptr;
    cachedPreact = nullptr;
    derivScratch = nullptr;
    biasGradScratch = nullptr;
  }

  ReleaseLocalArenaIfForeign(localArena, arena);
}

std::map<std::string, TensorDescriptor> DiscriminativePCLayer::GetStateDict() const
{
  return {{"W", {W, {(size_t)nextSize, (size_t)size}}},
          {"b", {b, {(size_t)nextSize}}},
          {"p", {p, {(size_t)size}}}};
}

template void DiscriminativePCLayer::BindMemory<MemoryArena>(MemoryArena& arena);
#if defined(DEEPITY_USE_CUDA)
template void DiscriminativePCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena& arena);
#endif
} // namespace Deep
