#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <deepity/backend/CPUBackend.h>
#include <deepity/layers/SimplePCLayer.h>
#include <deepity/utils/LayerBindHelpers.h>
#include <deepity/utils/Optimize.h>
#include <deepity/utils/RandomizeDistribution.h>
#include <omp.h>

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

SimplePCLayer::SimplePCLayer(size_t size, size_t nextSize, size_t batchSize, float learningRate,
                             float inferenceRate, float lmbda, void (*act)(float*, size_t),
                             void (*dAct)(float*, size_t, bool), IComputeBackend* backend)
    : backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
    , batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , lmbda(lmbda)
    , isClamped(false)
    , layerAbove(nullptr)
    , layerBelow(nullptr)
    , activation(act)
    , activationDerivative(dAct)
    , activationType(To_AType(act))
    , opt(OptimizerType::SGD)
{
  this->size = size;
  this->nextSize = nextSize;
  activationDerivativeInto = To_dFn2(To_AType(dAct));
  DynamicThread(batchSize);

  localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
  BindMemory(*localArena);
}

SimplePCLayer::SimplePCLayer(size_t size, size_t nextSize, size_t batchSize, float learningRate,
                             float inferenceRate, float lmbda, ActivationType aType,
                             ActivationType dType, IComputeBackend* backend)
    : backend(backend ? backend : new CPUBackend(), backend ? NoOpDeleter : DeleteBackend)
    , batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , lmbda(lmbda)
    , isClamped(false)
    , layerAbove(nullptr)
    , layerBelow(nullptr)
    , activationType(aType)
{
  this->activation = To_Fn(aType);
  this->activationDerivative = To_dFn(dType);
  this->activationDerivativeInto = To_dFn2(dType);
  this->size = size;
  this->nextSize = nextSize;
  DynamicThread(batchSize);

  localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
  BindMemory(*localArena);
}

void SimplePCLayer::RandomizeWeights(std::mt19937& seedGenerator, const char* distribution) noexcept
{
  std::uniform_int_distribution<uint32_t> seedDist;
  size_t Wsz = size * nextSize;
  uint32_t seed = seedDist(seedGenerator);

  RandomizeWithDistribution(
      backend.get(), W, Wsz, distribution, seed, 0.0f, std::sqrt(2.0f / (size + nextSize)));
}

float SimplePCLayer::CalculateState(bool needEnergy) noexcept
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

void SimplePCLayer::ComputeMuOnly() noexcept
{
  if (nextSize == 0)
    return;

  size_t Nout = batchSize * nextSize;

  if (isClamped && muCacheValid)
  {
    backend->Copy(mu, cachedMu, Nout);
    backend->Copy(preact, cachedPreact, Nout);
    return;
  }

  backend->MatMul(false,
                  true,
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

  // Snapshotted BEFORE Activation() overwrites mu in place: GELU's
  // derivative has no closed form from the activated value alone, so
  // every later derivative computation (UpdateState()/UpdateWeights())
  // reads this instead of mu. Same reasoning as SimpleConvPCLayer's
  // own preact.
  backend->Copy(preact, mu, Nout);
  backend->Activation(activationType, mu, Nout);

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    backend->Copy(cachedPreact, preact, Nout);
    muCacheValid = true;
  }
}

void SimplePCLayer::UpdateState() noexcept
{
  size_t N = batchSize * size;
  size_t Nout = batchSize * nextSize;

  if (isClamped)
    return;

  if (layerAbove != nullptr && nextSize > 0)
  {
    const float* e_above = layerAbove->GetErrors();

    // The derivative applies at the source (mu=phi(W@z+b)'s own
    // nonlinearity), to e_above, before projecting it through W -- not
    // after, and not via f'(z). Computed into feedbackError, never
    // into mu itself: mu must stay clean, since ComputeMuOnly() is
    // publicly callable (e.g. ProjectForward()) between this and
    // UpdateWeights(), which would silently clobber a derivative
    // stashed in mu back into the plain activated value.
    backend->ActivationDerivativeInto(ToDerivativeType(activationType), feedbackError, preact, Nout);
    backend->MultiplyInto(feedbackError, e_above, feedbackError, Nout);

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
  else // Output Layer
  {
    // z[i] += ir * (-e[i])  ==  z += (-ir) * e, i.e. AxpyInto
    // with alpha = -ir.
    backend->AxpyInto(z, e, N, -ir);
  }
}

void SimplePCLayer::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  size_t Nout = batchSize * nextSize;
  const float* e_above = layerAbove->GetErrors();

  // Recomputed fresh from preact, not read back from a previous
  // UpdateState() call -- see UpdateState()'s own comment on why mu
  // itself must never hold this.
  backend->ActivationDerivativeInto(ToDerivativeType(activationType), feedbackError, preact, Nout);
  backend->MultiplyInto(feedbackError, e_above, feedbackError, Nout);
  const float* local_grad = feedbackError;

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > 0.0f)
      backend->Scale(W, (size_t)nextSize * size, 1.0f - lmbda);

    backend->MatMul(true,
                    false,
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
    for (int batch = 0; batch < batchSize; batch++)
      backend->AxpyInto(b, local_grad + batch * nextSize, nextSize, lr_batch);

    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(t_device);

    size_t num_weights = (size_t)nextSize * size;
    float grad_scale = -1.0f;

    backend->MatMul(true,
                    false,
                    (int)nextSize,
                    (int)size,
                    (int)batchSize,
                    grad_scale,
                    local_grad,
                    (int)nextSize,
                    z,
                    (int)size,
                    0.0f,
                    grad_W,
                    (int)size);

    backend->Zero(grad_b, nextSize);
    for (int batch = 0; batch < batchSize; batch++)
      backend->AxpyInto(grad_b, local_grad + batch * nextSize, nextSize, grad_scale);

    if (opt == OptimizerType::ADAMW)
      backend->AdamWStep(W, grad_W, m_W, v_W, num_weights, t_device, lr_device, lmbda);
    else
      backend->AdamStep(W, grad_W, m_W, v_W, num_weights, t_device, lr_device);

    backend->AdamStep(b, grad_b, m_b, v_b, nextSize, t_device, lr_device);
    break;
  }
  }
}

void SimplePCLayer::ResetState() noexcept
{
  size_t N = (size_t)batchSize * size;
  backend->Zero(z, N);
}

void SimplePCLayer::ClampState(const std::vector<float>& inputData) noexcept
{
  size_t copyFloats = (std::min)(inputData.size(), (size_t)(batchSize * size));
  backend->CopyFromHost(z, inputData.data(), copyFloats);
  isClamped = true;
  muCacheValid = false;
}

void SimplePCLayer::UnclampState() noexcept
{
  isClamped = false;
}

size_t SimplePCLayer::GetRequiredFloats() const noexcept
{
  auto pad16 = [](size_t n) { return (n + 15) & ~(size_t)15; };

  size_t total = 0;
  size_t own_state_size = (size_t)batchSize * size;

  total += pad16(own_state_size) * 2;

  if (nextSize > 0)
  {
    size_t out_state_size = (size_t)batchSize * nextSize;
    size_t w_size = (size_t)size * nextSize;

    total += pad16(w_size);
    total += pad16(nextSize);
    total += pad16(out_state_size) * 5; // mu, cachedMu, preact, cachedPreact, feedbackError
    total += pad16(own_state_size);     // feedbackScratch

    // Always allocated, regardless of optimizer, so switching to Adam/AdamW after Compile() stays
    // safe.
    total += pad16(w_size) * 3;
    total += pad16(nextSize) * 3;
    total += pad16(1) * 2; // t_device, lr_device
  }

  return total;
}

void SimplePCLayer::SetLearningRate(float lr) noexcept
{
  this->lr = lr;
  if (lr_device)
    backend->CopyFromHost(lr_device, &this->lr, 1);
}

template <typename ArenaT> void SimplePCLayer::BindMemory(ArenaT& arena)
{
  size_t own_state_size = (size_t)batchSize * size;
  size_t out_state_size = (size_t)batchSize * nextSize;

  z = arena.AllocateFloats(own_state_size);
  e = arena.AllocateFloats(own_state_size);

  backend->Zero(z, own_state_size);
  backend->Zero(e, own_state_size);

  if (nextSize > 0)
  {
    size_t w_size = (size_t)size * nextSize;

    W = arena.AllocateFloats(w_size);
    b = arena.AllocateFloats(nextSize);
    mu = arena.AllocateFloats(out_state_size);
    cachedMu = arena.AllocateFloats(out_state_size);
    preact = arena.AllocateFloats(out_state_size);
    cachedPreact = arena.AllocateFloats(out_state_size);
    feedbackError = arena.AllocateFloats(out_state_size);
    feedbackScratch = arena.AllocateFloats(own_state_size);

    backend->Zero(b, nextSize);
    backend->Zero(mu, out_state_size);
    backend->Zero(cachedMu, out_state_size);
    backend->Zero(preact, out_state_size);
    backend->Zero(cachedPreact, out_state_size);
    backend->Zero(feedbackError, out_state_size);
    backend->Zero(feedbackScratch, own_state_size);

    // Always allocated, see GetRequiredFloats().
    BindAdamState(arena, backend.get(), w_size, grad_W, m_W, v_W);
    BindAdamState(arena, backend.get(), nextSize, grad_b, m_b, v_b);
    BindLearningRateState(arena, backend.get(), t_device, lr_device, lr);
  }
  ReleaseLocalArenaIfForeign(localArena, arena);
}

template void SimplePCLayer::BindMemory<MemoryArena>(MemoryArena& arena);
#if defined(DEEPITY_USE_CUDA)
template void SimplePCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena& arena);
#endif
} // namespace Deep
