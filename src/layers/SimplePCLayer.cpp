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

// TEMPORARY profiling accumulators -- see SimplePCLayer.h's
// DebugGetErrorSeconds() & co. declarations.
double g_debugErrorSeconds = 0.0;
double g_debugMuOnlyElementwiseSeconds = 0.0;
double g_debugMuGemmSeconds = 0.0;
double g_debugFeedbackGemmSeconds = 0.0;
double g_debugWeightGradGemmSeconds = 0.0;
double g_debugUpdateStateElementwiseSeconds = 0.0;
} // namespace

double DebugGetErrorSeconds() noexcept { return g_debugErrorSeconds; }
double DebugGetMuOnlyElementwiseSeconds() noexcept { return g_debugMuOnlyElementwiseSeconds; }
double DebugGetMuGemmSeconds() noexcept { return g_debugMuGemmSeconds; }
double DebugGetFeedbackGemmSeconds() noexcept { return g_debugFeedbackGemmSeconds; }
double DebugGetWeightGradGemmSeconds() noexcept { return g_debugWeightGradGemmSeconds; }
double DebugGetUpdateStateElementwiseSeconds() noexcept
{
  return g_debugUpdateStateElementwiseSeconds;
}

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
    auto t0 = std::chrono::steady_clock::now();
    backend->Zero(e, N);
    auto t1 = std::chrono::steady_clock::now();
    g_debugErrorSeconds += std::chrono::duration<double>(t1 - t0).count();

    if (nextSize > 0)
      ComputeMuOnly();
    return 0.0f;
  }

  float totalEnergy = 0.0f;
  // ComputeMuOnly() is about to read z again right after this to compute
  // zF=f(z) for its own GEMM -- unless it's about to skip that entirely
  // via the mu cache-hit branch (see ComputeMuOnly()'s own early
  // return), in which case zF isn't needed at all right now and
  // computing it here would be pure waste.
  bool tryMergeActivation =
      nextSize > 0 && !useCrossEntropy && !needEnergy && !(isClamped && muCacheValid);
  bool mergedActivation = false;

  auto t0 = std::chrono::steady_clock::now();
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
  else if (tryMergeActivation &&
          backend->TryComputeErrorAndActivation(activationType, e, zF, z, layerBelow->mu, N))
  {
    mergedActivation = true;
  }
  else
  {
    backend->ComputeError(e, z, layerBelow->mu, N);
  }
  auto t1 = std::chrono::steady_clock::now();
  g_debugErrorSeconds += std::chrono::duration<double>(t1 - t0).count();

  if (nextSize > 0)
    ComputeMuOnly(/*zFAlreadyComputed=*/mergedActivation);

  return totalEnergy;
}

void SimplePCLayer::ComputeMuOnly(bool zFAlreadyComputed) noexcept
{
  if (nextSize == 0)
    return;

  size_t Nout = batchSize * nextSize;
  size_t N = batchSize * size;

  if (isClamped && muCacheValid)
  {
    auto t0 = std::chrono::steady_clock::now();
    backend->Copy(mu, cachedMu, Nout);
    auto t1 = std::chrono::steady_clock::now();
    g_debugMuOnlyElementwiseSeconds += std::chrono::duration<double>(t1 - t0).count();
    return;
  }

  auto t0 = std::chrono::steady_clock::now();
  // zF survives a later cache hit untouched: while clamped, z doesn't
  // change, so zF computed here stays valid for UpdateWeights() even
  // on settling steps that skip straight to the cache-hit branch above.
  if (!zFAlreadyComputed)
    backend->ActivationInto(activationType, zF, z, N);
  auto tAct = std::chrono::steady_clock::now();

  // mu = zF @ W^T + b, with no further activation -- see this class's
  // file comment on the mu=W@phi(z)+b convention. TryFusedMatMulBias
  // fuses the bias-add into the GEMM itself via oneDNN
  // (DEEPITY_USE_ONEDNN only; falls back to the separate calls
  // otherwise), since cblas_sgemm has no bias epilogue of its own.
  // TEMPORARY: timed separately from the activation above, since
  // TryFusedMatMulBias's oneDNN path bypasses CPUBackend::MatMul()
  // (and so DebugGetGemmSeconds() never sees this GEMM) -- see
  // DebugGetMuGemmSeconds()'s declaration.
  bool fused = backend->TryFusedMatMulBias(
      false, true, (int)batchSize, (int)nextSize, (int)size, zF, (int)size, W, (int)size, b, mu,
      (int)nextSize);
  if (!fused)
  {
    backend->MatMul(false,
                    true,
                    (int)batchSize,
                    (int)nextSize,
                    (int)size,
                    1.0f,
                    zF,
                    (int)size,
                    W,
                    (int)size,
                    0.0f,
                    mu,
                    (int)nextSize);
    backend->AddBiasBroadcast(mu, b, batchSize, nextSize);
  }
  auto tGemm = std::chrono::steady_clock::now();

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    muCacheValid = true;
  }
  auto t1 = std::chrono::steady_clock::now();
  g_debugMuOnlyElementwiseSeconds +=
      std::chrono::duration<double>(tAct - t0).count() + std::chrono::duration<double>(t1 - tGemm).count();
  g_debugMuGemmSeconds += std::chrono::duration<double>(tGemm - tAct).count();
}

void SimplePCLayer::UpdateState() noexcept
{
  size_t N = batchSize * size;

  if (isClamped)
    return;

  if (layerAbove != nullptr && nextSize > 0)
  {
    const float* e_above = layerAbove->GetErrors();

    // e_above projects through W raw -- no derivative pre-multiply
    // here, unlike the mu=f(W@z+b) convention: under mu=W@phi(z)+b,
    // the nonlinearity lives on z, so its derivative applies AFTER
    // this projection (inside FusedStateUpdate below), not before it.
    // TryFusedMatMul (oneDNN) is tried first: no bias to fuse here,
    // but its kernel may simply beat cblas_sgemm's per-call dispatch
    // overhead at this shape -- see IComputeBackend::TryFusedMatMul.
    // TEMPORARY: timed separately from MatMul()'s own g_debugGemmSeconds,
    // since the oneDNN path bypasses it entirely (same reason as
    // DebugGetMuGemmSeconds() above) -- see DebugGetFeedbackGemmSeconds().
    // TryFusedFeedbackUpdate fuses this GEMM directly into the
    // elementwise state update below, welding it on as oneDNN post-ops
    // on the SAME matmul primitive (DEEPITY_USE_ONEDNN, the default) or,
    // failing that, an AOT Halide kernel (DEEPITY_USE_HALIDE, off by
    // default -- verified correct but slower, kept only as a documented
    // dead end) -- see IComputeBackend::TryFusedFeedbackUpdate for which
    // mechanism actually ran and why. Falls back to the separate GEMM +
    // FusedStateUpdate pair otherwise.
    // TEMPORARY: both branches' timing is folded into the two existing
    // buckets below so the fused path's buckets stay comparable to the
    // baseline's in profiling output.
    auto tFeedbackGemm0 = std::chrono::steady_clock::now();
    if (backend->TryFusedFeedbackUpdate(ToDerivativeType(activationType), (int)batchSize,
                                        (int)size, (int)nextSize, e_above, W, z, e, ir,
                                        feedbackScratch, feedbackDerivScratch))
    {
      auto tFused1 = std::chrono::steady_clock::now();
      g_debugFeedbackGemmSeconds +=
          std::chrono::duration<double>(tFused1 - tFeedbackGemm0).count();
      return;
    }

    if (!backend->TryFusedMatMul(
            /*transA=*/false,
            /*transB=*/false,
            (int)batchSize,
            (int)size,
            (int)nextSize,
            1.0f,
            e_above,
            (int)nextSize,
            W,
            (int)size,
            feedbackScratch,
            (int)size))
    {
      backend->MatMul(
          /*transA=*/false,
          /*transB=*/false,
          (int)batchSize,
          (int)size,
          (int)nextSize,
          1.0f,
          e_above,
          (int)nextSize,
          W,
          (int)size,
          0.0f,
          feedbackScratch,
          (int)size);
    }
    auto tFeedbackGemm1 = std::chrono::steady_clock::now();
    g_debugFeedbackGemmSeconds +=
        std::chrono::duration<double>(tFeedbackGemm1 - tFeedbackGemm0).count();

    // z += ir*(f'(z)*feedbackScratch - e), one fused pass (handles its
    // own scalar fallback internally for activations needing a
    // transcendental derivative, e.g. SIGMOID -- see
    // CPUBackend::FusedStateUpdate).
    auto t0 = std::chrono::steady_clock::now();
    backend->FusedStateUpdate(z, feedbackScratch, ToDerivativeType(activationType), e, N, ir);
    auto t1 = std::chrono::steady_clock::now();
    g_debugUpdateStateElementwiseSeconds += std::chrono::duration<double>(t1 - t0).count();
  }
  else // Output Layer
  {
    // z[i] += ir * (-e[i])  ==  z += (-ir) * e, i.e. AxpyInto
    // with alpha = -ir.
    auto t0 = std::chrono::steady_clock::now();
    backend->AxpyInto(z, e, N, -ir);
    auto t1 = std::chrono::steady_clock::now();
    g_debugUpdateStateElementwiseSeconds += std::chrono::duration<double>(t1 - t0).count();
  }
}

void SimplePCLayer::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  // No derivative multiply here: mu=W@phi(z)+b has no activation of
  // its own for dW to chain through -- the nonlinearity already lives
  // on zF (consumed below), not on mu, so dE/dW is just e_above^T@zF
  // directly. Contrast UpdateState() above, where z (not W) is what's
  // being adjusted, and f'(z) is needed there.
  const float* local_grad = layerAbove->GetErrors();

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
                    zF,
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

    // TryFusedMatMul (oneDNN) first -- see UpdateState()'s feedback
    // GEMM above for why; this write is fresh (beta=0), so it fits the
    // same no-bias, no-accumulate contract. TEMPORARY timing -- see
    // DebugGetWeightGradGemmSeconds().
    auto tGradGemm0 = std::chrono::steady_clock::now();
    if (!backend->TryFusedMatMul(true,
                                 false,
                                 (int)nextSize,
                                 (int)size,
                                 (int)batchSize,
                                 grad_scale,
                                 local_grad,
                                 (int)nextSize,
                                 zF,
                                 (int)size,
                                 grad_W,
                                 (int)size))
    {
      backend->MatMul(true,
                      false,
                      (int)nextSize,
                      (int)size,
                      (int)batchSize,
                      grad_scale,
                      local_grad,
                      (int)nextSize,
                      zF,
                      (int)size,
                      0.0f,
                      grad_W,
                      (int)size);
    }
    auto tGradGemm1 = std::chrono::steady_clock::now();
    g_debugWeightGradGemmSeconds += std::chrono::duration<double>(tGradGemm1 - tGradGemm0).count();

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

  total += pad16(own_state_size) * 2; // z, e
  total += pad16(batchSize);          // rowEnergies

  if (nextSize > 0)
  {
    size_t out_state_size = (size_t)batchSize * nextSize;
    size_t w_size = (size_t)size * nextSize;

    total += pad16(w_size);
    total += pad16(nextSize);
    total += pad16(out_state_size) * 2; // mu, cachedMu
    total += pad16(own_state_size) * 3; // zF, feedbackScratch,
                                         // feedbackDerivScratch

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
  rowEnergies = arena.AllocateFloats(batchSize);

  backend->Zero(z, own_state_size);
  backend->Zero(e, own_state_size);
  backend->Zero(rowEnergies, batchSize);

  if (nextSize > 0)
  {
    size_t w_size = (size_t)size * nextSize;

    W = arena.AllocateFloats(w_size);
    b = arena.AllocateFloats(nextSize);
    mu = arena.AllocateFloats(out_state_size);
    cachedMu = arena.AllocateFloats(out_state_size);
    zF = arena.AllocateFloats(own_state_size);
    feedbackScratch = arena.AllocateFloats(own_state_size);
    feedbackDerivScratch = arena.AllocateFloats(own_state_size);

    backend->Zero(b, nextSize);
    backend->Zero(mu, out_state_size);
    backend->Zero(cachedMu, out_state_size);
    backend->Zero(zF, own_state_size);
    backend->Zero(feedbackScratch, own_state_size);
    backend->Zero(feedbackDerivScratch, own_state_size);

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
