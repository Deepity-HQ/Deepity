#include "deepity/layers/SimplePCLayerT.h"
#include "deepity/backend/CPUBackendT.h"
#include "deepity/utils/LayerBindHelpers.h"
#include "deepity/utils/Optimize.h"
#include "deepity/utils/QuantFormat.h"
#include "deepity/utils/RandomizeDistribution.h"
#include <algorithm>
#include <cstring>
#include <hwy/base.h>

namespace Deep
{
namespace
{
template <typename T, typename AccumT> void DeleteBackendT(IComputeBackendT<T, AccumT>* p)
{
  delete p;
}
template <typename T, typename AccumT> void NoOpDeleterT(IComputeBackendT<T, AccumT>*) {}
} // namespace

template <typename T, typename AccumT>
SimplePCLayerT<T, AccumT>::SimplePCLayerT(size_t size, size_t nextSize, size_t batchSize,
                                          AccumT learningRate, AccumT inferenceRate, AccumT lmbda,
                                          ActivationType aType, ActivationType /*dType*/,
                                          IComputeBackendT<T, AccumT>* backend)
    : size(size)
    , nextSize(nextSize)
    , batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , lmbda(lmbda)
    , activationType(aType)
    , backend(backend ? backend : new CPUBackendT<T, AccumT>(),
              backend ? &NoOpDeleterT<T, AccumT> : &DeleteBackendT<T, AccumT>)
{
  DynamicThread((int)batchSize);

  localArena = std::make_unique<MemoryArenaT<T>>(GetRequiredElements());
  BindMemory(*localArena);
}

template <typename T, typename AccumT>
void SimplePCLayerT<T, AccumT>::SetLearningRate(AccumT learningRate) noexcept
{
  lr = learningRate;
  // Direct write, not backend->CopyFromHost() -- see
  // FullPCLayerT::SetLearningRate()'s identical comment on why.
  lr_device = lr;
}

template <typename T, typename AccumT>
template <typename ArenaT>
void SimplePCLayerT<T, AccumT>::BindMemory(ArenaT& arena)
{
  size_t own_state_size = batchSize * size;
  size_t out_state_size = batchSize * nextSize;

  z = arena.Allocate(own_state_size);
  e = arena.Allocate(own_state_size);
  rowEnergies.assign(batchSize, AccumT(0));

  backend->Zero(z, own_state_size);
  backend->Zero(e, own_state_size);

  if (nextSize > 0)
  {
    size_t w_size = size * nextSize;

    W = arena.Allocate(w_size);
    b = arena.Allocate(nextSize);
    mu = arena.Allocate(out_state_size);
    cachedMu = arena.Allocate(out_state_size);
    zF = arena.Allocate(own_state_size);
    feedbackScratch.assign(own_state_size, AccumT(0));
    feedbackDerivScratch.assign(own_state_size, AccumT(0));

    backend->Zero(b, nextSize);
    backend->Zero(mu, out_state_size);
    backend->Zero(cachedMu, out_state_size);
    backend->Zero(zF, own_state_size);

    BindAdamStateT(w_size, grad_W, m_W, v_W);
    BindAdamStateT(nextSize, grad_b, m_b, v_b);
    BindLearningRateStateT(t_device, lr_device, lr);
  }

  ReleaseLocalArenaIfForeignT<T>(localArena, arena);
}

template <typename T, typename AccumT>
size_t SimplePCLayerT<T, AccumT>::GetRequiredElements() const noexcept
{
  // See FullPCLayerT::GetRequiredElements()'s identical comment on why
  // the padding granularity is in elements, not a hardcoded 16.
  constexpr size_t kAlignElems = (64 + sizeof(T) - 1) / sizeof(T);
  auto pad16 = [](size_t n) { return (n + kAlignElems - 1) & ~(kAlignElems - 1); };

  size_t total = 0;
  size_t own_state_size = (size_t)batchSize * size;

  total += pad16(own_state_size) * 2; // z, e

  if (nextSize > 0)
  {
    size_t out_state_size = (size_t)batchSize * nextSize;
    size_t w_size = (size_t)size * nextSize;

    total += pad16(w_size);
    total += pad16(nextSize);
    total += pad16(out_state_size) * 2; // mu, cachedMu
    total += pad16(own_state_size);     // zF

    // Adam state no longer lives in this T-typed arena at all (see
    // BindAdamStateT()'s own doc comment); kept here anyway, matching
    // FullPCLayerT::GetRequiredElements()'s identical, deliberately
    // generous padding for its own W/Psi Adam state.
    total += pad16(w_size) * 3;
    total += pad16(nextSize) * 3;
    total += pad16(1) * 2;
  }

  return total;
}

template <typename T, typename AccumT>
void SimplePCLayerT<T, AccumT>::RandomizeWeights(std::mt19937& seedGenerator,
                                                 const char* distribution) noexcept
{
  std::uniform_int_distribution<uint32_t> seedDist;
  size_t Wsz = size * nextSize;
  uint32_t seed = seedDist(seedGenerator);

  RandomizeWithDistributionT<T, AccumT>(backend.get(), W, Wsz, distribution, seed, AccumT(0),
                                        std::sqrt(AccumT(2) / AccumT(size + nextSize)));
}

template <typename T, typename AccumT>
AccumT SimplePCLayerT<T, AccumT>::CalculateState(bool needEnergy) noexcept
{
  const size_t N = batchSize * size;

  if (layerBelow == nullptr)
  {
    backend->Zero(e, N);
    if (nextSize > 0)
      ComputeMuOnly();
    return AccumT(0);
  }

  AccumT totalEnergy(0);

  if (useCrossEntropy)
  {
    if (needEnergy)
      totalEnergy = backend->ComputeSoftmaxCrossEntropyErrorAndEnergy(
          e, z, layerBelow->mu, batchSize, size, rowEnergies.data());
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

  return totalEnergy;
}

template <typename T, typename AccumT>
void SimplePCLayerT<T, AccumT>::ComputeMuOnly(bool zFAlreadyComputed) noexcept
{
  if (nextSize == 0)
    return;

  size_t Nout = batchSize * nextSize;
  size_t N = batchSize * size;

  if (isClamped && muCacheValid)
  {
    backend->Copy(mu, cachedMu, Nout);
    return;
  }

  if (!zFAlreadyComputed)
    backend->ActivationInto(activationType, zF, z, N);

  // mu = zF @ W^T + b, no further activation -- see SimplePCLayer's
  // own file comment on the mu=W@phi(z)+b convention.
  backend->MatMul(false, true, (int)batchSize, (int)nextSize, (int)size, AccumT(1), zF, (int)size,
                  W, (int)size, AccumT(0), mu, (int)nextSize);
  backend->AddBiasBroadcast(mu, b, batchSize, nextSize);

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    muCacheValid = true;
  }
}

template <typename T, typename AccumT> void SimplePCLayerT<T, AccumT>::UpdateState() noexcept
{
  size_t N = batchSize * size;

  if (isClamped)
    return;

  if (layerAbove != nullptr && nextSize > 0)
  {
    const T* e_above = layerAbove->GetErrors();

    // e_above projects through W raw, into AccumT (never T -- see this
    // file's own class comment); f'(z) applies AFTER this projection,
    // matching SimplePCLayer::UpdateState()'s own ordering.
    backend->MatMulAccum(
        /*transA=*/false,
        /*transB=*/false,
        (int)batchSize,
        (int)size,
        (int)nextSize,
        AccumT(1),
        e_above,
        (int)nextSize,
        W,
        (int)size,
        AccumT(0),
        feedbackScratch.data(),
        (int)size);

    backend->ActivationDerivativeIntoAccum(ToDerivativeType(activationType),
                                           feedbackDerivScratch.data(), z, N);

    // Pure AccumT*AccumT elementwise, both operands already decoded by
    // the calls above -- inlined rather than adding a backend method
    // for a product that never touches storage type T, same reasoning
    // as FullPCLayerT::DirectFeedbackUpdate()'s identical pattern.
    for (size_t i = 0; i < N; ++i)
      feedbackScratch[i] *= feedbackDerivScratch[i];

    // z += ir*((feedback*deriv) - e), same two-call pattern as
    // FullPCLayerT::UpdateState()'s non-momentum branch.
    backend->AxpyIntoAccum(feedbackScratch.data(), e, N, AccumT(-1));
    backend->AxpyIntoFromAccum(z, feedbackScratch.data(), N, ir);
  }
  else // Output layer
  {
    backend->AxpyInto(z, e, N, -ir);
  }
}

template <typename T, typename AccumT> void SimplePCLayerT<T, AccumT>::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  // No derivative multiply here -- mu=W@phi(z)+b has no activation of
  // its own for dW to chain through, so dE/dW is just e_above^T@zF
  // directly, and local_grad stays a raw T* the whole time (unlike
  // UpdateState()'s feedbackScratch above).
  const T* local_grad = layerAbove->GetErrors();

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > AccumT(0))
      backend->Scale(W, (size_t)nextSize * size, AccumT(1) - lmbda);

    backend->MatMul(true, false, (int)nextSize, (int)size, (int)batchSize,
                    lr / AccumT(batchSize), local_grad, (int)nextSize, zF, (int)size, AccumT(1), W,
                    (int)size);

    AccumT lr_batch = lr / AccumT(batchSize);
    for (size_t batch = 0; batch < batchSize; batch++)
      backend->AxpyInto(b, local_grad + batch * nextSize, nextSize, lr_batch);

    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(&t_device);

    size_t num_weights = (size_t)nextSize * size;
    AccumT grad_scale = AccumT(-1);

    backend->MatMulAccum(true, false, (int)nextSize, (int)size, (int)batchSize, grad_scale,
                         local_grad, (int)nextSize, zF, (int)size, AccumT(0), grad_W.data(),
                         (int)size);

    std::fill(grad_b.begin(), grad_b.begin() + nextSize, AccumT(0));
    for (size_t batch = 0; batch < batchSize; batch++)
      backend->AxpyIntoAccum(grad_b.data(), local_grad + batch * nextSize, nextSize, grad_scale);

    if (opt == OptimizerType::ADAMW)
      backend->AdamWStep(W, grad_W.data(), m_W.data(), v_W.data(), num_weights, &t_device,
                         &lr_device, lmbda);
    else
      backend->AdamStep(W, grad_W.data(), m_W.data(), v_W.data(), num_weights, &t_device,
                        &lr_device);

    backend->AdamStep(b, grad_b.data(), m_b.data(), v_b.data(), nextSize, &t_device, &lr_device);
    break;
  }
  }
}

template <typename T, typename AccumT> void SimplePCLayerT<T, AccumT>::ResetState() noexcept
{
  size_t N = (size_t)batchSize * size;
  backend->Zero(z, N);
}

template <typename T, typename AccumT>
void SimplePCLayerT<T, AccumT>::ClampState(const std::vector<T>& inputData) noexcept
{
  size_t copyElements = (std::min)(inputData.size(), (size_t)(batchSize * size));
  backend->CopyFromHost(z, inputData.data(), copyElements);
  isClamped = true;
  muCacheValid = false;
}

template <typename T, typename AccumT> void SimplePCLayerT<T, AccumT>::UnclampState() noexcept
{
  isClamped = false;
}

template class SimplePCLayerT<float, float>;
template void SimplePCLayerT<float, float>::BindMemory<MemoryArenaT<float>>(
    MemoryArenaT<float>& arena);

template class SimplePCLayerT<double, double>;
template void SimplePCLayerT<double, double>::BindMemory<MemoryArenaT<double>>(
    MemoryArenaT<double>& arena);

// AccumT=float, never the default AccumT=T -- see CPUBackendT.cpp's own
// explicit-instantiation comment for why.
template class SimplePCLayerT<Fp4E2M1Value, float>;
template void SimplePCLayerT<Fp4E2M1Value, float>::BindMemory<MemoryArenaT<Fp4E2M1Value>>(
    MemoryArenaT<Fp4E2M1Value>& arena);

template class SimplePCLayerT<hwy::float16_t, float>;
template void SimplePCLayerT<hwy::float16_t, float>::BindMemory<MemoryArenaT<hwy::float16_t>>(
    MemoryArenaT<hwy::float16_t>& arena);

template class SimplePCLayerT<Posit4_0Value, float>;
template void SimplePCLayerT<Posit4_0Value, float>::BindMemory<MemoryArenaT<Posit4_0Value>>(
    MemoryArenaT<Posit4_0Value>& arena);

template class SimplePCLayerT<Posit4_1Value, float>;
template void SimplePCLayerT<Posit4_1Value, float>::BindMemory<MemoryArenaT<Posit4_1Value>>(
    MemoryArenaT<Posit4_1Value>& arena);

template class SimplePCLayerT<Posit4_2Value, float>;
template void SimplePCLayerT<Posit4_2Value, float>::BindMemory<MemoryArenaT<Posit4_2Value>>(
    MemoryArenaT<Posit4_2Value>& arena);

#if defined(DEEPITY_USE_CUDA)
template void SimplePCLayerT<float, float>::BindMemory<DeviceMemoryArenaT<float, float>>(
    DeviceMemoryArenaT<float, float>& arena);
template void SimplePCLayerT<double, double>::BindMemory<DeviceMemoryArenaT<double, double>>(
    DeviceMemoryArenaT<double, double>& arena);

// Dead at runtime (CreateBackendT throws for DEVICE_GPU before any of
// these T's Compile() ever reaches this call) but still need to LINK --
// see FullPCLayerT.cpp's identical comment for why.
template void SimplePCLayerT<Fp4E2M1Value, float>::BindMemory<DeviceMemoryArenaT<Fp4E2M1Value, float>>(
    DeviceMemoryArenaT<Fp4E2M1Value, float>& arena);
template void
SimplePCLayerT<hwy::float16_t, float>::BindMemory<DeviceMemoryArenaT<hwy::float16_t, float>>(
    DeviceMemoryArenaT<hwy::float16_t, float>& arena);
template void SimplePCLayerT<Posit4_0Value, float>::BindMemory<DeviceMemoryArenaT<Posit4_0Value, float>>(
    DeviceMemoryArenaT<Posit4_0Value, float>& arena);
template void SimplePCLayerT<Posit4_1Value, float>::BindMemory<DeviceMemoryArenaT<Posit4_1Value, float>>(
    DeviceMemoryArenaT<Posit4_1Value, float>& arena);
template void SimplePCLayerT<Posit4_2Value, float>::BindMemory<DeviceMemoryArenaT<Posit4_2Value, float>>(
    DeviceMemoryArenaT<Posit4_2Value, float>& arena);
#endif
} // namespace Deep
