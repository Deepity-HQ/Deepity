#include "deepity/layers/FullPCLayerT.h"
#include "deepity/backend/CPUBackendT.h"
#include "deepity/utils/LayerBindHelpers.h"
#include "deepity/utils/Optimize.h"
#include "deepity/utils/QuantFormat.h"
#include "deepity/utils/RandomizeDistribution.h"
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
FullPCLayerT<T, AccumT>::FullPCLayerT(size_t size, size_t nextSize, size_t terminalSize,
                                      size_t batchSize, AccumT learningRate, AccumT inferenceRate,
                                      AccumT feedback, AccumT lmbda, ActivationType aType,
                                      ActivationType /*dType*/,
                                      IComputeBackendT<T, AccumT>* backend)
    : size(size)
    , nextSize(nextSize)
    , terminalSize(terminalSize)
    , batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , fl(feedback)
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
void FullPCLayerT<T, AccumT>::SetLearningRate(AccumT learningRate) noexcept
{
  lr = learningRate;
  // Direct write, not backend->CopyFromHost(): lr_device is AccumT*,
  // and CopyFromHost() takes T* -- see BindLearningRateStateT's own
  // doc comment on why these host-resident AccumT scratch writes go
  // around the backend rather than through it (valid today since
  // every templated backend is host-resident; see that comment for
  // the CUDA caveat).
  lr_device = lr;
}

template <typename T, typename AccumT>
void FullPCLayerT<T, AccumT>::SetFeedbackRate(AccumT feedbackRate) noexcept
{
  fl = feedbackRate;
  fl_device = fl;
}

template <typename T, typename AccumT>
template <typename ArenaT>
void FullPCLayerT<T, AccumT>::BindMemory(ArenaT& arena)
{
  size_t own_state_size = batchSize * size;
  size_t out_state_size = batchSize * nextSize;
  size_t direct_size = size * terminalSize;

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
    preact = arena.Allocate(out_state_size);
    cachedPreact = arena.Allocate(out_state_size);
    derivScratch = arena.Allocate(out_state_size);
    feedbackError = arena.Allocate(out_state_size);
    proj = arena.Allocate(out_state_size);
    Psi = arena.Allocate(direct_size);
    feedbackScratch = arena.Allocate(own_state_size);
    v = arena.Allocate(own_state_size);
    backend->Zero(v, own_state_size);
    adjoint = arena.Allocate(own_state_size);
    backend->Zero(adjoint, own_state_size);
    biasGradScratch = arena.Allocate(nextSize);

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

    BindAdamStateT(w_size, grad_W, m_W, v_W);
    BindAdamStateT(nextSize, grad_b, m_b, v_b);
    BindLearningRateStateT(t_device, lr_device, lr);

    BindAdamStateT(direct_size, grad_Psi, m_Psi, v_Psi);
    BindLearningRateStateT(tPsi_device, fl_device, fl);
  }

  ReleaseLocalArenaIfForeignT<T>(localArena, arena);
}

template <typename T, typename AccumT>
size_t FullPCLayerT<T, AccumT>::GetRequiredElements() const noexcept
{
  // Padding granularity in *elements*, not a hardcoded 16: Allocate()
  // below rounds every call up to a 64-BYTE boundary regardless of T,
  // so the element-count padding this must match is 64/sizeof(T) --
  // 16 for float, 8 for double, 64 for a 1-byte T like Fp4E2M1Value.
  // Getting this wrong undercounts the arena's real consumption for
  // any T smaller than 4 bytes and throws "capacity exceeded" at
  // runtime despite this function's own total looking sufficient.
  constexpr size_t kAlignElems = (64 + sizeof(T) - 1) / sizeof(T);
  auto pad16 = [](size_t n) { return (n + kAlignElems - 1) & ~(kAlignElems - 1); };

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
    total += pad16(out_state_size) * 3; // mu, cachedMu, proj
    total += pad16(out_state_size) * 4; // preact, cachedPreact, derivScratch, feedbackError
    total += pad16(own_state_size);     // feedbackScratch
    total += pad16(own_state_size) * 2; // v, adjoint
    total += pad16(direct_size);
    total += pad16(nextSize);

    total += pad16(w_size) * 3;
    total += pad16(nextSize) * 3;
    total += pad16(1) * 2;

    total += pad16(direct_size) * 3;
    total += pad16(1) * 2;
  }

  return total;
}

template <typename T, typename AccumT>
void FullPCLayerT<T, AccumT>::RandomizeWeights(std::mt19937& seedGenerator,
                                               const char* distribution) noexcept
{
  if (nextSize == 0)
    return;

  std::uniform_int_distribution<uint32_t> seedDist;
  size_t Wsz = size * nextSize;
  size_t Psisz = terminalSize * size;

  AccumT limit = useMuPCInit ? AccumT(1) : std::sqrt(AccumT(2) / AccumT(size + nextSize));
  AccumT limPsi = std::sqrt(AccumT(2) / AccumT(size + terminalSize));

  uint32_t seedW = seedDist(seedGenerator);
  uint32_t seedPsi = seedDist(seedGenerator);

  RandomizeWithDistributionT<T, AccumT>(backend.get(), W, Wsz, distribution, seedW, AccumT(0),
                                        limit);
  backend->RandomizeNormal(Psi, Psisz, AccumT(0), limPsi, seedPsi);
}

template <typename T, typename AccumT>
std::map<std::string, TensorDescriptorT<T>> FullPCLayerT<T, AccumT>::GetStateDict() const
{
  return {{"W", {W, std::vector<size_t>{nextSize, size}}},
          {"b", {b, std::vector<size_t>{nextSize}}},
          {"Psi", {Psi, std::vector<size_t>{nextSize, terminalSize}}}};
}

template <typename T, typename AccumT>
AccumT FullPCLayerT<T, AccumT>::CalculateState(bool needEnergy) noexcept
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

  return needEnergy ? totalEnergy : AccumT(0);
}

template <typename T, typename AccumT>
void FullPCLayerT<T, AccumT>::ComputeMuOnly() noexcept
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
      AccumT(0),
      mu,
      (int)nextSize);

  backend->AddBiasBroadcast(mu, b, batchSize, nextSize);

  backend->Copy(preact, mu, Nout);
  backend->Activation(activationType, mu, Nout);

  if (useResidual)
    backend->AxpyInto(mu, z, Nout, AccumT(1));

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    backend->Copy(cachedPreact, preact, Nout);
    muCacheValid = true;
  }
}

template <typename T, typename AccumT> void FullPCLayerT<T, AccumT>::UpdateState() noexcept
{
  size_t N = (size_t)batchSize * size;

  if (isClamped)
    return;

  if (useCrossEntropy)
    return;

  if (layerAbove != nullptr && nextSize > 0)
  {
    const T* e_above = layerAbove->GetErrors();

    bool topDownMeaningless = layerAbove->GetCrossEntropy() && !layerAbove->IsClamped();

    if (topDownMeaningless)
    {
      backend->AxpyInto(z, e, N, -ir);
    }
    else
    {
      size_t Nout = (size_t)batchSize * nextSize;

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
          AccumT(0),
          feedbackScratch,
          (int)size);

      if (useMomentum)
      {
        backend->AxpyInto(feedbackScratch, e, N, AccumT(-1));
        backend->Scale(v, N, momentumBeta);
        backend->AxpyInto(v, feedbackScratch, N, AccumT(1) - momentumBeta);
        backend->AxpyInto(z, v, N, ir);
      }
      else
      {
        backend->AxpyInto(feedbackScratch, e, N, AccumT(-1));
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

template <typename T, typename AccumT> void FullPCLayerT<T, AccumT>::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  size_t Nout = (size_t)batchSize * nextSize;
  const T* e_above = layerAbove->GetErrors();

  ActivationType dType = ToDerivativeType(activationType);
  backend->ActivationDerivativeInto(dType, derivScratch, preact, Nout);
  backend->MultiplyInto(feedbackError, e_above, derivScratch, Nout);
  const T* local_grad = feedbackError;

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > AccumT(0))
      backend->Scale(W, (size_t)nextSize * size, AccumT(1) - lmbda);

    backend->MatMul(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        (int)batchSize,
        a * lr / AccumT(batchSize),
        local_grad,
        (int)nextSize,
        z,
        (int)size,
        AccumT(1),
        W,
        (int)size);

    AccumT lr_batch = lr / AccumT(batchSize);
    backend->SumRows(biasGradScratch, local_grad, batchSize, nextSize);
    backend->AxpyInto(b, biasGradScratch, nextSize, lr_batch);

    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(&t_device);

    size_t num_weights = (size_t)nextSize * size;
    AccumT adam_scale = AccumT(-1) * a;

    backend->MatMulAccum(
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
        AccumT(0),
        grad_W.data(),
        (int)size);

    backend->SumRowsAccum(grad_b.data(), local_grad, batchSize, nextSize);
    // Direct negate, not backend->Scale(): grad_b is AccumT*, and
    // Scale() takes T* -- same host-resident-AccumT-scratch caveat as
    // SetLearningRate()'s own comment above.
    for (size_t i = 0; i < nextSize; ++i)
      grad_b[i] = -grad_b[i];

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
        fl / AccumT(batchSize),
        z,
        (int)size,
        terminalLayer->GetErrors(),
        (int)terminalSize,
        AccumT(1),
        Psi,
        (int)terminalSize);
    break;
  }
  case OptimizerType::ADAMW:
  case OptimizerType::ADAM:
  {
    backend->IncrementCounter(&tPsi_device);

    size_t num_weights_psi = size * terminalSize;
    AccumT adam_scale = AccumT(-1);

    backend->MatMulAccum(
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
        AccumT(0),
        grad_Psi.data(),
        (int)terminalSize);

    if (optPsi == OptimizerType::ADAMW)
      backend->AdamWStep(Psi, grad_Psi.data(), m_Psi.data(), v_Psi.data(), num_weights_psi,
                         &tPsi_device, &fl_device, lmbda);
    else
      backend->AdamStep(Psi, grad_Psi.data(), m_Psi.data(), v_Psi.data(), num_weights_psi,
                        &tPsi_device, &fl_device);

    break;
  }
  }
}

template <typename T, typename AccumT> void FullPCLayerT<T, AccumT>::DirectFeedbackUpdate() noexcept
{
  if (layerAbove == nullptr || layerAbove->GetDirectFeedbackWeights() == nullptr)
    return;

  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true,
      (int)batchSize,
      (int)nextSize,
      (int)terminalSize,
      AccumT(1),
      terminalLayer->GetErrors(),
      (int)terminalSize,
      layerAbove->GetDirectFeedbackWeights(),
      (int)terminalSize,
      AccumT(0),
      proj,
      (int)nextSize);

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
      fl / AccumT(batchSize),
      proj,
      (int)nextSize,
      z,
      (int)size,
      AccumT(1),
      W,
      (int)size);
}

template <typename T, typename AccumT> void FullPCLayerT<T, AccumT>::ReconstructBelief() noexcept
{
  if (layerBelow == nullptr)
    return;

  const size_t N = batchSize * size;
  backend->Copy(z, layerBelow->GetMu(), N);
  backend->AxpyInto(z, e, N, AccumT(1));
}

template <typename T, typename AccumT>
void FullPCLayerT<T, AccumT>::ComputeAdjoint(const T* adjointAbove,
                                             AccumT adjointAboveScale) noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  const size_t N = batchSize * size;
  const size_t Nout = batchSize * nextSize;

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
      AccumT(0),
      adjoint,
      (int)size);

  if (useResidual)
    backend->AxpyInto(adjoint, adjointAbove, N, adjointAboveScale);
}

template <typename T, typename AccumT> void FullPCLayerT<T, AccumT>::UpdateErrorEPC() noexcept
{
  if (isClamped)
    return;

  const size_t N = batchSize * size;
  backend->Scale(e, N, AccumT(1) - ir);
  backend->AxpyInto(e, adjoint, N, -ir);
}

template <typename T, typename AccumT>
void FullPCLayerT<T, AccumT>::ClampState(const std::vector<T>& inputData) noexcept
{
  size_t copyElements = (std::min)(inputData.size(), (size_t)(batchSize * size));
  backend->CopyFromHost(z, inputData.data(), copyElements);
  isClamped = true;
  muCacheValid = false;
}

template <typename T, typename AccumT> void FullPCLayerT<T, AccumT>::UnclampState() noexcept
{
  isClamped = false;
}

template <typename T, typename AccumT> void FullPCLayerT<T, AccumT>::ResetState() noexcept
{
  size_t N = (size_t)batchSize * size;
  backend->Zero(z, N);
  backend->Zero(e, N);
  if (v)
    backend->Zero(v, N);
}

template class FullPCLayerT<float, float>;
template void FullPCLayerT<float, float>::BindMemory<MemoryArenaT<float>>(
    MemoryArenaT<float>& arena);

template class FullPCLayerT<double, double>;
template void FullPCLayerT<double, double>::BindMemory<MemoryArenaT<double>>(
    MemoryArenaT<double>& arena);

// Fp4E2M1Value pairs with AccumT=float, never the default AccumT=T --
// see CPUBackendT.cpp's own explicit-instantiation comment for why.
template class FullPCLayerT<Fp4E2M1Value, float>;
template void FullPCLayerT<Fp4E2M1Value, float>::BindMemory<MemoryArenaT<Fp4E2M1Value>>(
    MemoryArenaT<Fp4E2M1Value>& arena);

// Same AccumT=float pairing for half precision -- see this plan's own
// "optimizer state is a second, fixed, wider type" design note; a
// float16 shadow accumulator would defeat the point of accumulating
// wider than storage.
template class FullPCLayerT<hwy::float16_t, float>;
template void FullPCLayerT<hwy::float16_t, float>::BindMemory<MemoryArenaT<hwy::float16_t>>(
    MemoryArenaT<hwy::float16_t>& arena);

// The three posit formats, same AccumT=float pairing as Fp4E2M1Value.
template class FullPCLayerT<Posit4_0Value, float>;
template void FullPCLayerT<Posit4_0Value, float>::BindMemory<MemoryArenaT<Posit4_0Value>>(
    MemoryArenaT<Posit4_0Value>& arena);

template class FullPCLayerT<Posit4_1Value, float>;
template void FullPCLayerT<Posit4_1Value, float>::BindMemory<MemoryArenaT<Posit4_1Value>>(
    MemoryArenaT<Posit4_1Value>& arena);

template class FullPCLayerT<Posit4_2Value, float>;
template void FullPCLayerT<Posit4_2Value, float>::BindMemory<MemoryArenaT<Posit4_2Value>>(
    MemoryArenaT<Posit4_2Value>& arena);

#if defined(DEEPITY_USE_CUDA)
template void FullPCLayerT<float, float>::BindMemory<DeviceMemoryArenaT<float, float>>(
    DeviceMemoryArenaT<float, float>& arena);
template void FullPCLayerT<double, double>::BindMemory<DeviceMemoryArenaT<double, double>>(
    DeviceMemoryArenaT<double, double>& arena);
#endif
} // namespace Deep
