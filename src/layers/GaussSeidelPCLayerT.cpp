#include "deepity/layers/GaussSeidelPCLayerT.h"
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
GaussSeidelPCLayerT<T, AccumT>::GaussSeidelPCLayerT(size_t size, size_t nextSize, size_t batchSize,
                                                    AccumT learningRate, AccumT inferenceRate,
                                                    AccumT lmbda, ActivationType aType,
                                                    ActivationType dType,
                                                    IComputeBackendT<T, AccumT>* backend)
    : size(size)
    , nextSize(nextSize)
    , batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , lmbda(lmbda)
    , activationType(aType)
    , derivativeType(dType)
    , backend(backend ? backend : new CPUBackendT<T, AccumT>(),
              backend ? &NoOpDeleterT<T, AccumT> : &DeleteBackendT<T, AccumT>)
{
  DynamicThread((int)batchSize);

  localArena = std::make_unique<MemoryArenaT<T>>(GetRequiredElements());
  BindMemory(*localArena);
}

template <typename T, typename AccumT>
void GaussSeidelPCLayerT<T, AccumT>::SetLearningRate(AccumT learningRate) noexcept
{
  lr = learningRate;
  // Direct write, not backend->CopyFromHost() -- see
  // FullPCLayerT::SetLearningRate()'s identical comment on why.
  lr_device = lr;
}

template <typename T, typename AccumT>
void GaussSeidelPCLayerT<T, AccumT>::ClampState(const std::vector<T>& inputData) noexcept
{
  size_t copyElements = (std::min)(inputData.size(), (size_t)(batchSize * size));
  backend->CopyFromHost(z, inputData.data(), copyElements);
  isClamped = true;
}

template <typename T, typename AccumT> void GaussSeidelPCLayerT<T, AccumT>::UnclampState() noexcept
{
  isClamped = false;
}

template <typename T, typename AccumT> void GaussSeidelPCLayerT<T, AccumT>::ResetState() noexcept
{
  size_t ownStateSize = batchSize * size;
  backend->Zero(z, ownStateSize);
  backend->Zero(e, ownStateSize);
  std::fill(dz_dt.begin(), dz_dt.end(), AccumT(0));

  if (nextSize > 0)
  {
    size_t outStateSize = batchSize * nextSize;
    backend->Zero(mu, outStateSize);
    backend->Zero(preact, outStateSize);
  }
  isClamped = false;
}

template <typename T, typename AccumT> void GaussSeidelPCLayerT<T, AccumT>::UpdateState() noexcept
{
  size_t ownStateSize = batchSize * size;

  if (isClamped)
    return;

  // Feedback goes through E (the independently-initialized,
  // never-updated feedback-alignment matrix), not W. The derivative
  // applies at the source (mu's own nonlinearity), to e_above using
  // the PREVIOUS timestep's preact (held over, not yet fresh for this
  // one -- same staleness as mu/e_above), before projecting through
  // E. derivScratch/feedbackError/dz_dt are AccumT -- see this file's
  // own class comment.
  if (layerAbove != nullptr && nextSize > 0)
  {
    size_t outSize = batchSize * nextSize;
    const T* e_above = layerAbove->GetErrors();

    backend->ActivationDerivativeIntoAccum(derivativeType, derivScratch.data(), preact, outSize);
    backend->MultiplyIntoAccum(feedbackError.data(), e_above, derivScratch.data(), outSize);

    backend->MatMulAccumMixed(
        /*transA=*/false,
        /*transB=*/false,
        (int)batchSize,
        (int)size,
        (int)nextSize,
        AccumT(1),
        feedbackError.data(),
        (int)nextSize,
        E,
        (int)size,
        AccumT(0),
        dz_dt.data(),
        (int)size);
  }
  else
  {
    std::fill(dz_dt.begin(), dz_dt.end(), AccumT(0));
  }

  backend->AxpyIntoAccum(dz_dt.data(), e, ownStateSize, AccumT(-1));
  backend->AxpyIntoFromAccum(z, dz_dt.data(), ownStateSize, ir);
}

template <typename T, typename AccumT>
void GaussSeidelPCLayerT<T, AccumT>::ComputePrediction() noexcept
{
  if (nextSize == 0)
    return;

  size_t Nout = batchSize * nextSize;

  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true,
      (int)batchSize,
      (int)nextSize,
      (int)size,
      AccumT(1),
      z,
      (int)size,
      W,
      (int)size,
      AccumT(0),
      mu,
      (int)nextSize);

  backend->AddBiasBroadcast(mu, b, batchSize, nextSize);

  // Snapshotted BEFORE Activation() overwrites mu in place, same
  // reasoning as the original class. mu still ends this function
  // clean and activated.
  backend->Copy(preact, mu, Nout);
  backend->Activation(activationType, mu, Nout);
}

template <typename T, typename AccumT>
AccumT GaussSeidelPCLayerT<T, AccumT>::ComputeError(bool needEnergy) noexcept
{
  size_t ownStateSize = batchSize * size;

  if (layerBelow == nullptr)
  {
    backend->Zero(e, ownStateSize);
    return AccumT(0);
  }

  if (needEnergy)
    return backend->ComputeErrorAndEnergy(e, z, layerBelow->GetMu(), ownStateSize);

  backend->ComputeError(e, z, layerBelow->GetMu(), ownStateSize);
  return AccumT(0);
}

template <typename T, typename AccumT> void GaussSeidelPCLayerT<T, AccumT>::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || nextSize == 0)
    return;

  size_t Nout = batchSize * nextSize;
  const T* e_above = layerAbove->GetErrors();

  // feedbackError := e_above * f'(preact), using the final preact from
  // this timestep's last ComputePrediction() call.
  backend->ActivationDerivativeIntoAccum(derivativeType, derivScratch.data(), preact, Nout);
  backend->MultiplyIntoAccum(feedbackError.data(), e_above, derivScratch.data(), Nout);
  const AccumT* local_grad = feedbackError.data();

  // local_grad's own per-batch-element reduction into grad_b: pure
  // AccumT->AccumT, no decode needed -- inlined rather than adding a
  // third backend method, same reasoning as
  // FullPCLayerT::UpdateWeights()'s identical sumRowsIntoGradB().
  auto sumRowsIntoGradB = [&]()
  {
    std::fill(grad_b.begin(), grad_b.begin() + nextSize, AccumT(0));
    for (size_t bi = 0; bi < batchSize; ++bi)
      for (size_t i = 0; i < nextSize; ++i)
        grad_b[i] += local_grad[bi * nextSize + i];
  };

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > AccumT(0))
      backend->Scale(W, (size_t)nextSize * size, AccumT(1) - lmbda);

    // Computed into grad_W/grad_b (this layer's own Adam accumulator
    // buffers, unused by SGD otherwise and free as scratch here) and
    // then applied via AxpyIntoFromAccum, rather than a single blended
    // MatMul() straight into W -- see FullPCLayerT::UpdateWeights()'s
    // identical SGD-branch comment: local_grad is AccumT now, and a
    // GEMM that blends an AccumT operand directly into an existing T*
    // accumulator (beta != 0) is a shape MatMulAccumMixed() doesn't
    // support.
    backend->MatMulAccumMixed(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        (int)batchSize,
        AccumT(1),
        local_grad,
        (int)nextSize,
        z,
        (int)size,
        AccumT(0),
        grad_W.data(),
        (int)size);
    backend->AxpyIntoFromAccum(W, grad_W.data(), (size_t)nextSize * size, lr / AccumT(batchSize));

    sumRowsIntoGradB();
    backend->AxpyIntoFromAccum(b, grad_b.data(), nextSize, lr / AccumT(batchSize));
    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(&t_device);

    AccumT grad_scale = AccumT(-1) / AccumT(batchSize);

    backend->MatMulAccumMixed(
        /*transA=*/true,
        /*transB=*/false,
        (int)nextSize,
        (int)size,
        (int)batchSize,
        grad_scale,
        local_grad,
        (int)nextSize,
        z,
        (int)size,
        AccumT(0),
        grad_W.data(),
        (int)size);

    sumRowsIntoGradB();
    for (size_t i = 0; i < nextSize; ++i)
      grad_b[i] *= grad_scale;

    if (opt == OptimizerType::ADAMW)
      backend->AdamWStep(W, grad_W.data(), m_W.data(), v_W.data(), (size_t)nextSize * size,
                         &t_device, &lr_device, lmbda);
    else
      backend->AdamStep(W, grad_W.data(), m_W.data(), v_W.data(), (size_t)nextSize * size,
                        &t_device, &lr_device);

    backend->AdamStep(b, grad_b.data(), m_b.data(), v_b.data(), nextSize, &t_device, &lr_device);
    break;
  }
  }
}

template <typename T, typename AccumT>
void GaussSeidelPCLayerT<T, AccumT>::RandomizeWeights(std::mt19937& twister,
                                                      const char* distribution) noexcept
{
  if (nextSize == 0)
    return;

  size_t Wsz = size * nextSize;
  AccumT limit = std::sqrt(AccumT(2) / AccumT(size + nextSize));

  std::uniform_int_distribution<uint32_t> seedDist;
  uint32_t seedW = seedDist(twister);
  RandomizeWithDistributionT<T, AccumT>(backend.get(), W, Wsz, distribution, seedW, AccumT(0),
                                        limit);

  // E: feedback-alignment matrix, an independent random draw with its
  // own seed, uniform +-0.3, not the Gaussian/Xavier-style limit W
  // uses above. Never touched again after this.
  uint32_t seedE = seedDist(twister);
  backend->RandomizeUniform(E, Wsz, AccumT(-0.3), AccumT(0.3), seedE);
}

template <typename T, typename AccumT>
size_t GaussSeidelPCLayerT<T, AccumT>::GetRequiredElements() const noexcept
{
  // See FullPCLayerT::GetRequiredElements()'s identical comment on why
  // the padding granularity is in elements, not a hardcoded 16.
  constexpr size_t kAlignElems = (64 + sizeof(T) - 1) / sizeof(T);
  auto pad16 = [](size_t n) { return (n + kAlignElems - 1) & ~(kAlignElems - 1); };

  size_t total = 0;
  size_t own_state_size = batchSize * size;

  total += pad16(own_state_size) * 2; // z, e

  if (nextSize > 0)
  {
    size_t w_size = size * nextSize;
    size_t out_state_size = batchSize * nextSize;

    total += pad16(w_size);           // W
    total += pad16(nextSize);         // b
    total += pad16(out_state_size) * 2; // mu, preact
    total += pad16(w_size);           // E (feedback alignment, same shape as W)

    // Adam state no longer lives in this T-typed arena at all (see
    // BindAdamStateT()'s own doc comment); kept here anyway, matching
    // FullPCLayerT/SimplePCLayerT's identical, deliberately generous
    // padding for their own Adam state.
    total += pad16(w_size) * 3;
    total += pad16(nextSize) * 3;
    total += pad16(1) * 2;
  }

  return total;
}

template <typename T, typename AccumT>
template <typename ArenaT>
void GaussSeidelPCLayerT<T, AccumT>::BindMemory(ArenaT& arena)
{
  size_t own_state_size = batchSize * size;

  z = arena.Allocate(own_state_size);
  e = arena.Allocate(own_state_size);
  dz_dt.assign(own_state_size, AccumT(0));

  backend->Zero(z, own_state_size);
  backend->Zero(e, own_state_size);

  if (nextSize > 0)
  {
    size_t w_size = size * nextSize;
    size_t out_state_size = batchSize * nextSize;

    W = arena.Allocate(w_size);
    b = arena.Allocate(nextSize);
    mu = arena.Allocate(out_state_size);
    preact = arena.Allocate(out_state_size);
    E = arena.Allocate(w_size);
    derivScratch.assign(out_state_size, AccumT(0));
    feedbackError.assign(out_state_size, AccumT(0));

    backend->Zero(b, nextSize);
    backend->Zero(mu, out_state_size);
    backend->Zero(preact, out_state_size);
    backend->Zero(E, w_size);

    BindAdamStateT(w_size, grad_W, m_W, v_W);
    BindAdamStateT(nextSize, grad_b, m_b, v_b);
    BindLearningRateStateT(t_device, lr_device, lr);
  }
  else
  {
    W = nullptr;
    b = nullptr;
    mu = nullptr;
    preact = nullptr;
    E = nullptr;
  }

  ReleaseLocalArenaIfForeignT<T>(localArena, arena);
}

template <typename T, typename AccumT>
std::map<std::string, TensorDescriptorT<T>> GaussSeidelPCLayerT<T, AccumT>::GetStateDict() const
{
  return {{"W", {W, std::vector<size_t>{nextSize, size}}},
          {"b", {b, std::vector<size_t>{nextSize}}},
          {"E", {E, std::vector<size_t>{nextSize, size}}}};
}

template class GaussSeidelPCLayerT<float, float>;
template void GaussSeidelPCLayerT<float, float>::BindMemory<MemoryArenaT<float>>(
    MemoryArenaT<float>& arena);

template class GaussSeidelPCLayerT<double, double>;
template void GaussSeidelPCLayerT<double, double>::BindMemory<MemoryArenaT<double>>(
    MemoryArenaT<double>& arena);

// AccumT=float, never the default AccumT=T -- see CPUBackendT.cpp's own
// explicit-instantiation comment for why.
template class GaussSeidelPCLayerT<Fp4E2M1Value, float>;
template void GaussSeidelPCLayerT<Fp4E2M1Value, float>::BindMemory<MemoryArenaT<Fp4E2M1Value>>(
    MemoryArenaT<Fp4E2M1Value>& arena);

template class GaussSeidelPCLayerT<hwy::float16_t, float>;
template void GaussSeidelPCLayerT<hwy::float16_t, float>::BindMemory<MemoryArenaT<hwy::float16_t>>(
    MemoryArenaT<hwy::float16_t>& arena);

template class GaussSeidelPCLayerT<Posit4_0Value, float>;
template void GaussSeidelPCLayerT<Posit4_0Value, float>::BindMemory<MemoryArenaT<Posit4_0Value>>(
    MemoryArenaT<Posit4_0Value>& arena);

template class GaussSeidelPCLayerT<Posit4_1Value, float>;
template void GaussSeidelPCLayerT<Posit4_1Value, float>::BindMemory<MemoryArenaT<Posit4_1Value>>(
    MemoryArenaT<Posit4_1Value>& arena);

template class GaussSeidelPCLayerT<Posit4_2Value, float>;
template void GaussSeidelPCLayerT<Posit4_2Value, float>::BindMemory<MemoryArenaT<Posit4_2Value>>(
    MemoryArenaT<Posit4_2Value>& arena);

#if defined(DEEPITY_USE_CUDA)
template void GaussSeidelPCLayerT<float, float>::BindMemory<DeviceMemoryArenaT<float, float>>(
    DeviceMemoryArenaT<float, float>& arena);
template void GaussSeidelPCLayerT<double, double>::BindMemory<DeviceMemoryArenaT<double, double>>(
    DeviceMemoryArenaT<double, double>& arena);

// Dead at runtime (CreateBackendT throws for DEVICE_GPU before any of
// these T's Compile() ever reaches this call) but still need to LINK --
// see FullPCLayerT.cpp's identical comment for why.
template void
GaussSeidelPCLayerT<Fp4E2M1Value, float>::BindMemory<DeviceMemoryArenaT<Fp4E2M1Value, float>>(
    DeviceMemoryArenaT<Fp4E2M1Value, float>& arena);
template void
GaussSeidelPCLayerT<hwy::float16_t, float>::BindMemory<DeviceMemoryArenaT<hwy::float16_t, float>>(
    DeviceMemoryArenaT<hwy::float16_t, float>& arena);
template void
GaussSeidelPCLayerT<Posit4_0Value, float>::BindMemory<DeviceMemoryArenaT<Posit4_0Value, float>>(
    DeviceMemoryArenaT<Posit4_0Value, float>& arena);
template void
GaussSeidelPCLayerT<Posit4_1Value, float>::BindMemory<DeviceMemoryArenaT<Posit4_1Value, float>>(
    DeviceMemoryArenaT<Posit4_1Value, float>& arena);
template void
GaussSeidelPCLayerT<Posit4_2Value, float>::BindMemory<DeviceMemoryArenaT<Posit4_2Value, float>>(
    DeviceMemoryArenaT<Posit4_2Value, float>& arena);
#endif
} // namespace Deep
