#include "deepity/layers/ConvPCLayerT.h"
#include "deepity/backend/CPUBackendT.h"
#include "deepity/utils/LayerBindHelpers.h"
#include "deepity/utils/Optimize.h"
#include "deepity/utils/QuantFormat.h"
#include "deepity/utils/RandomizeDistribution.h"
#include <algorithm>
#include <cmath>
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
ConvPCLayerT<T, AccumT>::ConvPCLayerT(int inChannels, int outChannels, int inHeight, int inWidth,
                                      int kernelH, int kernelW, int strideH, int strideW, int padH,
                                      int padW, int batchSize, AccumT learningRate,
                                      AccumT inferenceRate, AccumT precisionRate, AccumT lmbda,
                                      ActivationType aType, ActivationType dType,
                                      IComputeBackendT<T, AccumT>* backend)
    : inChannels(inChannels)
    , outChannels(outChannels)
    , inHeight(inHeight)
    , inWidth(inWidth)
    , kernelH(kernelH)
    , kernelW(kernelW)
    , strideH(strideH)
    , strideW(strideW)
    , padH(padH)
    , padW(padW)
    , batchSize(batchSize)
    , lr(learningRate)
    , ir(inferenceRate)
    , pr(precisionRate)
    , lmbda(lmbda)
    , activationType(aType)
    , derivativeType(dType)
    , backend(backend ? backend : new CPUBackendT<T, AccumT>(),
              backend ? &NoOpDeleterT<T, AccumT> : &DeleteBackendT<T, AccumT>)
{
  outHeight = (outChannels > 0) ? ConvOutDim(inHeight, kernelH, strideH, padH) : 0;
  outWidth = (outChannels > 0) ? ConvOutDim(inWidth, kernelW, strideW, padW) : 0;
  DynamicThread(batchSize);

  localArena = std::make_unique<MemoryArenaT<T>>(GetRequiredElements());
  BindMemory(*localArena);
}

template <typename T, typename AccumT>
void ConvPCLayerT<T, AccumT>::SetLearningRate(AccumT learningRate) noexcept
{
  lr = learningRate;
  lr_device = lr;
}

template <typename T, typename AccumT>
size_t ConvPCLayerT<T, AccumT>::GetRequiredElements() const noexcept
{
  constexpr size_t kAlignElems = (64 + sizeof(T) - 1) / sizeof(T);
  auto pad16 = [](size_t n) { return (n + kAlignElems - 1) & ~(kAlignElems - 1); };

  size_t total = 0;
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  total += pad16(ownStateSize) * 3; // z, e, dz_dt
  total += pad16(ownSize) * 2;      // p, log_p

  if (outChannels > 0)
  {
    size_t outSize = (size_t)outChannels * outHeight * outWidth;
    size_t outStateSize = (size_t)batchSize * outSize;
    size_t colRows = (size_t)inChannels * kernelH * kernelW;
    size_t colCols = (size_t)outHeight * outWidth;
    size_t colSize = colRows * colCols;
    size_t Wsize = (size_t)outChannels * colRows;
    size_t M = (size_t)batchSize * colCols;

    total += pad16(Wsize);                           // W
    total += pad16((size_t)outChannels);             // b
    total += pad16(outStateSize) * 4;                // mu, cachedMu, preact, cachedPreact
    total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
    total += pad16(outStateSize);                    // bottom_up_cols
    total += pad16((size_t)batchSize * colSize);     // colsRepacked
    total += pad16(outStateSize);                    // lgRepacked
    total += pad16(M);                               // onesVector
  }

  return total;
}

template <typename T, typename AccumT>
template <typename ArenaT>
void ConvPCLayerT<T, AccumT>::BindMemory(ArenaT& arena)
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  z = arena.Allocate(ownStateSize);
  e = arena.Allocate(ownStateSize);
  dz_dt = arena.Allocate(ownStateSize);
  p = arena.Allocate(ownSize);
  log_p = arena.Allocate(ownSize);

  backend->Zero(z, ownStateSize);
  backend->Zero(e, ownStateSize);
  backend->Zero(dz_dt, ownStateSize);
  backend->Fill(p, ownSize, AccumT(1));
  backend->Zero(log_p, ownSize);

  if (outChannels > 0)
  {
    size_t colRows = (size_t)inChannels * kernelH * kernelW;
    size_t colCols = (size_t)outHeight * outWidth;
    size_t outStateSize = (size_t)batchSize * outChannels * colCols;
    size_t colSize = (size_t)batchSize * colRows * colCols;
    size_t Wsize = (size_t)outChannels * colRows;
    size_t M = (size_t)batchSize * colCols;

    W = arena.Allocate(Wsize);
    b = arena.Allocate(outChannels);
    mu = arena.Allocate(outStateSize);
    cachedMu = arena.Allocate(outStateSize);
    preact = arena.Allocate(outStateSize);
    cachedPreact = arena.Allocate(outStateSize);
    colBuffer = arena.Allocate(colSize);
    feedbackScratch = arena.Allocate(colSize);
    bottom_up_cols = arena.Allocate(outStateSize);
    colsRepacked = arena.Allocate(colSize);
    lgRepacked = arena.Allocate(outStateSize);
    onesVector = arena.Allocate(M);

    backend->Zero(b, outChannels);
    backend->Zero(mu, outStateSize);
    backend->Zero(cachedMu, outStateSize);
    backend->Zero(preact, outStateSize);
    backend->Zero(cachedPreact, outStateSize);
    backend->Zero(colBuffer, colSize);
    backend->Zero(feedbackScratch, colSize);
    backend->Zero(bottom_up_cols, outStateSize);
    backend->Zero(colsRepacked, colSize);
    backend->Zero(lgRepacked, outStateSize);
    backend->Fill(onesVector, M, AccumT(1));

    BindAdamStateT(Wsize, grad_W, m_W, v_W);
    BindAdamStateT((size_t)outChannels, grad_b, m_b, v_b);
    BindLearningRateStateT(t_device, lr_device, lr);
  }
  else
  {
    W = nullptr;
    b = nullptr;
    mu = nullptr;
    cachedMu = nullptr;
    preact = nullptr;
    cachedPreact = nullptr;
    colBuffer = nullptr;
    feedbackScratch = nullptr;
    bottom_up_cols = nullptr;
    colsRepacked = nullptr;
    lgRepacked = nullptr;
    onesVector = nullptr;
  }

  ReleaseLocalArenaIfForeignT<T>(localArena, arena);
}

template <typename T, typename AccumT>
void ConvPCLayerT<T, AccumT>::RandomizeWeights(std::mt19937& seedGenerator,
                                               const char* distribution) noexcept
{
  if (outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t Wsz = (size_t)outChannels * colRows;
  AccumT limit = std::sqrt(AccumT(2) / AccumT(colRows));

  std::uniform_int_distribution<uint32_t> seedDist;
  uint32_t seed = seedDist(seedGenerator);

  RandomizeWithDistributionT<T, AccumT>(backend.get(), W, Wsz, distribution, seed, AccumT(0),
                                        limit);
}

template <typename T, typename AccumT>
AccumT ConvPCLayerT<T, AccumT>::CalculateState(bool needEnergy) noexcept
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  if (layerBelow == nullptr)
  {
    backend->Zero(e, ownStateSize);
    AccumT totalEnergy(0);
    if (needEnergy)
      totalEnergy = AccumT(-0.5) * AccumT(batchSize) * backend->Sum(log_p, ownSize);
    if (outChannels > 0)
      ComputeMuOnly();
    return totalEnergy;
  }

  AccumT totalEnergy(0);
  if (needEnergy)
    totalEnergy =
        backend->ComputePrecisionWeightedErrorAndEnergy(e, z, layerBelow->mu, p, batchSize, ownSize);
  else
    backend->ComputePrecisionWeightedError(e, z, layerBelow->mu, p, batchSize, ownSize);

  if (outChannels > 0)
    ComputeMuOnly();

  return totalEnergy;
}

template <typename T, typename AccumT> void ConvPCLayerT<T, AccumT>::ComputeMuOnly() noexcept
{
  if (outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t colCols = (size_t)outHeight * outWidth;
  size_t Nout = (size_t)batchSize * outChannels * colCols;

  if (isClamped && muCacheValid)
  {
    backend->Copy(mu, cachedMu, Nout);
    backend->Copy(preact, cachedPreact, Nout);
    return;
  }

  if (!backend->TryFusedConvForward(z, W, mu, AccumT(1), batchSize, inChannels, inHeight, inWidth,
                                    outChannels, kernelH, kernelW, strideH, strideW, padH, padW))
  {
    backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW, strideH,
                    strideW, padH, padW, colBuffer);

    backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
    backend->MatMul(
        /*transA=*/false,
        /*transB=*/false, outChannels, (int)(batchSize * colCols), (int)colRows, AccumT(1), W,
        (int)colRows, colsRepacked, (int)(batchSize * colCols), AccumT(0), lgRepacked,
        (int)(batchSize * colCols));
    backend->RepackForBatchedGemm(mu, lgRepacked, outChannels, batchSize, colCols);
  }

  backend->AddBiasPerChannelNCHW(mu, b, batchSize, outChannels, colCols);

  backend->Copy(preact, mu, Nout);
  backend->Activation(activationType, mu, Nout);

  if (isClamped)
  {
    backend->Copy(cachedMu, mu, Nout);
    backend->Copy(cachedPreact, preact, Nout);
    muCacheValid = true;
  }
}

template <typename T, typename AccumT> void ConvPCLayerT<T, AccumT>::UpdateState() noexcept
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;
  size_t colCols = (outChannels > 0) ? (size_t)outHeight * outWidth : 0;
  size_t colRows = (outChannels > 0) ? (size_t)inChannels * kernelH * kernelW : 0;

  if (outChannels > 0)
  {
    size_t outTotal = (size_t)batchSize * outChannels * colCols;
    backend->ActivationDerivativeInto(derivativeType, mu, preact, outTotal);
  }

  if (isClamped)
    return;

  backend->Zero(dz_dt, ownStateSize);

  if (layerAbove != nullptr && outChannels > 0)
  {
    const T* e_above = layerAbove->GetErrors();
    const T* p_above = layerAbove->GetPrecisions();
    size_t outSize = (size_t)outChannels * colCols;

    backend->MultiplyBroadcastInto(bottom_up_cols, e_above, p_above, mu, batchSize, outSize);

    if (!backend->TryFusedConvBackwardData(bottom_up_cols, W, dz_dt, AccumT(1), batchSize,
                                           inChannels, inHeight, inWidth, outChannels, kernelH,
                                           kernelW, strideH, strideW, padH, padW))
    {
      backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);
      backend->MatMul(
          /*transA=*/true,
          /*transB=*/false, (int)colRows, (int)(batchSize * colCols), outChannels, AccumT(1), W,
          (int)colRows, lgRepacked, (int)(batchSize * colCols), AccumT(0), colsRepacked,
          (int)(batchSize * colCols));
      backend->RepackForBatchedGemm(feedbackScratch, colsRepacked, colRows, batchSize, colCols);

      backend->Col2Im(feedbackScratch, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW,
                      strideH, strideW, padH, padW, dz_dt);
    }
  }

  backend->AxpyBroadcastInto(dz_dt, e, p, batchSize, ownSize, AccumT(-1));
  backend->AxpyInto(z, dz_dt, ownStateSize, ir);
}

template <typename T, typename AccumT> void ConvPCLayerT<T, AccumT>::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t colCols = (size_t)outHeight * outWidth;
  size_t outSize = (size_t)outChannels * colCols;
  size_t Wsize = (size_t)outChannels * colRows;

  const T* e_above = layerAbove->GetErrors();
  const T* p_above = layerAbove->GetPrecisions();

  backend->MultiplyBroadcastInto(bottom_up_cols, e_above, p_above, mu, batchSize, outSize);
  backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);

  size_t M = (size_t)batchSize * colCols;

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > AccumT(0))
      backend->Scale(W, Wsize, AccumT(1) - lmbda);

    AccumT lr_batch = lr / AccumT(batchSize);

    // TryFusedConvBackwardFilter() is skipped here, same reasoning
    // as SimpleConvPCLayerT::UpdateWeights()'s identical comment --
    // always false on CPU, no CPU backend has a real fused conv path.
    backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW, strideH,
                    strideW, padH, padW, colBuffer);
    backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
    backend->MatMul(
        /*transA=*/false,
        /*transB=*/true, outChannels, (int)colRows, (int)M, lr_batch, lgRepacked, (int)M,
        colsRepacked, (int)M, AccumT(1), W, (int)colRows);

    backend->MatMul(
        /*transA=*/false,
        /*transB=*/false, outChannels, 1, (int)M, lr_batch, lgRepacked, (int)M, onesVector, 1,
        AccumT(1), b, 1);
    break;
  }
  case OptimizerType::ADAM:
  case OptimizerType::ADAMW:
  {
    backend->IncrementCounter(&t_device);

    AccumT grad_scale = AccumT(-1) / AccumT(batchSize);

    backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW, strideH,
                    strideW, padH, padW, colBuffer);
    backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
    backend->MatMulAccum(
        /*transA=*/false,
        /*transB=*/true, outChannels, (int)colRows, (int)M, grad_scale, lgRepacked, (int)M,
        colsRepacked, (int)M, AccumT(0), grad_W.data(), (int)colRows);

    backend->MatMulAccum(
        /*transA=*/false,
        /*transB=*/false, outChannels, 1, (int)M, grad_scale, lgRepacked, (int)M, onesVector, 1,
        AccumT(0), grad_b.data(), 1);

    if (opt == OptimizerType::ADAMW)
      backend->AdamWStep(W, grad_W.data(), m_W.data(), v_W.data(), Wsize, &t_device, &lr_device,
                         lmbda);
    else
      backend->AdamStep(W, grad_W.data(), m_W.data(), v_W.data(), Wsize, &t_device, &lr_device);

    backend->AdamStep(b, grad_b.data(), m_b.data(), v_b.data(), outChannels, &t_device,
                      &lr_device);
    break;
  }
  }
}

template <typename T, typename AccumT> void ConvPCLayerT<T, AccumT>::UpdatePrecision() noexcept
{
  if (layerBelow == nullptr)
    return;

  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  backend->UpdatePrecisionFromError(p, log_p, e, batchSize, ownSize, pr);
}

template <typename T, typename AccumT> void ConvPCLayerT<T, AccumT>::ResetState() noexcept
{
  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  backend->Zero(z, ownStateSize);
}

template <typename T, typename AccumT>
void ConvPCLayerT<T, AccumT>::ClampState(const std::vector<T>& inputData) noexcept
{
  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  size_t copyElems = (std::min)(inputData.size(), ownStateSize);
  backend->CopyFromHost(z, inputData.data(), copyElems);
  isClamped = true;
  muCacheValid = false;
}

template <typename T, typename AccumT> void ConvPCLayerT<T, AccumT>::UnclampState() noexcept
{
  isClamped = false;
}

template <typename T, typename AccumT>
void ConvPCLayerT<T, AccumT>::ResyncLogPrecision() noexcept
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  std::vector<T> hostP(ownSize);
  backend->CopyToHost(hostP.data(), p, ownSize);

  std::vector<T> hostLogP(ownSize);
  for (size_t i = 0; i < ownSize; ++i)
  {
    AccumT pv = (std::max)(hwy::ConvertScalarTo<AccumT>(hostP[i]), AccumT(1e-8));
    hostLogP[i] = hwy::ConvertScalarTo<T>(std::log(pv));
  }

  backend->CopyFromHost(log_p, hostLogP.data(), ownSize);
}

template class ConvPCLayerT<float, float>;
template class ConvPCLayerT<double, double>;
template class ConvPCLayerT<Fp4E2M1Value, float>;
template class ConvPCLayerT<hwy::float16_t, float>;
template class ConvPCLayerT<Posit4_0Value, float>;
template class ConvPCLayerT<Posit4_1Value, float>;
template class ConvPCLayerT<Posit4_2Value, float>;

template void ConvPCLayerT<float, float>::BindMemory<MemoryArenaT<float>>(
    MemoryArenaT<float>& arena);
template void ConvPCLayerT<double, double>::BindMemory<MemoryArenaT<double>>(
    MemoryArenaT<double>& arena);
template void ConvPCLayerT<Fp4E2M1Value, float>::BindMemory<MemoryArenaT<Fp4E2M1Value>>(
    MemoryArenaT<Fp4E2M1Value>& arena);
template void ConvPCLayerT<hwy::float16_t, float>::BindMemory<MemoryArenaT<hwy::float16_t>>(
    MemoryArenaT<hwy::float16_t>& arena);
template void ConvPCLayerT<Posit4_0Value, float>::BindMemory<MemoryArenaT<Posit4_0Value>>(
    MemoryArenaT<Posit4_0Value>& arena);
template void ConvPCLayerT<Posit4_1Value, float>::BindMemory<MemoryArenaT<Posit4_1Value>>(
    MemoryArenaT<Posit4_1Value>& arena);
template void ConvPCLayerT<Posit4_2Value, float>::BindMemory<MemoryArenaT<Posit4_2Value>>(
    MemoryArenaT<Posit4_2Value>& arena);
} // namespace Deep
