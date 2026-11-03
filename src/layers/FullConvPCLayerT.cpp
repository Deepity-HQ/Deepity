#include "deepity/layers/FullConvPCLayerT.h"
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
FullConvPCLayerT<T, AccumT>::FullConvPCLayerT(int inChannels, int outChannels, int inHeight,
                                              int inWidth, int kernelH, int kernelW, int strideH,
                                              int strideW, int padH, int padW, int terminalSize,
                                              int batchSize, AccumT learningRate,
                                              AccumT inferenceRate, AccumT feedback, AccumT lmbda,
                                              ActivationType aType, ActivationType dType,
                                              int poolH, int poolW, int poolStrideH,
                                              int poolStrideW, IComputeBackendT<T, AccumT>* backend)
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
    , terminalSize(terminalSize)
    , poolH(poolH)
    , poolW(poolW)
    , poolStrideH(poolStrideH)
    , poolStrideW(poolStrideW)
    , lr(learningRate)
    , ir(inferenceRate)
    , fl(feedback)
    , lmbda(lmbda)
    , activationType(aType)
    , derivativeType(dType)
    , backend(backend ? backend : new CPUBackendT<T, AccumT>(),
              backend ? &NoOpDeleterT<T, AccumT> : &DeleteBackendT<T, AccumT>)
{
  outHeight = (outChannels > 0) ? ConvOutDim(inHeight, kernelH, strideH, padH) : 0;
  outWidth = (outChannels > 0) ? ConvOutDim(inWidth, kernelW, strideW, padW) : 0;
  poolOutHeight =
      (outChannels > 0 && HasPooling()) ? PoolOutDim(outHeight, poolH, poolStrideH) : outHeight;
  poolOutWidth =
      (outChannels > 0 && HasPooling()) ? PoolOutDim(outWidth, poolW, poolStrideW) : outWidth;
  DynamicThread(batchSize);

  localArena = std::make_unique<MemoryArenaT<T>>(GetRequiredElements());
  BindMemory(*localArena);
}

template <typename T, typename AccumT>
void FullConvPCLayerT<T, AccumT>::SetLearningRate(AccumT learningRate) noexcept
{
  lr = learningRate;
  lr_device = lr;
}

template <typename T, typename AccumT>
void FullConvPCLayerT<T, AccumT>::SetFeedbackRate(AccumT feedbackRate) noexcept
{
  if ((feedbackRate == AccumT(0)) != (fl == AccumT(0)))
    flZeroDirty = true;

  fl = feedbackRate;
  fl_device = fl;
}

template <typename T, typename AccumT>
size_t FullConvPCLayerT<T, AccumT>::GetRequiredElements() const noexcept
{
  constexpr size_t kAlignElems = (64 + sizeof(T) - 1) / sizeof(T);
  auto pad16 = [](size_t n) { return (n + kAlignElems - 1) & ~(kAlignElems - 1); };

  size_t total = 0;
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  total += pad16(ownStateSize) * 3; // z, e, dz_dt

  if (outChannels > 0)
  {
    size_t outSize = (size_t)outChannels * outHeight * outWidth;
    size_t outStateSize = (size_t)batchSize * outSize;
    size_t colRows = (size_t)inChannels * kernelH * kernelW;
    size_t colCols = (size_t)outHeight * outWidth;
    size_t colSize = colRows * colCols;
    size_t Wsize = (size_t)outChannels * colRows;
    size_t M = (size_t)batchSize * colCols;
    size_t Psisize = (size_t)outChannels * terminalSize;

    total += pad16(Wsize);
    total += pad16((size_t)outChannels);             // b
    total += pad16(outStateSize) * 4;                // mu, cachedMu, preact, cachedPreact
    total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
    total += pad16(outStateSize);                    // bottom_up_cols
    total += pad16((size_t)batchSize * colSize);     // colsRepacked
    total += pad16(outStateSize);                    // lgRepacked
    total += pad16(M);                               // onesVector
    total += pad16(ownStateSize);                    // v (momentum)
    total += pad16(ownStateSize);                    // adjoint

    total += pad16(Wsize) * 3;               // grad_W, m_W, v_W
    total += pad16((size_t)outChannels) * 3; // grad_b, m_b, v_b

    total += pad16(Psisize);                         // Psi
    total += pad16((size_t)batchSize * outChannels); // projChannel
    total += pad16(outStateSize);                    // proj

    if (HasPooling())
    {
      size_t poolOutStateSize = (size_t)batchSize * outChannels * poolOutHeight * poolOutWidth;
      total += pad16(poolOutStateSize);     // pooledMu
      total += pad16(outStateSize);         // unpoolScratch
    }
  }

  return total;
}

template <typename T, typename AccumT>
template <typename ArenaT>
void FullConvPCLayerT<T, AccumT>::BindMemory(ArenaT& arena)
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  z = arena.Allocate(ownStateSize);
  e = arena.Allocate(ownStateSize);
  dz_dt = arena.Allocate(ownStateSize);
  rowEnergies.assign(batchSize, AccumT(0));

  backend->Zero(z, ownStateSize);
  backend->Zero(e, ownStateSize);
  backend->Zero(dz_dt, ownStateSize);

  if (outChannels > 0)
  {
    size_t colRows = (size_t)inChannels * kernelH * kernelW;
    size_t colCols = (size_t)outHeight * outWidth;
    size_t outStateSize = (size_t)batchSize * outChannels * colCols;
    size_t colSize = (size_t)batchSize * colRows * colCols;
    size_t Wsize = (size_t)outChannels * colRows;
    size_t M = (size_t)batchSize * colCols;
    size_t Psisize = (size_t)outChannels * terminalSize;

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
    v = arena.Allocate(ownStateSize);
    adjoint = arena.Allocate(ownStateSize);

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
    backend->Zero(v, ownStateSize);
    backend->Zero(adjoint, ownStateSize);

    BindAdamStateT(Wsize, grad_W, m_W, v_W);
    BindAdamStateT((size_t)outChannels, grad_b, m_b, v_b);
    BindLearningRateStateT(t_device, lr_device, lr);

    Psi = arena.Allocate(Psisize);
    projChannel = arena.Allocate((size_t)batchSize * outChannels);
    proj = arena.Allocate(outStateSize);

    backend->Zero(Psi, Psisize);
    backend->Zero(projChannel, (size_t)batchSize * outChannels);
    backend->Zero(proj, outStateSize);

    BindAdamStateT(Psisize, grad_Psi, m_Psi, v_Psi);
    BindLearningRateStateT(tPsi_device, fl_device, fl);

    if (HasPooling())
    {
      size_t poolOutStateSize = (size_t)batchSize * outChannels * poolOutHeight * poolOutWidth;
      pooledMu = arena.Allocate(poolOutStateSize);
      poolArgmax.assign(poolOutStateSize, 0);
      unpoolScratch = arena.Allocate(outStateSize);

      backend->Zero(pooledMu, poolOutStateSize);
      backend->Zero(unpoolScratch, outStateSize);
    }
  }

  ReleaseLocalArenaIfForeignT<T>(localArena, arena);
}

template <typename T, typename AccumT>
void FullConvPCLayerT<T, AccumT>::RandomizeWeights(std::mt19937& seedGenerator,
                                                   const char* distribution) noexcept
{
  if (outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t Wsz = (size_t)outChannels * colRows;
  size_t Psisz = (size_t)outChannels * terminalSize;

  AccumT limPsi = std::sqrt(AccumT(2) / AccumT(outChannels + terminalSize));

  std::uniform_int_distribution<uint32_t> seedDist;
  uint32_t seedW = seedDist(seedGenerator);
  uint32_t seedPsi = seedDist(seedGenerator);

  if (usePCXInit)
  {
    AccumT pcxLimit = std::sqrt(AccumT(1) / AccumT(colRows));
    backend->RandomizeUniform(W, Wsz, -pcxLimit, pcxLimit, seedW);
  }
  else
  {
    AccumT limit = useMuPCInit ? AccumT(1) : std::sqrt(AccumT(2) / AccumT(colRows));
    RandomizeWithDistributionT<T, AccumT>(backend.get(), W, Wsz, distribution, seedW, AccumT(0),
                                          limit);
  }
  backend->RandomizeNormal(Psi, Psisz, AccumT(0), limPsi, seedPsi);
}

template <typename T, typename AccumT>
AccumT FullConvPCLayerT<T, AccumT>::CalculateState(bool needEnergy) noexcept
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;

  if (layerBelow == nullptr)
  {
    backend->Zero(e, ownStateSize);
    if (outChannels > 0)
      ComputeMuOnly();
    return AccumT(0);
  }

  AccumT totalEnergy(0);
  if (useCrossEntropy)
  {
    if (needEnergy)
      totalEnergy = backend->ComputeSoftmaxCrossEntropyErrorAndEnergy(
          e, z, layerBelow->GetMu(), batchSize, ownSize, rowEnergies.data());
    else
      backend->ComputeSoftmaxCrossEntropyError(e, z, layerBelow->GetMu(), batchSize, ownSize);
  }
  else if (needEnergy)
  {
    totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->GetMu(), ownStateSize);
  }
  else
  {
    backend->ComputeError(e, z, layerBelow->GetMu(), ownStateSize);
  }

  if (outChannels > 0)
    ComputeMuOnly();

  return needEnergy ? totalEnergy : AccumT(0);
}

template <typename T, typename AccumT> void FullConvPCLayerT<T, AccumT>::ComputeMuOnly() noexcept
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
    if (HasPooling())
      backend->MaxPool2DForward(mu, batchSize, outChannels, outHeight, outWidth, poolH, poolW,
                                poolStrideH, poolStrideW, pooledMu, poolArgmax.data());
    return;
  }

  if (!backend->TryFusedConvForward(z, W, mu, a, batchSize, inChannels, inHeight, inWidth,
                                    outChannels, kernelH, kernelW, strideH, strideW, padH, padW))
  {
    backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW, strideH,
                    strideW, padH, padW, colBuffer);

    backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
    backend->MatMul(
        /*transA=*/false,
        /*transB=*/false, outChannels, (int)(batchSize * colCols), (int)colRows, a, W,
        (int)colRows, colsRepacked, (int)(batchSize * colCols), AccumT(0), lgRepacked,
        (int)(batchSize * colCols));
    backend->RepackForBatchedGemm(mu, lgRepacked, outChannels, batchSize, colCols);
  }

  backend->AddBiasPerChannelNCHW(mu, b, batchSize, outChannels, colCols);
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

  if (HasPooling())
    backend->MaxPool2DForward(mu, batchSize, outChannels, outHeight, outWidth, poolH, poolW,
                              poolStrideH, poolStrideW, pooledMu, poolArgmax.data());
}

template <typename T, typename AccumT>
const T* FullConvPCLayerT<T, AccumT>::UnpoolError(const T* pooledError) noexcept
{
  if (!HasPooling())
    return pooledError;

  size_t outStateSize = (size_t)batchSize * outChannels * outHeight * outWidth;

  backend->Zero(unpoolScratch, outStateSize);
  backend->MaxPool2DBackward(pooledError, poolArgmax.data(), batchSize, outChannels, outHeight, outWidth,
                             poolOutHeight, poolOutWidth, unpoolScratch);
  return unpoolScratch;
}

template <typename T, typename AccumT> void FullConvPCLayerT<T, AccumT>::UpdateState() noexcept
{
  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t ownStateSize = (size_t)batchSize * ownSize;
  size_t colCols = (outChannels > 0) ? (size_t)outHeight * outWidth : 0;
  size_t colRows = (outChannels > 0) ? (size_t)inChannels * kernelH * kernelW : 0;

  bool canFuse = (layerAbove != nullptr && outChannels > 0 && !isClamped);

  if (outChannels > 0 && !canFuse)
  {
    size_t outTotal = (size_t)batchSize * outChannels * colCols;
    backend->ActivationDerivativeInto(derivativeType, mu, preact, outTotal);
  }

  if (isClamped)
    return;

  backend->Zero(dz_dt, ownStateSize);

  if (layerAbove != nullptr && outChannels > 0)
  {
    const T* e_above = UnpoolError(layerAbove->GetErrors());
    size_t outSize = (size_t)outChannels * colCols;
    size_t outTotal = (size_t)batchSize * outSize;

    backend->ActivationDerivativeInto(derivativeType, mu, preact, outTotal);
    backend->MultiplyInto(bottom_up_cols, e_above, mu, outTotal);
    backend->Scale(bottom_up_cols, outTotal, a);
    backend->Scale(mu, outTotal, a);

    if (!backend->TryFusedConvBackwardData(bottom_up_cols, W, dz_dt, a, batchSize, inChannels,
                                           inHeight, inWidth, outChannels, kernelH, kernelW,
                                           strideH, strideW, padH, padW))
    {
      backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);
      backend->MatMul(
          /*transA=*/true,
          /*transB=*/false, (int)colRows, (int)(batchSize * colCols), outChannels, a, W,
          (int)colRows, lgRepacked, (int)(batchSize * colCols), AccumT(0), colsRepacked,
          (int)(batchSize * colCols));
      backend->RepackForBatchedGemm(feedbackScratch, colsRepacked, colRows, batchSize, colCols);

      backend->Col2Im(feedbackScratch, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW,
                      strideH, strideW, padH, padW, dz_dt);
    }
  }

  backend->AxpyInto(dz_dt, e, ownStateSize, AccumT(-1));

  if (useMomentum)
  {
    backend->Scale(v, ownStateSize, momentumBeta);
    backend->AxpyInto(v, dz_dt, ownStateSize, AccumT(1));
    backend->AxpyInto(z, v, ownStateSize, ir);
  }
  else
  {
    backend->AxpyInto(z, dz_dt, ownStateSize, ir);
  }
}

template <typename T, typename AccumT> void FullConvPCLayerT<T, AccumT>::UpdateWeights() noexcept
{
  if (layerAbove == nullptr || outChannels == 0)
    return;

  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t colCols = (size_t)outHeight * outWidth;
  size_t outSize = (size_t)outChannels * colCols;
  size_t Wsize = (size_t)outChannels * colRows;
  size_t outTotal = (size_t)batchSize * outSize;

  const T* e_above = UnpoolError(layerAbove->GetErrors());

  backend->MultiplyInto(bottom_up_cols, e_above, mu, outTotal);
  backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);

  size_t M = (size_t)batchSize * colCols;

  switch (opt)
  {
  case OptimizerType::SGD:
  {
    if (lmbda > AccumT(0))
      backend->Scale(W, Wsize, AccumT(1) - lmbda);

    AccumT lr_batch = lr / AccumT(batchSize);

    if (!backend->TryFusedConvBackwardFilter(z, bottom_up_cols, W, lr_batch, AccumT(1), batchSize,
                                             inChannels, inHeight, inWidth, outChannels, kernelH,
                                             kernelW, strideH, strideW, padH, padW))
    {
      backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW, strideH,
                      strideW, padH, padW, colBuffer);
      backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
      backend->MatMul(
          /*transA=*/false,
          /*transB=*/true, outChannels, (int)colRows, (int)M, lr_batch, lgRepacked, (int)M,
          colsRepacked, (int)M, AccumT(1), W, (int)colRows);
    }

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

    // TryFusedConvBackwardFilter() is skipped here, same reasoning as
    // SimpleConvPCLayerT::UpdateWeights()'s identical comment --
    // always false on CPU, and grad_W is AccumT* (the wider Adam
    // accumulator), which the fused entry point's T*-typed dW
    // parameter couldn't accept for a lossy T regardless.
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
                         lmbda, AccumT(0.9), AccumT(0.999), adamEpsilon);
    else
      backend->AdamStep(W, grad_W.data(), m_W.data(), v_W.data(), Wsize, &t_device, &lr_device,
                        AccumT(0.9), AccumT(0.999), adamEpsilon);

    backend->AdamStep(b, grad_b.data(), m_b.data(), v_b.data(), outChannels, &t_device,
                      &lr_device, AccumT(0.9), AccumT(0.999), adamEpsilon);
    break;
  }
  }
}

template <typename T, typename AccumT>
void FullConvPCLayerT<T, AccumT>::DirectFeedbackUpdate() noexcept
{
  if (layerAbove == nullptr || layerAbove->GetDirectFeedbackWeights() == nullptr)
    return;

  if (fl == AccumT(0))
    return;

  size_t colCols = (size_t)outHeight * outWidth;
  size_t colRows = (size_t)inChannels * kernelH * kernelW;

  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true, batchSize, outChannels, terminalSize, AccumT(1),
      terminalLayer->GetErrors(), terminalSize, layerAbove->GetDirectFeedbackWeights(),
      terminalSize, AccumT(0), projChannel, outChannels);

  backend->Zero(proj, (size_t)batchSize * outChannels * colCols);
  for (int batch = 0; batch < batchSize; ++batch)
  {
    T* proj_item = proj + (size_t)batch * outChannels * colCols;
    const T* projChannel_item = projChannel + (size_t)batch * outChannels;
    backend->AddBiasPerChannel(proj_item, projChannel_item, outChannels, colCols);
  }

  backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
  backend->RepackForBatchedGemm(lgRepacked, proj, batchSize, outChannels, colCols);

  size_t M = (size_t)batchSize * colCols;
  backend->MatMul(
      /*transA=*/false,
      /*transB=*/true, outChannels, (int)colRows, (int)M, fl / AccumT(batchSize), lgRepacked,
      (int)M, colsRepacked, (int)M, AccumT(1), W, (int)colRows);
}

template <typename T, typename AccumT> void FullConvPCLayerT<T, AccumT>::ReconstructBelief() noexcept
{
  if (layerBelow == nullptr)
    return;

  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  backend->Copy(z, layerBelow->GetMu(), ownStateSize);
  backend->AxpyInto(z, e, ownStateSize, AccumT(1));
}

template <typename T, typename AccumT>
void FullConvPCLayerT<T, AccumT>::ComputeAdjoint(const T* adjointAbove,
                                                 AccumT adjointAboveScale) noexcept
{
  if (layerAbove == nullptr || outChannels == 0)
    return;

  adjointAbove = UnpoolError(adjointAbove);

  size_t ownSize = (size_t)inChannels * inHeight * inWidth;
  size_t colRows = (size_t)inChannels * kernelH * kernelW;
  size_t colCols = (size_t)outHeight * outWidth;
  size_t outTotal = (size_t)batchSize * outChannels * colCols;

  backend->ActivationDerivativeInto(derivativeType, mu, preact, outTotal);
  backend->MultiplyInto(bottom_up_cols, adjointAbove, mu, outTotal);
  if (adjointAboveScale != AccumT(1))
    backend->Scale(bottom_up_cols, outTotal, adjointAboveScale);
  backend->Scale(mu, outTotal, a);

  backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);
  backend->MatMul(
      /*transA=*/true,
      /*transB=*/false, (int)colRows, (int)(batchSize * colCols), outChannels, a, W, (int)colRows,
      lgRepacked, (int)(batchSize * colCols), AccumT(0), colsRepacked, (int)(batchSize * colCols));
  backend->RepackForBatchedGemm(feedbackScratch, colsRepacked, colRows, batchSize, colCols);

  backend->Zero(adjoint, (size_t)batchSize * ownSize);
  backend->Col2Im(feedbackScratch, batchSize, inChannels, inHeight, inWidth, kernelH, kernelW,
                  strideH, strideW, padH, padW, adjoint);
}

template <typename T, typename AccumT>
void FullConvPCLayerT<T, AccumT>::EnsureMuHoldsDerivative() noexcept
{
  if (outChannels == 0)
    return;

  size_t outTotal = (size_t)batchSize * outChannels * outHeight * outWidth;
  backend->ActivationDerivativeInto(derivativeType, mu, preact, outTotal);
  backend->Scale(mu, outTotal, a);
}

template <typename T, typename AccumT> void FullConvPCLayerT<T, AccumT>::UpdateErrorEPC() noexcept
{
  if (isClamped)
    return;

  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  backend->Scale(e, ownStateSize, AccumT(1) - ir);
  backend->AxpyInto(e, adjoint, ownStateSize, -ir);
}

template <typename T, typename AccumT>
void FullConvPCLayerT<T, AccumT>::ClampState(const std::vector<T>& inputData) noexcept
{
  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  size_t copyElems = (std::min)(inputData.size(), ownStateSize);
  backend->CopyFromHost(z, inputData.data(), copyElems);
  isClamped = true;
  muCacheValid = false;
}

template <typename T, typename AccumT> void FullConvPCLayerT<T, AccumT>::UnclampState() noexcept
{
  isClamped = false;
}

template <typename T, typename AccumT> void FullConvPCLayerT<T, AccumT>::ResetState() noexcept
{
  size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
  backend->Zero(z, ownStateSize);
  backend->Zero(e, ownStateSize);
  if (v)
    backend->Zero(v, ownStateSize);
}

template class FullConvPCLayerT<float, float>;
template class FullConvPCLayerT<double, double>;
template class FullConvPCLayerT<Fp4E2M1Value, float>;
template class FullConvPCLayerT<hwy::float16_t, float>;
template class FullConvPCLayerT<Posit4_0Value, float>;
template class FullConvPCLayerT<Posit4_1Value, float>;
template class FullConvPCLayerT<Posit4_2Value, float>;

template void FullConvPCLayerT<float, float>::BindMemory<MemoryArenaT<float>>(
    MemoryArenaT<float>& arena);
template void FullConvPCLayerT<double, double>::BindMemory<MemoryArenaT<double>>(
    MemoryArenaT<double>& arena);
template void FullConvPCLayerT<Fp4E2M1Value, float>::BindMemory<MemoryArenaT<Fp4E2M1Value>>(
    MemoryArenaT<Fp4E2M1Value>& arena);
template void FullConvPCLayerT<hwy::float16_t, float>::BindMemory<MemoryArenaT<hwy::float16_t>>(
    MemoryArenaT<hwy::float16_t>& arena);
template void FullConvPCLayerT<Posit4_0Value, float>::BindMemory<MemoryArenaT<Posit4_0Value>>(
    MemoryArenaT<Posit4_0Value>& arena);
template void FullConvPCLayerT<Posit4_1Value, float>::BindMemory<MemoryArenaT<Posit4_1Value>>(
    MemoryArenaT<Posit4_1Value>& arena);
template void FullConvPCLayerT<Posit4_2Value, float>::BindMemory<MemoryArenaT<Posit4_2Value>>(
    MemoryArenaT<Posit4_2Value>& arena);
} // namespace Deep
