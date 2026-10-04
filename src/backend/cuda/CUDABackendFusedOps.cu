/**
 * @file CUDABackendFusedOps.cu
 * @brief CUDABackend's fused PC-specific ops: the settling-step state
 * update (plain and momentum variants), Gaussian error/energy, and softmax
 * cross-entropy error/energy. Mirrors the "Fused PC-specific ops" section of
 * IComputeBackend.h (minus TryFusedForwardPass, see CUDABackendGemm.cu).
 * Split out of the former monolithic CUDABackend.cu, see
 * CUDABackendCore.cu, CUDABackendGemm.cu, CUDABackendElementwise.cu,
 * CUDABackendActivations.cu, CUDABackendOptimizer.cu, CUDABackendConv.cu for
 * the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <cmath>
#include <iostream>

#ifdef DEEPITY_USE_CUDA
#include "CUDACommon.cuh"

namespace Deep
{
__global__ void FusedStateUpdateKernel(float* z, const float* feedback, ActivationType dType,
                                       const float* e, size_t n, float ir)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float fb = feedback ? feedback[i] : 0.0f;
    float dev = ActivationDerivativeDevice(dType, z[i]);
    float err = e ? e[i] : 0.0f;
    z[i] += ir * ((fb * dev) - err);
  }
}

__global__ void FusedStateUpdateMomentumKernel(float* z, float* v, const float* feedback,
                                               ActivationType dType, const float* e, size_t n,
                                               float ir, float beta)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float fb = feedback ? feedback[i] : 0.0f;
    float dev = ActivationDerivativeDevice(dType, z[i]);
    float err = e ? e[i] : 0.0f;
    float update = (fb * dev) - err;
    float v_new = beta * v[i] + (1.0f - beta) * update;
    v[i] = v_new;
    z[i] += ir * v_new;
  }
}

__global__ void ComputeErrorKernel(float* e, const float* z, const float* mu, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    e[i] = z[i] - mu[i];
}

void CUDABackend::FusedStateUpdate(float* z, const float* feedback, ActivationType dType,
                                   const float* e, size_t n, float ir) noexcept
{
  if (!z || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedStateUpdateKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(z, feedback, dType, e, n, ir);
  CHECK_CUDA_LAUNCH();
}

void CUDABackend::FusedStateUpdateMomentum(float* z, float* v, const float* feedback,
                                           ActivationType dType, const float* e, size_t n, float ir,
                                           float beta) noexcept
{
  if (!z || !v || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedStateUpdateMomentumKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      z, v, feedback, dType, e, n, ir, beta);
  CHECK_CUDA_LAUNCH();
}

float CUDABackend::ComputeErrorAndEnergy(float* e, const float* z, const float* mu,
                                         size_t n) noexcept
{
  if (!e || !z || !mu || n == 0)
    return 0.0f;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

  ComputeErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
  CHECK_CUDA_LAUNCH();

  float sum_of_squares = 0.0f;
  cublasSdot(handle, n, e, 1, e, 1, &sum_of_squares);
  cudaStreamSynchronize(stream);

  return 0.5f * sum_of_squares;
}

void CUDABackend::ComputeError(float* e, const float* z, const float* mu, size_t n) noexcept
{
  if (!e || !z || !mu || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  ComputeErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
  CHECK_CUDA_LAUNCH();
}

// One block per batch row (not one thread): SOFTMAX_BLOCK_SIZE threads
// cooperate across the nextSize class dimension via a classic
// shared-memory halving reduction, so a small batchSize no longer leaves
// almost the entire GPU idle while each of the few threads serially walks
// the whole class dimension four times. SOFTMAX_BLOCK_SIZE must stay a
// power of two (the halving reduction relies on it) and must match the
// launch configuration below exactly.
constexpr int SOFTMAX_BLOCK_SIZE = 256;

__global__ void SoftmaxCrossEntropyKernel(float* e, const float* z, const float* mu,
                                          size_t batchSize, size_t nextSize, float* rowEnergies)
{
  size_t b = blockIdx.x;
  if (b >= batchSize)
    return;

  size_t base = b * nextSize;
  const float eps = 1e-8f;
  int tid = threadIdx.x;

  __shared__ float sdata[SOFTMAX_BLOCK_SIZE];

  // Row max, reduced across the block (numerically-stable softmax).
  float local_max = -INFINITY;
  for (size_t j = tid; j < nextSize; j += SOFTMAX_BLOCK_SIZE)
    local_max = fmaxf(local_max, mu[base + j]);
  sdata[tid] = local_max;
  __syncthreads();
  for (int stride = SOFTMAX_BLOCK_SIZE / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
    __syncthreads();
  }
  float max_val = sdata[0];
  __syncthreads();

  // exp(mu - max) and its row sum, same reduction shape.
  float local_sum = 0.0f;
  for (size_t j = tid; j < nextSize; j += SOFTMAX_BLOCK_SIZE)
  {
    float ex = expf(mu[base + j] - max_val);
    e[base + j] = ex;
    local_sum += ex;
  }
  sdata[tid] = local_sum;
  __syncthreads();
  for (int stride = SOFTMAX_BLOCK_SIZE / 2; stride > 0; stride >>= 1)
  {
    if (tid < stride)
      sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }
  float sum_exp = sdata[0];
  // Every thread still needs to read sum_exp out of sdata[0] before any
  // thread starts overwriting sdata[] below -- without this barrier a
  // fast thread's write could stomp sdata[0] before a slow thread's read.
  __syncthreads();

  // Normalize to a probability (always), optionally reducing the
  // cross-entropy row energy alongside it.
  bool needRowEnergy = (rowEnergies != nullptr);
  float local_energy = 0.0f;
  for (size_t j = tid; j < nextSize; j += SOFTMAX_BLOCK_SIZE)
  {
    float prob = e[base + j] / sum_exp;
    e[base + j] = prob;
    if (needRowEnergy)
      local_energy -= z[base + j] * logf(prob + eps);
  }
  if (needRowEnergy)
  {
    sdata[tid] = local_energy;
    __syncthreads();
    for (int stride = SOFTMAX_BLOCK_SIZE / 2; stride > 0; stride >>= 1)
    {
      if (tid < stride)
        sdata[tid] += sdata[tid + stride];
      __syncthreads();
    }
    if (tid == 0)
      rowEnergies[b] = sdata[0];
  }

  for (size_t j = tid; j < nextSize; j += SOFTMAX_BLOCK_SIZE)
    e[base + j] = z[base + j] - e[base + j];
}

float CUDABackend::ComputeSoftmaxCrossEntropyErrorAndEnergy(float* e, const float* z,
                                                            const float* mu, size_t batchSize,
                                                            size_t nextSize,
                                                            float* rowEnergies) noexcept
{
  if (!e || !z || !mu || !rowEnergies || batchSize == 0 || nextSize == 0)
    return 0.0f;

  SoftmaxCrossEntropyKernel<<<(int)batchSize, SOFTMAX_BLOCK_SIZE, 0, stream>>>(
      e, z, mu, batchSize, nextSize, rowEnergies);
  CHECK_CUDA_LAUNCH();

  float total_energy = 0.0f;
  cublasSasum(handle, (int)batchSize, rowEnergies, 1, &total_energy);
  cudaStreamSynchronize(stream);

  return total_energy;
}

void CUDABackend::ComputeSoftmaxCrossEntropyError(float* e, const float* z, const float* mu,
                                                  size_t batchSize, size_t nextSize) noexcept
{
  if (!e || !z || !mu || batchSize == 0 || nextSize == 0)
    return;

  SoftmaxCrossEntropyKernel<<<(int)batchSize, SOFTMAX_BLOCK_SIZE, 0, stream>>>(
      e, z, mu, batchSize, nextSize, /*rowEnergies=*/nullptr);
  CHECK_CUDA_LAUNCH();
}

__global__ void FusedActivationDerivativeMultiplyKernel(float* dst, const float* a,
                                                         float* activatedInOut, ActivationType dType,
                                                         size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    float deriv = ActivationDerivativeFromActivatedDevice(dType, activatedInOut[i]);
    dst[i] = a[i] * deriv;
    activatedInOut[i] = deriv;
  }
}

void CUDABackend::FusedActivationDerivativeMultiply(float* dst, const float* a, float* activatedInOut,
                                                     ActivationType dType, size_t n) noexcept
{
  if (!dst || !a || !activatedInOut || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FusedActivationDerivativeMultiplyKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, a, activatedInOut,
                                                                             dType, n);
  CHECK_CUDA_LAUNCH();
}

// One thread per element; each thread's energy contribution is folded into
// a single device accumulator via atomicAdd. Simple and obviously correct
// rather than a tree reduction, "port first, optimize" (see
// RepackForBatchedGemm's own doc comment for the same reasoning applied
// elsewhere in this file).
//
// IMPORTANT, read before touching this function: ConvPCNetwork and
// DiscriminativePCNetwork's TrainStep()/TrainStepWithProjection() DO now
// use CUDA graph capture (this comment previously said otherwise -- that
// was true when written, no longer is). This kernel launch itself is
// fine to capture, but CUDABackend::ComputePrecisionWeightedErrorAndEnergy()
// below it does a synchronous cudaMemcpyAsync+cudaStreamSynchronize to
// return the energy as a host float, which is illegal while a stream is
// being captured and crashes. Every call site inside a settling loop
// that's ever captured MUST use ComputePrecisionWeightedError() (below,
// no energy, no sync) instead, exactly like ComputeError() exists
// alongside ComputeErrorAndEnergy() for the same reason.
__global__ void PrecisionWeightedErrorEnergyKernel(float* e, const float* z, const float* mu,
                                                    const float* p, size_t n, size_t width,
                                                    float* energyAccum)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    size_t i = idx % width;
    float err = z[idx] - mu[idx];
    e[idx] = err;
    float precision = fmaxf(p[i], 1e-8f);
    float contribution = 0.5f * precision * err * err - 0.5f * logf(precision);
    atomicAdd(energyAccum, contribution);
  }
}

// No-energy counterpart: just e = z - mu, no atomicAdd/reduction/sync at
// all, safe to call from inside a captured region.
__global__ void PrecisionWeightedErrorKernel(float* e, const float* z, const float* mu, size_t n)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    e[idx] = z[idx] - mu[idx];
}

float CUDABackend::ComputePrecisionWeightedErrorAndEnergy(float* e, const float* z, const float* mu,
                                                          const float* p, size_t batchSize,
                                                          size_t width) noexcept
{
  size_t n = batchSize * width;
  if (!e || !z || !mu || !p || n == 0)
    return 0.0f;

  cudaMemsetAsync(scalarScratch, 0, sizeof(float), stream);

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  PrecisionWeightedErrorEnergyKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, p, n, width,
                                                                        scalarScratch);
  CHECK_CUDA_LAUNCH();

  float total = 0.0f;
  cudaMemcpyAsync(&total, scalarScratch, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return total;
}

void CUDABackend::ComputePrecisionWeightedError(float* e, const float* z, const float* mu,
                                                const float* /*p*/, size_t batchSize,
                                                size_t width) noexcept
{
  size_t n = batchSize * width;
  if (!e || !z || !mu || n == 0)
    return;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  PrecisionWeightedErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
  CHECK_CUDA_LAUNCH();
}

// One thread per element, same atomicAdd-into-one-accumulator pattern as
// PrecisionWeightedErrorEnergyKernel above.
__global__ void SumKernel(const float* buf, size_t n, float* sumAccum)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    atomicAdd(sumAccum, buf[idx]);
}

float CUDABackend::Sum(const float* buf, size_t n) noexcept
{
  if (!buf || n == 0)
    return 0.0f;

  cudaMemsetAsync(scalarScratch, 0, sizeof(float), stream);

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  SumKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(buf, n, scalarScratch);
  CHECK_CUDA_LAUNCH();

  // Synchronous readback, same as ComputeErrorAndEnergy and friends --
  // see this method's own IComputeBackend doc: never call from inside a
  // captured graph region.
  float total = 0.0f;
  cudaMemcpyAsync(&total, scalarScratch, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return total;
}

void CUDABackend::ComputePrecisionWeightedError(float* e, const float* z, const float* mu,
                                                const float* /*p*/, size_t batchSize,
                                                size_t width) noexcept
{
  size_t n = batchSize * width;
  if (!e || !z || !mu || n == 0)
    return;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  PrecisionWeightedErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
  CHECK_CUDA_LAUNCH();
}

// One thread per element, same atomicAdd-into-one-accumulator pattern as
// PrecisionWeightedErrorEnergyKernel above.
__global__ void SumKernel(const float* buf, size_t n, float* sumAccum)
{
  size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n)
    atomicAdd(sumAccum, buf[idx]);
}

float CUDABackend::Sum(const float* buf, size_t n) noexcept
{
  if (!buf || n == 0)
    return 0.0f;

  float* sumAccum = Allocate(1);
  if (!sumAccum)
    return 0.0f;
  cudaMemsetAsync(sumAccum, 0, sizeof(float), stream);

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  SumKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(buf, n, sumAccum);
  CHECK_CUDA_LAUNCH();

  // Synchronous readback, same as ComputeErrorAndEnergy and friends --
  // see this method's own IComputeBackend doc: never call from inside a
  // captured graph region.
  float total = 0.0f;
  cudaMemcpyAsync(&total, sumAccum, sizeof(float), cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);
  Free(sumAccum);

  return total;
}

__global__ void UpdatePrecisionFromErrorKernel(float* p, float* log_p, const float* e,
                                               size_t batchSize, size_t width, float pr)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < width)
  {
    float grad = 0.0f;
    for (size_t b = 0; b < batchSize; ++b)
    {
      float err = e[b * width + i];
      grad += 0.5f * (p[i] * err * err - 1.0f);
    }
    grad /= (float)batchSize;

    float newLogP = log_p[i] - pr * grad;
    newLogP = fmaxf(-5.0f, fminf(newLogP, 5.0f));
    log_p[i] = newLogP;
    p[i] = expf(newLogP);
  }
}

void CUDABackend::UpdatePrecisionFromError(float* p, float* log_p, const float* e, size_t batchSize,
                                           size_t width, float pr) noexcept
{
  if (!p || !log_p || !e || batchSize == 0 || width == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((width + BLOCK_SIZE - 1) / BLOCK_SIZE);
  UpdatePrecisionFromErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(p, log_p, e, batchSize, width,
                                                                    pr);
  CHECK_CUDA_LAUNCH();
}
} // namespace Deep

#endif
