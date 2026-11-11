/**
 * @file CUDABackendElementwise.cu
 * @brief CUDABackend's elementwise scalar ops: Scale/AxpyInto, the
 * AddBiasBroadcast/AddBiasPerChannel bias-add pair, MultiplyInto, and Fill.
 * Mirrors the "Elementwise scalar ops" section of IComputeBackend.h. Split
 * out of the former monolithic CUDABackend.cu, see CUDABackendCore.cu,
 * CUDABackendGemm.cu, CUDABackendActivations.cu, CUDABackendFusedOps.cu,
 * CUDABackendOptimizer.cu, CUDABackendConv.cu for the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <algorithm>
#include <iostream>

#ifdef DEEPITY_USE_CUDA
#include "CUDACommon.cuh"

namespace Deep
{
void CUDABackend::Scale(float* buf, size_t n, float alpha) noexcept
{
  if (!buf || n == 0)
    return;
  cublasSscal(handle, n, &alpha, buf, 1);
}

void CUDABackend::AxpyInto(float* y, const float* x, size_t n, float alpha) noexcept
{
  if (!x || !y || n == 0)
    return;
  cublasSaxpy(handle, n, &alpha, x, 1, y, 1);
}

__global__ void AddBiasBroadcastKernel(float* buf, const float* bias, size_t batchSize,
                                       size_t width)
{
  size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x; // maps to width
  size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y; // maps to batchSize

  if (col < width && row < batchSize)
    buf[row * width + col] += bias[col];
}

void CUDABackend::AddBiasBroadcast(float* buf, const float* bias, size_t batchSize,
                                   size_t width) noexcept
{
  dim3 threads(32, 8);
  dim3 blocks((unsigned int)((width + threads.x - 1) / threads.x),
              (unsigned int)((batchSize + threads.y - 1) / threads.y));

  AddBiasBroadcastKernel<<<blocks, threads, 0, stream>>>(buf, bias, batchSize, width);
  CHECK_CUDA_LAUNCH();
}

// Same op as AddBiasBroadcastKernel above, fused across numLayers pairs.
// layer/row are folded into the grid's X dimension, not Y: CUDA caps
// grid Y (and Z) at 65535 on every current architecture, while X goes
// up to 2^31-1 -- numLayers*batchSize is the dimension with no
// realistic ceiling here (width, on Y, is bounded by hidden-layer size).
__global__ void AddBiasBroadcastBatchedKernel(float* buf, const float* bias, size_t batchSize,
                                              size_t width, size_t numLayers)
{
  size_t layerRow = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t col = (size_t)blockIdx.y * blockDim.y + threadIdx.y;
  size_t layer = layerRow / batchSize;
  size_t row = layerRow % batchSize;

  if (col < width && layer < numLayers)
    buf[layer * batchSize * width + row * width + col] += bias[layer * width + col];
}

void CUDABackend::AddBiasBroadcastBatched(float* buf, const float* bias, size_t batchSize,
                                          size_t width, size_t numLayers) noexcept
{
  if (!buf || !bias || numLayers == 0)
    return;

  dim3 threads(32, 8);
  dim3 blocks((unsigned int)((numLayers * batchSize + threads.x - 1) / threads.x),
              (unsigned int)((width + threads.y - 1) / threads.y));

  AddBiasBroadcastBatchedKernel<<<blocks, threads, 0, stream>>>(buf, bias, batchSize, width,
                                                                numLayers);
  CHECK_CUDA_LAUNCH();
}

__global__ void MultiplyIntoKernel(float* dst, const float* a, const float* b, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    dst[i] = (a && b) ? (a[i] * b[i]) : 0.0f;
}

void CUDABackend::MultiplyInto(float* dst, const float* a, const float* b, size_t n) noexcept
{
  if (!dst || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  MultiplyIntoKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, a, b, n);
  CHECK_CUDA_LAUNCH();
}

__global__ void FillKernel(float* buf, size_t n, float value)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    buf[i] = value;
}

void CUDABackend::Fill(float* buf, size_t n, float value) noexcept
{
  if (!buf || n == 0)
    return;
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FillKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(buf, n, value);
  CHECK_CUDA_LAUNCH();
}

__global__ void AddBiasPerChannelKernel(float* buf, const float* bias, size_t channels,
                                        size_t spatialSize)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = channels * spatialSize;
  if (i < total)
  {
    size_t c = i / spatialSize;
    buf[i] += bias ? bias[c] : 0.0f;
  }
}

void CUDABackend::AddBiasPerChannel(float* buf, const float* bias, size_t channels,
                                    size_t spatialSize) noexcept
{
  if (!buf || !bias)
    return;
  constexpr int BLOCK_SIZE = 256;
  size_t total = channels * spatialSize;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  AddBiasPerChannelKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(buf, bias, channels, spatialSize);
  CHECK_CUDA_LAUNCH();
}

__global__ void AddBiasPerChannelNCHWKernel(float* buf, const float* bias, size_t batchSize,
                                            size_t channels, size_t spatialSize)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = batchSize * channels * spatialSize;
  if (i < total)
  {
    size_t c = (i / spatialSize) % channels;
    buf[i] += bias ? bias[c] : 0.0f;
  }
}

void CUDABackend::AddBiasPerChannelNCHW(float* buf, const float* bias, size_t batchSize,
                                        size_t channels, size_t spatialSize) noexcept
{
  if (!buf || !bias)
    return;
  constexpr int BLOCK_SIZE = 256;
  size_t total = batchSize * channels * spatialSize;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  AddBiasPerChannelNCHWKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      buf, bias, batchSize, channels, spatialSize);
  CHECK_CUDA_LAUNCH();
}

__global__ void AxpyBroadcastIntoKernel(float* y, const float* x, const float* factor,
                                        size_t batchSize, size_t width, float alpha)
{
  size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x; // maps to width
  size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y; // maps to batchSize

  if (col < width && row < batchSize)
  {
    size_t idx = row * width + col;
    y[idx] += alpha * x[idx] * factor[col];
  }
}

void CUDABackend::AxpyBroadcastInto(float* y, const float* x, const float* factor, size_t batchSize,
                                    size_t width, float alpha) noexcept
{
  if (!y || !x || !factor || batchSize == 0 || width == 0)
    return;
  dim3 threads(32, 8);
  dim3 blocks((unsigned int)((width + threads.x - 1) / threads.x),
              (unsigned int)((batchSize + threads.y - 1) / threads.y));
  AxpyBroadcastIntoKernel<<<blocks, threads, 0, stream>>>(y, x, factor, batchSize, width, alpha);
  CHECK_CUDA_LAUNCH();
}

__global__ void MultiplyBroadcastIntoKernel(float* dst, const float* a, const float* factor,
                                            const float* b, size_t batchSize, size_t width)
{
  size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y;

  if (col < width && row < batchSize)
  {
    size_t idx = row * width + col;
    dst[idx] = a[idx] * factor[col] * b[idx];
  }
}

void CUDABackend::MultiplyBroadcastInto(float* dst, const float* a, const float* factor,
                                        const float* b, size_t batchSize, size_t width) noexcept
{
  if (!dst || !a || !factor || !b || batchSize == 0 || width == 0)
    return;
  dim3 threads(32, 8);
  dim3 blocks((unsigned int)((width + threads.x - 1) / threads.x),
              (unsigned int)((batchSize + threads.y - 1) / threads.y));
  MultiplyBroadcastIntoKernel<<<blocks, threads, 0, stream>>>(dst, a, factor, b, batchSize, width);
  CHECK_CUDA_LAUNCH();
}

// One block per row: strided partial sums, block reduce via warp shuffles,
// then every thread subtracts the row mean. Dense row-major, in place.
__global__ void CenterRowsKernel(float* buf, size_t cols)
{
  __shared__ float warpSums[32];
  float* row = buf + (size_t)blockIdx.x * cols;

  float sum = 0.0f;
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x)
    sum += row[c];
  for (int off = 16; off > 0; off >>= 1)
    sum += __shfl_down_sync(0xffffffffu, sum, off);
  if ((threadIdx.x & 31) == 0)
    warpSums[threadIdx.x >> 5] = sum;
  __syncthreads();

  if (threadIdx.x < 32)
  {
    float v = threadIdx.x < (blockDim.x + 31) / 32 ? warpSums[threadIdx.x] : 0.0f;
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (threadIdx.x == 0)
      warpSums[0] = v / (float)cols;
  }
  __syncthreads();

  const float mean = warpSums[0];
  for (size_t c = threadIdx.x; c < cols; c += blockDim.x)
    row[c] -= mean;
}

void CUDABackend::CenterRows(float* buf, size_t rows, size_t cols) noexcept
{
  if (!buf || rows == 0 || cols == 0)
    return;
  // Block size: a multiple of 32, no larger than the row needs, max 256.
  unsigned int threads = (unsigned int)std::min<size_t>(256, ((cols + 31) / 32) * 32);
  CenterRowsKernel<<<(unsigned int)rows, threads, 0, stream>>>(buf, cols);
  CHECK_CUDA_LAUNCH();
}
} // namespace Deep

#endif
