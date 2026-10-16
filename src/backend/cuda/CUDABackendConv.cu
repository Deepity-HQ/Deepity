/**
 * @file CUDABackendConv.cu
 * @brief CUDABackend's im2col-based convolution support: the batched-GEMM
 * repack, the Im2Col/Col2Im transform pair, and 2D max pooling. Mirrors
 * the "Convolution (im2col-based, ConvPCLayer family)" section of
 * IComputeBackend.h. Split out of the former monolithic CUDABackend.cu,
 * see CUDABackendCore.cu, CUDABackendGemm.cu, CUDABackendElementwise.cu,
 * CUDABackendActivations.cu, CUDABackendFusedOps.cu, CUDABackendOptimizer.cu
 * for the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <deepity/utils/Im2Col.h>
#include <deepity/utils/MaxPool2D.h>
#include <cfloat>
#include <iostream>

#ifdef DEEPITY_USE_CUDA
#include "CUDACommon.cuh"

namespace Deep
{
__global__ void RepackForBatchedGemmKernel(float* dst, const float* src, size_t batchSize,
                                           size_t rows, size_t cols)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t total = rows * batchSize * cols;
  if (i >= total)
    return;

  size_t row = i / (batchSize * cols);
  size_t rem = i % (batchSize * cols);
  size_t batch = rem / cols;
  size_t col = rem % cols;

  size_t srcIdx = batch * rows * cols + row * cols + col;
  dst[i] = src[srcIdx];
}

void CUDABackend::RepackForBatchedGemm(float* dst, const float* src, size_t batchSize, size_t rows,
                                       size_t cols) noexcept
{
  if (!dst || !src)
    return;
  constexpr int BLOCK_SIZE = 256;
  size_t total = rows * batchSize * cols;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  RepackForBatchedGemmKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, batchSize, rows, cols);
  CHECK_CUDA_LAUNCH();
}

// batchSize folded into the total-work index space (one extra outer
// dimension on top of the existing row/col indexing) instead of a
// host-side per-image loop: every other kernel in this backend already
// processes a whole batch per launch, Im2Col/Col2Im were the two
// exceptions, launching batchSize separate kernels (at BATCH_SIZE=250,
// that's 250 kernel-launch round-trips per call) for work that's
// genuinely independent per image and trivially fits in one launch's
// grid instead.
__global__ void Im2ColKernel(const float* input, int batchSize, int channels, int height,
                             int width, int kH, int kW, int strideH, int strideW, int padH,
                             int padW, int outH, int outW, float* columns)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t colCols = (size_t)outH * outW;
  size_t colRows = (size_t)channels * kH * kW;
  size_t perImage = colRows * colCols;
  size_t total = (size_t)batchSize * perImage;
  if (i >= total)
    return;

  size_t batch = i / perImage;
  size_t rem = i % perImage;
  size_t row = rem / colCols;
  size_t col = rem % colCols;

  int c = (int)(row / ((size_t)kH * kW));
  int kh = (int)((row / kW) % kH);
  int kw = (int)(row % kW);

  int oh = (int)(col / outW);
  int ow = (int)(col % outW);

  int inRow = oh * strideH - padH + kh;
  int inCol = ow * strideW - padW + kw;

  const float* image = input + batch * (size_t)channels * height * width;
  if (inRow < 0 || inRow >= height || inCol < 0 || inCol >= width)
    columns[i] = 0.0f;
  else
    columns[i] = image[((size_t)c * height + inRow) * width + inCol];
}

void CUDABackend::Im2Col(const float* input, int batchSize, int channels, int height, int width,
                         int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                         float* columns) noexcept
{
  if (!input || !columns)
    return;
  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  size_t total = (size_t)batchSize * channels * kernelH * kernelW * outH * outW;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  Im2ColKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input,
                                                  batchSize,
                                                  channels,
                                                  height,
                                                  width,
                                                  kernelH,
                                                  kernelW,
                                                  strideH,
                                                  strideW,
                                                  padH,
                                                  padW,
                                                  outH,
                                                  outW,
                                                  columns);
  CHECK_CUDA_LAUNCH();
}

// Same batching rationale as Im2ColKernel above. atomicAdd is still
// per-element (needed regardless of batching -- overlapping receptive
// fields within a single image already require it), just now spread
// across one launch's full grid instead of batchSize launches each
// re-paying kernel-launch overhead for the exact same atomic pattern.
__global__ void Col2ImKernel(const float* columns, int batchSize, int channels, int height,
                             int width, int kH, int kW, int strideH, int strideW, int padH,
                             int padW, int outH, int outW, float* outputImage)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t colCols = (size_t)outH * outW;
  size_t colRows = (size_t)channels * kH * kW;
  size_t perImage = colRows * colCols;
  size_t total = (size_t)batchSize * perImage;
  if (i >= total)
    return;

  size_t batch = i / perImage;
  size_t rem = i % perImage;
  size_t row = rem / colCols;
  size_t col = rem % colCols;

  int c = (int)(row / ((size_t)kH * kW));
  int kh = (int)((row / kW) % kH);
  int kw = (int)(row % kW);

  int oh = (int)(col / outW);
  int ow = (int)(col % outW);

  int inRow = oh * strideH - padH + kh;
  int inCol = ow * strideW - padW + kw;

  if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
  {
    float* image = outputImage + batch * (size_t)channels * height * width;
    size_t dstIdx = ((size_t)c * height + inRow) * width + inCol;
    atomicAdd(&image[dstIdx], columns[i]);
  }
}

void CUDABackend::Col2Im(const float* columns, int batchSize, int channels, int height, int width,
                         int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                         float* outputImage) noexcept
{
  if (!columns || !outputImage)
    return;
  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  size_t total = (size_t)batchSize * channels * kernelH * kernelW * outH * outW;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  Col2ImKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(columns,
                                                  batchSize,
                                                  channels,
                                                  height,
                                                  width,
                                                  kernelH,
                                                  kernelW,
                                                  strideH,
                                                  strideW,
                                                  padH,
                                                  padW,
                                                  outH,
                                                  outW,
                                                  outputImage);
  CHECK_CUDA_LAUNCH();
}

// Same one-thread-per-output-element, whole-batch-per-launch pattern as
// Im2Col/Col2Im above. No padding (see Deep::PoolOutDim), so every window
// is fully in-bounds by construction -- no boundary checks needed inside
// the pooling loop itself.
__global__ void MaxPool2DForwardKernel(const float* input, int batchSize, int channels,
                                       int height, int width, int poolH, int poolW, int strideH,
                                       int strideW, int outH, int outW, float* output, int* argmax)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t perImage = (size_t)channels * outH * outW;
  size_t total = (size_t)batchSize * perImage;
  if (i >= total)
    return;

  size_t batch = i / perImage;
  size_t rem = i % perImage;
  size_t c = rem / ((size_t)outH * outW);
  size_t spatial = rem % ((size_t)outH * outW);
  int oh = (int)(spatial / outW);
  int ow = (int)(spatial % outW);

  const float* inPlane = input + (batch * channels + c) * (size_t)height * width;

  float bestVal = -FLT_MAX;
  int bestIdx = 0;
  for (int ph = 0; ph < poolH; ++ph)
  {
    int ih = oh * strideH + ph;
    for (int pw = 0; pw < poolW; ++pw)
    {
      int iw = ow * strideW + pw;
      int idx = ih * width + iw;
      float v = inPlane[idx];
      if (v > bestVal)
      {
        bestVal = v;
        bestIdx = idx;
      }
    }
  }

  output[i] = bestVal;
  argmax[i] = bestIdx;
}

void CUDABackend::MaxPool2DForward(const float* input, int batchSize, int channels, int height,
                                   int width, int poolH, int poolW, int strideH, int strideW,
                                   float* output, int* argmax) noexcept
{
  if (!input || !output || !argmax)
    return;
  int outH = PoolOutDim(height, poolH, strideH);
  int outW = PoolOutDim(width, poolW, strideW);
  size_t total = (size_t)batchSize * channels * outH * outW;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  MaxPool2DForwardKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      input, batchSize, channels, height, width, poolH, poolW, strideH, strideW, outH, outW,
      output, argmax);
  CHECK_CUDA_LAUNCH();
}

// The adjoint of MaxPool2DForwardKernel: one thread per POOLED (output)
// element, scattering into inputGrad at the position its own argmax
// recorded. atomicAdd for the same reason Col2ImKernel above needs it --
// overlapping windows (stride < pool) can make one input position the
// argmax for more than one output position.
__global__ void MaxPool2DBackwardKernel(const float* outputGrad, const int* argmax,
                                        int batchSize, int channels, int height, int width,
                                        int outH, int outW, float* inputGrad)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t perImage = (size_t)channels * outH * outW;
  size_t total = (size_t)batchSize * perImage;
  if (i >= total)
    return;

  size_t batch = i / perImage;
  size_t rem = i % perImage;
  size_t c = rem / ((size_t)outH * outW);

  float* inPlane = inputGrad + (batch * channels + c) * (size_t)height * width;
  atomicAdd(&inPlane[argmax[i]], outputGrad[i]);
}

void CUDABackend::MaxPool2DBackward(const float* outputGrad, const int* argmax, int batchSize,
                                    int channels, int height, int width, int outH, int outW,
                                    float* inputGrad) noexcept
{
  if (!outputGrad || !argmax || !inputGrad)
    return;
  size_t total = (size_t)batchSize * channels * outH * outW;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  MaxPool2DBackwardKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
      outputGrad, argmax, batchSize, channels, height, width, outH, outW, inputGrad);
  CHECK_CUDA_LAUNCH();
}
} // namespace Deep

#endif
