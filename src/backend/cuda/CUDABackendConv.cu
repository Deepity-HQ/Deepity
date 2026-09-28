/**
 * @file CUDABackendConv.cu
 * @brief CUDABackend's im2col-based convolution support: the batched-GEMM
 * repack, and the Im2Col/Col2Im transform pair. Mirrors the "Convolution
 * (im2col-based, ConvPCLayer family)" section of IComputeBackend.h. Split
 * out of the former monolithic CUDABackend.cu, see CUDABackendCore.cu,
 * CUDABackendGemm.cu, CUDABackendElementwise.cu, CUDABackendActivations.cu,
 * CUDABackendFusedOps.cu, CUDABackendOptimizer.cu for the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <deepity/utils/Im2Col.h>
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

__global__ void Im2ColKernel(const float* input, int channels, int height, int width, int kH,
                             int kW, int strideH, int strideW, int padH, int padW, int outH,
                             int outW, float* columns)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t colCols = (size_t)outH * outW;
  size_t colRows = (size_t)channels * kH * kW;
  size_t total = colRows * colCols;
  if (i >= total)
    return;

  size_t row = i / colCols;
  size_t col = i % colCols;

  int c = (int)(row / ((size_t)kH * kW));
  int kh = (int)((row / kW) % kH);
  int kw = (int)(row % kW);

  int oh = (int)(col / outW);
  int ow = (int)(col % outW);

  int inRow = oh * strideH - padH + kh;
  int inCol = ow * strideW - padW + kw;

  if (inRow < 0 || inRow >= height || inCol < 0 || inCol >= width)
    columns[i] = 0.0f;
  else
    columns[i] = input[((size_t)c * height + inRow) * width + inCol];
}

void CUDABackend::Im2Col(const float* input, int channels, int height, int width, int kernelH,
                         int kernelW, int strideH, int strideW, int padH, int padW,
                         float* columns) noexcept
{
  if (!input || !columns)
    return;
  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  size_t total = (size_t)channels * kernelH * kernelW * outH * outW;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  Im2ColKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input,
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

__global__ void Col2ImKernel(const float* columns, int channels, int height, int width, int kH,
                             int kW, int strideH, int strideW, int padH, int padW, int outH,
                             int outW, float* outputImage)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  size_t colCols = (size_t)outH * outW;
  size_t colRows = (size_t)channels * kH * kW;
  size_t total = colRows * colCols;
  if (i >= total)
    return;

  size_t row = i / colCols;
  size_t col = i % colCols;

  int c = (int)(row / ((size_t)kH * kW));
  int kh = (int)((row / kW) % kH);
  int kw = (int)(row % kW);

  int oh = (int)(col / outW);
  int ow = (int)(col % outW);

  int inRow = oh * strideH - padH + kh;
  int inCol = ow * strideW - padW + kw;

  if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
  {
    size_t dstIdx = ((size_t)c * height + inRow) * width + inCol;
    atomicAdd(&outputImage[dstIdx], columns[i]);
  }
}

void CUDABackend::Col2Im(const float* columns, int channels, int height, int width, int kernelH,
                         int kernelW, int strideH, int strideW, int padH, int padW,
                         float* outputImage) noexcept
{
  if (!columns || !outputImage)
    return;
  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  size_t total = (size_t)channels * kernelH * kernelW * outH * outW;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
  Col2ImKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(columns,
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
} // namespace Deep

#endif
