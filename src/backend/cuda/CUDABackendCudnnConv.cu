/**
 * @file CUDABackendCudnnConv.cu
 * @brief CUDABackend's optional cuDNN-backed convolution path:
 * TryFusedConvForward()/TryFusedConvBackwardData()/
 * TryFusedConvBackwardFilter(), replacing the Im2Col()+
 * RepackForBatchedGemm()+MatMul() composition CUDABackendConv.cu's
 * primitives otherwise support. The win isn't raw GEMM throughput --
 * cuBLAS already runs that GEMM on TF32 tensor cores (see
 * CUDABackendCore.cu's constructor) -- it's skipping the full im2col
 * column-buffer materialization and the extra repack/launch overhead
 * that composition pays on every settling step, for the ImageNet-scale
 * FullConvPCLayer workloads where that overhead is no longer noise.
 * Builds to false-returning stubs (selecting the existing im2col+GEMM
 * fallback every caller already has) when DEEPITY_USE_CUDNN isn't
 * defined, i.e. cuDNN wasn't found at configure time -- see
 * CMakeLists.txt -- so this file always compiles whenever CUDA is
 * enabled.
 */
#include <deepity/backend/CUDABackend.h>

#ifdef DEEPITY_USE_CUDA
#include <deepity/utils/Im2Col.h>

#ifndef DEEPITY_USE_CUDNN

namespace Deep
{
bool CUDABackend::TryFusedConvForward(const float*, const float*, float*, float, int, int, int,
                                      int, int, int, int, int, int, int, int) noexcept
{
  return false; // cuDNN not found at configure time, caller falls back
                // to Im2Col()+RepackForBatchedGemm()+MatMul()
}
bool CUDABackend::TryFusedConvBackwardData(const float*, const float*, float*, float, int, int,
                                           int, int, int, int, int, int, int, int, int) noexcept
{
  return false; // ditto, falls back to MatMul()+RepackForBatchedGemm()+Col2Im()
}
bool CUDABackend::TryFusedConvBackwardFilter(const float*, const float*, float*, float, float, int,
                                             int, int, int, int, int, int, int, int, int,
                                             int) noexcept
{
  return false; // ditto, falls back to RepackForBatchedGemm()+MatMul()
}
void CUDABackend::PrepareConvWorkspace(int, int, int, int, int, int, int, int, int, int,
                                       int) noexcept
{
  // no workspace to grow -- TryFusedConv*() above always return false in
  // this build, so nothing ever calls GrowWorkspace()
}
} // namespace Deep

#else // DEEPITY_USE_CUDNN

#include <algorithm>
#include <iostream>

/// @brief Logs the most recent cuDNN error (if any) to stderr, tagged
/// with the call site -- the cuDNN counterpart of CUDACommon.cuh's
/// CHECK_CUDA_LAUNCH, for calls that return a status directly rather
/// than needing a separate cudaGetLastError() query.
#define CHECK_CUDNN(call)                                                                         \
  do                                                                                               \
  {                                                                                               \
    cudnnStatus_t _status = (call);                                                               \
    if (_status != CUDNN_STATUS_SUCCESS)                                                          \
      std::cerr << "cuDNN error at " << __FILE__ << ":" << __LINE__ << " -> "                     \
                << cudnnGetErrorString(_status) << std::endl;                                     \
  } while (0)

namespace Deep
{
namespace
{
/// @brief One call's worth of cuDNN tensor/filter/convolution
/// descriptors, built fresh per TryFusedConv*() invocation and torn
/// down at scope exit. These are small host-side structs, not device
/// allocations, so this costs a handful of cheap driver calls rather
/// than a device round-trip -- not free, but not the bottleneck either.
/// Caching one of these per distinct (shape, layer) pair instead --
/// batchSize/channels/kernel are fixed once a layer is Compile()d, so
/// the same descriptors would serve every settling step and training
/// batch for that layer's whole lifetime -- is the next optimization
/// once this path is verified end-to-end on real hardware.
struct ConvDescriptors
{
  cudnnTensorDescriptor_t input{}, output{};
  cudnnFilterDescriptor_t filter{};
  cudnnConvolutionDescriptor_t conv{};

  ConvDescriptors(int batchSize, int inChannels, int height, int width, int outChannels,
                  int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                  int outH, int outW, bool allowTF32)
  {
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&input));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(input, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize,
                                           inChannels, height, width));
    CHECK_CUDNN(cudnnCreateTensorDescriptor(&output));
    CHECK_CUDNN(cudnnSetTensor4dDescriptor(output, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batchSize,
                                           outChannels, outH, outW));
    // [outChannels, inChannels, kernelH, kernelW], the exact memory
    // layout every ConvPCLayer-family layer already stores its flat
    // [outChannels, inChannels*kernelH*kernelW] weights in.
    CHECK_CUDNN(cudnnCreateFilterDescriptor(&filter));
    CHECK_CUDNN(cudnnSetFilter4dDescriptor(filter, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, outChannels,
                                           inChannels, kernelH, kernelW));
    // CROSS_CORRELATION (no kernel flip), matching the im2col+GEMM path
    // this replaces -- that path never flips the kernel either.
    CHECK_CUDNN(cudnnCreateConvolutionDescriptor(&conv));
    CHECK_CUDNN(cudnnSetConvolution2dDescriptor(conv, padH, padW, strideH, strideW,
                                                /*dilationH=*/1, /*dilationW=*/1,
                                                CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));
    // Mirrors SetAllowTF32()'s own cublasSetMathMode() toggle (see
    // CUDABackend::allowTF32's doc comment) -- without this, this path
    // would silently run full FP32 while MatMul() runs TF32, the exact
    // fused/unfused divergence this project's settling loop can't
    // tolerate.
    CHECK_CUDNN(cudnnSetConvolutionMathType(conv, allowTF32 ? CUDNN_TENSOR_OP_MATH
                                                             : CUDNN_FMA_MATH));
  }

  ~ConvDescriptors()
  {
    cudnnDestroyConvolutionDescriptor(conv);
    cudnnDestroyFilterDescriptor(filter);
    cudnnDestroyTensorDescriptor(output);
    cudnnDestroyTensorDescriptor(input);
  }
};

/// @brief Grows (never shrinks) CUDABackend's shared cuDNN workspace
/// buffer to at least @p bytes, the one-workspace-per-handle convention
/// cuDNN itself recommends: no two convolutions on this backend's
/// single stream are ever in flight at once, so reusing one persistent
/// buffer across every TryFusedConv*() call (like PrepareForBatchSize()
/// already does for onesVector) is safe.
void* GrowWorkspace(float** workspacePtr, size_t* capacityBytes, size_t bytes)
{
  if (bytes > *capacityBytes)
  {
    if (*workspacePtr)
      cudaFree(*workspacePtr);
    if (cudaMalloc(workspacePtr, bytes) != cudaSuccess)
    {
      *workspacePtr = nullptr;
      *capacityBytes = 0;
      return nullptr;
    }
    *capacityBytes = bytes;
  }
  return *workspacePtr;
}
} // namespace

bool CUDABackend::TryFusedConvForward(const float* input, const float* W, float* output,
                                      float alpha, int batchSize, int inChannels, int height,
                                      int width, int outChannels, int kernelH, int kernelW,
                                      int strideH, int strideW, int padH, int padW) noexcept
{
  if (!input || !W || !output)
    return false;

  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  ConvDescriptors d(batchSize, inChannels, height, width, outChannels, kernelH, kernelW, strideH,
                    strideW, padH, padW, outH, outW, allowTF32);

  cudnnConvolutionFwdAlgoPerf_t perf;
  int returned = 0;
  if (cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle, d.input, d.filter, d.conv, d.output, 1,
                                             &returned, &perf) != CUDNN_STATUS_SUCCESS ||
      returned == 0 || perf.status != CUDNN_STATUS_SUCCESS)
    return false;

  size_t workspaceSize = 0;
  CHECK_CUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, d.input, d.filter, d.conv,
                                                      d.output, perf.algo, &workspaceSize));
  void* ws = GrowWorkspace(&workspace, &workspaceBytes, workspaceSize);

  float beta = 0.0f;
  cudnnStatus_t status = cudnnConvolutionForward(cudnnHandle, &alpha, d.input, input, d.filter, W,
                                                 d.conv, perf.algo, ws, workspaceSize, &beta,
                                                 d.output, output);
  return status == CUDNN_STATUS_SUCCESS;
}

bool CUDABackend::TryFusedConvBackwardData(const float* outputGrad, const float* W,
                                           float* inputGrad, float alpha, int batchSize,
                                           int inChannels, int height, int width, int outChannels,
                                           int kernelH, int kernelW, int strideH, int strideW,
                                           int padH, int padW) noexcept
{
  if (!outputGrad || !W || !inputGrad)
    return false;

  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  ConvDescriptors d(batchSize, inChannels, height, width, outChannels, kernelH, kernelW, strideH,
                    strideW, padH, padW, outH, outW, allowTF32);

  cudnnConvolutionBwdDataAlgoPerf_t perf;
  int returned = 0;
  if (cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle, d.filter, d.output, d.conv, d.input,
                                                  1, &returned, &perf) != CUDNN_STATUS_SUCCESS ||
      returned == 0 || perf.status != CUDNN_STATUS_SUCCESS)
    return false;

  size_t workspaceSize = 0;
  CHECK_CUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, d.filter, d.output, d.conv,
                                                           d.input, perf.algo, &workspaceSize));
  void* ws = GrowWorkspace(&workspace, &workspaceBytes, workspaceSize);

  float beta = 0.0f;
  cudnnStatus_t status = cudnnConvolutionBackwardData(cudnnHandle, &alpha, d.filter, W, d.output,
                                                       outputGrad, d.conv, perf.algo, ws,
                                                       workspaceSize, &beta, d.input, inputGrad);
  return status == CUDNN_STATUS_SUCCESS;
}

bool CUDABackend::TryFusedConvBackwardFilter(const float* input, const float* outputGrad,
                                             float* dW, float alpha, float beta, int batchSize,
                                             int inChannels, int height, int width,
                                             int outChannels, int kernelH, int kernelW,
                                             int strideH, int strideW, int padH,
                                             int padW) noexcept
{
  if (!input || !outputGrad || !dW)
    return false;

  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  ConvDescriptors d(batchSize, inChannels, height, width, outChannels, kernelH, kernelW, strideH,
                    strideW, padH, padW, outH, outW, allowTF32);

  cudnnConvolutionBwdFilterAlgoPerf_t perf;
  int returned = 0;
  if (cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle, d.input, d.output, d.conv, d.filter,
                                                     1, &returned, &perf) != CUDNN_STATUS_SUCCESS ||
      returned == 0 || perf.status != CUDNN_STATUS_SUCCESS)
    return false;

  size_t workspaceSize = 0;
  CHECK_CUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(
      cudnnHandle, d.input, d.output, d.conv, d.filter, perf.algo, &workspaceSize));
  void* ws = GrowWorkspace(&workspace, &workspaceBytes, workspaceSize);

  cudnnStatus_t status = cudnnConvolutionBackwardFilter(cudnnHandle, &alpha, d.input, input,
                                                        d.output, outputGrad, d.conv, perf.algo, ws,
                                                        workspaceSize, &beta, d.filter, dW);
  return status == CUDNN_STATUS_SUCCESS;
}

void CUDABackend::PrepareConvWorkspace(int batchSize, int inChannels, int height, int width,
                                       int outChannels, int kernelH, int kernelW, int strideH,
                                       int strideW, int padH, int padW) noexcept
{
  int outH = ConvOutDim(height, kernelH, strideH, padH);
  int outW = ConvOutDim(width, kernelW, strideW, padW);
  ConvDescriptors d(batchSize, inChannels, height, width, outChannels, kernelH, kernelW, strideH,
                    strideW, padH, padW, outH, outW, allowTF32);

  size_t maxBytes = 0;

  cudnnConvolutionFwdAlgoPerf_t fwdPerf;
  int returned = 0;
  if (cudnnGetConvolutionForwardAlgorithm_v7(cudnnHandle, d.input, d.filter, d.conv, d.output, 1,
                                             &returned, &fwdPerf) == CUDNN_STATUS_SUCCESS &&
      returned > 0 && fwdPerf.status == CUDNN_STATUS_SUCCESS)
  {
    size_t bytes = 0;
    cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle, d.input, d.filter, d.conv, d.output,
                                            fwdPerf.algo, &bytes);
    maxBytes = std::max(maxBytes, bytes);
  }

  cudnnConvolutionBwdDataAlgoPerf_t dataPerf;
  returned = 0;
  if (cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnnHandle, d.filter, d.output, d.conv, d.input,
                                                  1, &returned, &dataPerf) ==
          CUDNN_STATUS_SUCCESS &&
      returned > 0 && dataPerf.status == CUDNN_STATUS_SUCCESS)
  {
    size_t bytes = 0;
    cudnnGetConvolutionBackwardDataWorkspaceSize(cudnnHandle, d.filter, d.output, d.conv, d.input,
                                                 dataPerf.algo, &bytes);
    maxBytes = std::max(maxBytes, bytes);
  }

  cudnnConvolutionBwdFilterAlgoPerf_t filterPerf;
  returned = 0;
  if (cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnnHandle, d.input, d.output, d.conv, d.filter,
                                                    1, &returned, &filterPerf) ==
          CUDNN_STATUS_SUCCESS &&
      returned > 0 && filterPerf.status == CUDNN_STATUS_SUCCESS)
  {
    size_t bytes = 0;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnnHandle, d.input, d.output, d.conv, d.filter,
                                                   filterPerf.algo, &bytes);
    maxBytes = std::max(maxBytes, bytes);
  }

  GrowWorkspace(&workspace, &workspaceBytes, maxBytes);
}
} // namespace Deep

#endif // DEEPITY_USE_CUDNN
#endif // DEEPITY_USE_CUDA
