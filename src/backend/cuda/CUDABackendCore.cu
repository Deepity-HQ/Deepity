/**
 * @file CUDABackendCore.cu
 * @brief CUDABackend lifecycle: construction/destruction, CUDA-graph
 * capture/replay, and raw device memory management (alloc/free/copy/RNG).
 * Mirrors the "Graphs" and "Memory" sections of IComputeBackend.h. Split out
 * of the former monolithic CUDABackend.cu, see CUDABackendGemm.cu,
 * CUDABackendElementwise.cu, CUDABackendActivations.cu,
 * CUDABackendFusedOps.cu, CUDABackendOptimizer.cu, CUDABackendConv.cu for
 * the rest.
 */
#include <deepity/backend/CUDABackend.h>
#include <iostream>

#ifdef DEEPITY_USE_CUDA
#include <curand_kernel.h>

#include "CUDACommon.cuh"

namespace Deep
{
CUDABackend::CUDABackend()
{
  cudaStreamCreate(&this->stream);
  cublasCreate(&this->handle);
  cublasSetStream(this->handle, this->stream);
  // Without this, every GEMM (every conv, since this backend lowers conv
  // to im2col+GEMM) runs in plain FP32 on CUDA cores (~19.5 TFLOPS on an
  // A100) instead of TF32 on tensor cores (~156 TFLOPS) -- the single
  // largest gap between this backend and PyTorch's cuDNN/cuBLAS
  // convolutions, which use TF32 by default. See SetAllowTF32's own doc
  // comment for why a caller doing a finite-difference gradient check
  // needs to turn this back off.
  SetAllowTF32(true);
  cudaMalloc(&scalarScratch, sizeof(float));
}

CUDABackend::~CUDABackend()
{
  if (hasGraph)
  {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
  }
  if (onesVector)
    cudaFree(onesVector);
  if (workspace)
    cudaFree(workspace);
  if (scalarScratch)
    cudaFree(scalarScratch);
  cublasDestroy(this->handle);
  cudaStreamDestroy(this->stream);
}

void CUDABackend::BeginGraphCapture() noexcept
{
  cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
  if (err != cudaSuccess)
    std::cerr << "cudaStreamBeginCapture failed: " << cudaGetErrorString(err) << "\n";
}

bool CUDABackend::EndGraphCapture() noexcept
{
  cudaGraph_t newGraph;
  cudaError_t err = cudaStreamEndCapture(stream, &newGraph);
  if (err != cudaSuccess)
  {
    std::cerr << "cudaStreamEndCapture failed: " << cudaGetErrorString(err) << "\n";
    return false;
  }

  if (hasGraph)
  {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    hasGraph = false;
  }

  graph = newGraph;
  err = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
  if (err != cudaSuccess)
  {
    std::cerr << "cudaGraphInstantiate failed: " << cudaGetErrorString(err) << "\n";
    cudaGraphDestroy(graph);
    graph = nullptr;
    return false;
  }

  hasGraph = true;
  return true;
}

void CUDABackend::ReplayGraph() noexcept
{
  if (!hasGraph)
  {
    std::cerr << "ReplayGraph() called before any graph was captured.\n";
    return;
  }
  cudaError_t err = cudaGraphLaunch(graphExec, stream);
  if (err != cudaSuccess)
    std::cerr << "cudaGraphLaunch failed: " << cudaGetErrorString(err) << "\n";
}

void CUDABackend::Synchronize() noexcept
{
  cudaError_t err = cudaStreamSynchronize(stream);
  if (err != cudaSuccess)
    std::cerr << "cudaStreamSynchronize failed: " << cudaGetErrorString(err) << "\n";
}

float* CUDABackend::Allocate(size_t numFloats)
{
  float* ptr = nullptr;
  if (cudaMalloc(&ptr, numFloats * sizeof(float)) != cudaSuccess)
  {
    std::cerr << "Could not allocate memory to CUDA backend.\n";
    return nullptr;
  }
  return ptr;
}

void CUDABackend::Free(float* ptr) noexcept
{
  if (ptr)
    cudaFree(ptr);
}

void CUDABackend::Zero(float* ptr, size_t numFloats) noexcept
{
  if (ptr && numFloats > 0)
    cudaMemsetAsync(ptr, 0, numFloats * sizeof(float), stream);
}

void CUDABackend::Copy(float* dst, const float* src, size_t numFloats) noexcept
{
  if (dst && src && numFloats > 0)
    cudaMemcpyAsync(dst, src, numFloats * sizeof(float), cudaMemcpyDefault, stream);
}

void CUDABackend::CopyFromHost(float* deviceDst, const float* hostSrc, size_t numFloats) noexcept
{
  if (deviceDst && hostSrc && numFloats > 0)
    cudaMemcpy(deviceDst, hostSrc, numFloats * sizeof(float), cudaMemcpyHostToDevice);
}

void CUDABackend::CopyToHost(float* hostDst, const float* deviceSrc, size_t numFloats) noexcept
{
  if (hostDst && deviceSrc && numFloats > 0)
    cudaMemcpy(hostDst, deviceSrc, numFloats * sizeof(float), cudaMemcpyDeviceToHost);
}

__global__ void normal_generation(curandState* state, float* random_numbers, size_t n, float mean,
                                  float stddev, uint32_t seed)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    curand_init(seed, static_cast<unsigned long long>(i), 0, &state[i]);
    random_numbers[i] = mean + stddev * curand_normal(&state[i]);
  }
}

__global__ void uniform_generation(curandState* state, float* random_numbers, size_t n, float min,
                                   float range, uint32_t seed)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
  {
    curand_init(seed, static_cast<unsigned long long>(i), 0, &state[i]);
    random_numbers[i] = min + curand_uniform(&state[i]) * range;
  }
}

void CUDABackend::RandomizeNormal(float* buf, size_t n, float mean, float stddev,
                                  uint32_t seed) noexcept
{
  if (!buf || n == 0)
    return;

  curandState* state = nullptr;
  if (cudaMalloc(&state, n * sizeof(curandState)) != cudaSuccess)
    return;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  normal_generation<<<blocks, BLOCK_SIZE, 0, stream>>>(state, buf, n, mean, stddev, seed);
  CHECK_CUDA_LAUNCH();
  cudaStreamSynchronize(stream);
  cudaFree(state);
}

void CUDABackend::RandomizeUniform(float* buf, size_t n, float min, float max,
                                   uint32_t seed) noexcept
{
  if (!buf || n == 0)
    return;

  curandState* state = nullptr;
  if (cudaMalloc(&state, n * sizeof(curandState)) != cudaSuccess)
    return;

  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
  uniform_generation<<<blocks, BLOCK_SIZE, 0, stream>>>(state, buf, n, min, max - min, seed);
  CHECK_CUDA_LAUNCH();
  cudaStreamSynchronize(stream);
  cudaFree(state);
}

__global__ void FillOnesKernel(float* buf, size_t n)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    buf[i] = 1.0f;
}

void CUDABackend::PrepareForBatchSize(size_t batchSize) noexcept
{
  if (onesCapacity < batchSize)
  {
    if (onesVector)
      cudaFree(onesVector);
    cudaMalloc(&onesVector, batchSize * sizeof(float));
    onesCapacity = batchSize;
  }
  constexpr int BLOCK_SIZE = 256;
  const int blocks = static_cast<int>((batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
  FillOnesKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(onesVector, batchSize);
  CHECK_CUDA_LAUNCH();
}
} // namespace Deep

#endif
