#pragma once
#include <deepity/backend/IComputeBackend.h>
#ifdef DEEPITY_USE_CUDA
#include <cublas_v2.h>
#endif
#include <deepity/backend/Tensor.h>

namespace Deep
{
/// @brief CUDA implementation of IComputeBackend: device kernels for
/// elementwise ops, cuBLAS for GEMM, CUTLASS for the fused forward path,
/// CUDA graphs for capture/replay. See src/backend/cuda/ for the
/// implementation, split by concern to mirror this interface's own
/// section grouping.
class CUDABackend : public IComputeBackend
{
public:
  CUDABackend();
  ~CUDABackend() override;
  /// @copydoc Deep::IComputeBackend::BeginGraphCapture
  void BeginGraphCapture() noexcept override;
  /// @copydoc Deep::IComputeBackend::EndGraphCapture
  bool EndGraphCapture() noexcept override;
  /// @copydoc Deep::IComputeBackend::ReplayGraph
  void ReplayGraph() noexcept override;

  /// @copydoc Deep::IComputeBackend::Allocate
  float* Allocate(size_t numFloats) override;
  /// @copydoc Deep::IComputeBackend::Free
  void Free(float* ptr) noexcept override;
  /// @copydoc Deep::IComputeBackend::Zero
  void Zero(float* ptr, size_t numFloats) noexcept override;
  /// @copydoc Deep::IComputeBackend::Copy
  void Copy(float* dst, const float* src, size_t numFloats) noexcept override;
  /// @copydoc Deep::IComputeBackend::CopyFromHost
  void CopyFromHost(float* deviceDst, const float* hostSrc, size_t numFloats) noexcept override;
  /// @copydoc Deep::IComputeBackend::CopyToHost
  void CopyToHost(float* hostDst, const float* deviceSrc, size_t numFloats) noexcept override;
  /// @copydoc Deep::IComputeBackend::RandomizeNormal
  void RandomizeNormal(float* buf, size_t n, float mean, float stddev,
                       uint32_t seed) noexcept override;
  /// @copydoc Deep::IComputeBackend::RandomizeUniform
  void RandomizeUniform(float* buf, size_t n, float min, float max,
                        uint32_t seed) noexcept override;

  /// @copydoc Deep::IComputeBackend::SumRows
  void SumRows(float* dst, const float* src, size_t batchSize, size_t width) noexcept override;

  /// @brief Must be called once, before Compile()'s first
  /// BeginGraphCapture(), for any backend that needs to prepare
  /// batch-size-dependent state (e.g. CUDABackend's cached all-ones
  /// vector for SumRows' GEMV). No-op on CPUBackend.
  void PrepareForBatchSize(size_t batchSize) noexcept override;

  /// @copydoc Deep::IComputeBackend::MatMul
  void MatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda,
              const float* B, int ldb, float beta, float* C, int ldc) noexcept override;

  /// @copydoc Deep::IComputeBackend::Scale
  void Scale(float* buf, size_t n, float alpha) noexcept override;
  /// @copydoc Deep::IComputeBackend::AxpyInto
  void AxpyInto(float* y, const float* x, size_t n, float alpha) noexcept override;
  /// @copydoc Deep::IComputeBackend::AddBiasBroadcast
  void AddBiasBroadcast(float* buf, const float* bias, size_t batchSize,
                        size_t width) noexcept override;

  /// @copydoc Deep::IComputeBackend::FusedStateUpdateMomentum
  void FusedStateUpdateMomentum(float* z, float* v, const float* feedback, ActivationType dType,
                                const float* e, size_t n, float ir, float beta) noexcept override;

  /// @copydoc Deep::IComputeBackend::TryFusedForwardPass
  bool TryFusedForwardPass(ActivationType actType, const float* zF, const float* W,
                           const float* bias, float* mu, int batchSize, int size,
                           int nextSize) noexcept override;

  /// @copydoc Deep::IComputeBackend::Activation
  void Activation(ActivationType type, float* buf, size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::ActivationInto
  void ActivationInto(ActivationType type, float* dst, const float* src,
                      size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::ActivationDerivative
  void ActivationDerivative(ActivationType type, float* buf, size_t n,
                            bool activated) noexcept override;
  /// @copydoc Deep::IComputeBackend::ActivationDerivativeInto
  void ActivationDerivativeInto(ActivationType type, float* dst, const float* src,
                                size_t n) noexcept override;

  /// @copydoc Deep::IComputeBackend::FusedStateUpdate
  void FusedStateUpdate(float* z, const float* feedback, ActivationType dType, const float* e,
                        size_t n, float ir) noexcept override;
  /// @copydoc Deep::IComputeBackend::ComputeErrorAndEnergy
  float ComputeErrorAndEnergy(float* e, const float* z, const float* mu,
                              size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::ComputeError
  void ComputeError(float* e, const float* z, const float* mu, size_t n) noexcept override;

  /// @copydoc Deep::IComputeBackend::ComputeSoftmaxCrossEntropyErrorAndEnergy
  float ComputeSoftmaxCrossEntropyErrorAndEnergy(float* e, const float* z, const float* mu,
                                                 size_t batchSize, size_t nextSize,
                                                 float* rowEnergies) noexcept override;

  /// @copydoc Deep::IComputeBackend::ComputeSoftmaxCrossEntropyError
  void ComputeSoftmaxCrossEntropyError(float* e, const float* z, const float* mu, size_t batchSize,
                                       size_t nextSize) noexcept override;

  /// @copydoc Deep::IComputeBackend::IncrementCounter
  void IncrementCounter(int* ptr) noexcept override;

  /// @copydoc Deep::IComputeBackend::AdamStep
  void AdamStep(float* param, const float* grad, float* m, float* v, size_t n, const int* t,
                const float* lr, float beta1 = 0.9f, float beta2 = 0.999f,
                float eps = 1e-8f) noexcept override;
  /// @copydoc Deep::IComputeBackend::AdamWStep
  void AdamWStep(float* param, const float* grad, float* m, float* v, size_t n, const int* t,
                 const float* lr, float weightDecay, float beta1 = 0.9f, float beta2 = 0.999f,
                 float eps = 1e-8f) noexcept override;

  /// @copydoc Deep::IComputeBackend::GetDeviceType
  DeviceType GetDeviceType() const noexcept override
  {
    return DeviceType::DEVICE_GPU;
  }

  /// @copydoc Deep::IComputeBackend::MultiplyInto
  void MultiplyInto(float* dst, const float* a, const float* b, size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::FusedActivationDerivativeMultiply
  void FusedActivationDerivativeMultiply(float* dst, const float* a, float* activatedInOut,
                                         ActivationType dType, size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::Fill
  void Fill(float* buf, size_t n, float value) noexcept override;

  /// @copydoc Deep::IComputeBackend::Im2Col
  void Im2Col(const float* input, int channels, int height, int width, int kernelH, int kernelW,
              int strideH, int strideW, int padH, int padW, float* columns) noexcept override;
  /// @copydoc Deep::IComputeBackend::Col2Im
  void Col2Im(const float* columns, int channels, int height, int width, int kernelH, int kernelW,
              int strideH, int strideW, int padH, int padW, float* outputImage) noexcept override;
  /// @copydoc Deep::IComputeBackend::RepackForBatchedGemm
  void RepackForBatchedGemm(float* dst, const float* src, size_t batchSize, size_t rows,
                            size_t cols) noexcept override;
  /// @copydoc Deep::IComputeBackend::AddBiasPerChannel
  void AddBiasPerChannel(float* buf, const float* bias, size_t channels,
                         size_t spatialSize) noexcept override;

  /// @copydoc Deep::IComputeBackend::ComputePrecisionWeightedErrorAndEnergy
  float ComputePrecisionWeightedErrorAndEnergy(float* e, const float* z, const float* mu,
                                               const float* p, size_t batchSize,
                                               size_t width) noexcept override;
  /// @copydoc Deep::IComputeBackend::AxpyBroadcastInto
  void AxpyBroadcastInto(float* y, const float* x, const float* factor, size_t batchSize,
                         size_t width, float alpha) noexcept override;
  /// @copydoc Deep::IComputeBackend::MultiplyBroadcastInto
  void MultiplyBroadcastInto(float* dst, const float* a, const float* factor, const float* b,
                             size_t batchSize, size_t width) noexcept override;
  /// @copydoc Deep::IComputeBackend::UpdatePrecisionFromError
  void UpdatePrecisionFromError(float* p, float* log_p, const float* e, size_t batchSize,
                                size_t width, float pr) noexcept override;

private:
#ifdef DEEPITY_USE_CUDA
  cublasHandle_t handle;
  cudaStream_t stream;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
#endif
  bool hasGraph = false;
  float* onesVector = nullptr;
  size_t onesCapacity = 0;
  float* workspace = nullptr;
};
} // namespace Deep