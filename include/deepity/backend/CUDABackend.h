#pragma once
#include <deepity/backend/IComputeBackend.h>
#ifdef DEEPITY_USE_CUDA
#include <cublas_v2.h>
#ifdef DEEPITY_USE_CUDNN
#include <cudnn.h>
#endif
#endif
#include <deepity/backend/Tensor.h>
#include <vector>

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
  void BeginGraphCapture(size_t slot = 0) noexcept override;
  /// @copydoc Deep::IComputeBackend::EndGraphCapture
  bool EndGraphCapture(size_t slot = 0) noexcept override;
  /// @copydoc Deep::IComputeBackend::ReplayGraph
  void ReplayGraph(size_t slot = 0) noexcept override;
  /// @copydoc Deep::IComputeBackend::Synchronize
  void Synchronize() noexcept override;

  /// @copydoc Deep::IComputeBackend::Allocate
  float* Allocate(size_t numFloats) override;
  /// @copydoc Deep::IComputeBackend::Free
  void Free(float* ptr) noexcept override;
  /// @copydoc Deep::IComputeBackend::Zero
  void Zero(float* ptr, size_t numFloats) noexcept override;
  /// @copydoc Deep::IComputeBackend::Copy
  void Copy(float* dst, const float* src, size_t numFloats) noexcept override;
  void Copy2D(float* dst, size_t dstStride, const float* src, size_t srcStride, size_t rows,
             size_t cols) noexcept override;
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

  /// @brief Benchmark-only: runs RandomizeNormal()'s host-generate +
  /// cudaMemcpy path unconditionally, bypassing the size threshold that
  /// picks between it and the curand-kernel path. Exists so
  /// tCUDARandomizeBenchmark.cpp can time both paths at the same n and
  /// find the real crossover point for issue #67. Not part of
  /// IComputeBackend; not meant for production call sites.
  void RandomizeNormalForceHostPath(float* buf, size_t n, float mean, float stddev,
                                    uint32_t seed) noexcept;
  /// @brief Benchmark-only counterpart of RandomizeNormalForceHostPath()
  /// that always runs the curand-kernel path.
  void RandomizeNormalForceGPUPath(float* buf, size_t n, float mean, float stddev,
                                   uint32_t seed) noexcept;
  /// @copydoc RandomizeNormalForceHostPath
  void RandomizeUniformForceHostPath(float* buf, size_t n, float min, float max,
                                     uint32_t seed) noexcept;
  /// @copydoc RandomizeNormalForceGPUPath
  void RandomizeUniformForceGPUPath(float* buf, size_t n, float min, float max,
                                    uint32_t seed) noexcept;

  /// @copydoc Deep::IComputeBackend::SumRows
  void SumRows(float* dst, const float* src, size_t batchSize, size_t width) noexcept override;
  /// @copydoc Deep::IComputeBackendT::SumRowsAccum
  void SumRowsAccum(float* dst, const float* src, size_t batchSize, size_t width) noexcept override
  {
    SumRows(dst, src, batchSize, width);
  }
  /// @copydoc Deep::IComputeBackend::Sum
  float Sum(const float* buf, size_t n) noexcept override;

  /// @copydoc Deep::IComputeBackend::PrepareForBatchSize
  void PrepareForBatchSize(size_t batchSize) noexcept override;

  /// @copydoc Deep::IComputeBackend::PrewarmBatchedGemmScratch
  bool PrewarmBatchedGemmScratch(size_t batchCount) noexcept override;

  /// @copydoc Deep::IComputeBackend::MatMul
  void MatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A, int lda,
              const float* B, int ldb, float beta, float* C, int ldc) noexcept override;
  /// @copydoc Deep::IComputeBackendT::MatMulAccum
  void MatMulAccum(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
                   int lda, const float* B, int ldb, float beta, float* C,
                   int ldc) noexcept override
  {
    MatMul(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  /// @copydoc Deep::IComputeBackendT::MatMulAccumMixed
  void MatMulAccumMixed(bool transA, bool transB, int M, int N, int K, float alpha,
                        const float* A, int lda, const float* B, int ldb, float beta, float* C,
                        int ldc) noexcept override
  {
    MatMul(transA, transB, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  /// @copydoc Deep::IComputeBackend::SetAllowTF32
  void SetAllowTF32(bool allow) noexcept override;

  /// @copydoc Deep::IComputeBackendT::MatMulBatched
  /// @note Requires identical M/N/K across the whole batch (classic
  /// cublasSgemmBatched's own requirement) -- true for every caller
  /// today, since each passes the same M/N/K for every layer by
  /// construction, but NOT yet exploiting a network with genuinely
  /// ragged per-layer widths (would need cublasGemmGroupedBatchedEx).
  /// @note Falls back to a plain per-layer MatMul() loop whenever any of
  /// M/N/K is below 128 -- cuBLASLt's heuristic picks a 128x128-tile
  /// kernel that writes past the output buffer for small dimensions
  /// (confirmed via compute-sanitizer), same class of bug as
  /// CPUBackend::MatMulBatched's own small-matrix guard.
  void MatMulBatched(bool transA, bool transB, int M, int N, int K, float alpha,
                     const float* const* A_array, int lda, const float* const* B_array, int ldb,
                     float beta, float* const* C_array, int ldc, int batchCount) noexcept override;

  /// @copydoc Deep::IComputeBackend::Scale
  void Scale(float* buf, size_t n, float alpha) noexcept override;
  /// @copydoc Deep::IComputeBackend::AxpyInto
  void AxpyInto(float* y, const float* x, size_t n, float alpha) noexcept override;
  /// @copydoc Deep::IComputeBackendT::AxpyIntoAccum
  void AxpyIntoAccum(float* y, const float* x, size_t n, float alpha) noexcept override
  {
    AxpyInto(y, x, n, alpha);
  }
  /// @copydoc Deep::IComputeBackendT::AxpyIntoFromAccum
  void AxpyIntoFromAccum(float* y, const float* x, size_t n, float alpha) noexcept override
  {
    AxpyInto(y, x, n, alpha);
  }
  /// @copydoc Deep::IComputeBackend::AddBiasBroadcast
  void AddBiasBroadcast(float* buf, const float* bias, size_t batchSize,
                        size_t width) noexcept override;
  /// @copydoc Deep::IComputeBackend::TryFusedMatMulBias
  bool TryFusedMatMulBias(bool transA, bool transB, int M, int N, int K, const float* A, int lda,
                          const float* B, int ldb, const float* bias, float* C,
                          int ldc) noexcept override
  {
    (void)transA, (void)transB, (void)M, (void)N, (void)K, (void)A, (void)lda, (void)B, (void)ldb,
        (void)bias, (void)C, (void)ldc;
    return false;
  }

  /// @copydoc Deep::IComputeBackend::TryFusedMatMul
  bool TryFusedMatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
                      int lda, const float* B, int ldb, float* C, int ldc) noexcept override
  {
    (void)transA, (void)transB, (void)M, (void)N, (void)K, (void)alpha, (void)A, (void)lda,
        (void)B, (void)ldb, (void)C, (void)ldc;
    return false;
  }

  /// @copydoc Deep::IComputeBackend::TryFusedFeedbackUpdate
  bool TryFusedFeedbackUpdate(ActivationType dType, int batchSize, int size, int nextSize,
                              const float* eAbove, const float* W, float* z, const float* e,
                              float ir, float* scratch, float* derivScratch) noexcept override
  {
    (void)dType, (void)batchSize, (void)size, (void)nextSize, (void)eAbove, (void)W, (void)z,
        (void)e, (void)ir, (void)scratch, (void)derivScratch;
    return false;
  }

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
  /// @copydoc Deep::IComputeBackendT::ActivationDerivativeIntoAccum
  void ActivationDerivativeIntoAccum(ActivationType type, float* dst, const float* src,
                                     size_t n) noexcept override
  {
    ActivationDerivativeInto(type, dst, src, n);
  }

  /// @copydoc Deep::IComputeBackend::FusedStateUpdate
  void FusedStateUpdate(float* z, const float* feedback, ActivationType dType, const float* e,
                        size_t n, float ir) noexcept override;
  /// @copydoc Deep::IComputeBackend::ComputeErrorAndEnergy
  float ComputeErrorAndEnergy(float* e, const float* z, const float* mu,
                              size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::ComputeError
  void ComputeError(float* e, const float* z, const float* mu, size_t n) noexcept override;

  /// @copydoc Deep::IComputeBackend::TryComputeErrorAndActivation
  bool TryComputeErrorAndActivation(ActivationType activationType, float* e, float* zF,
                                    const float* z, const float* mu, size_t n) noexcept override
  {
    (void)activationType, (void)e, (void)zF, (void)z, (void)mu, (void)n;
    return false;
  }

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
  /// @copydoc Deep::IComputeBackendT::MultiplyIntoAccum
  void MultiplyIntoAccum(float* dst, const float* a, const float* b, size_t n) noexcept override
  {
    MultiplyInto(dst, a, b, n);
  }
  /// @copydoc Deep::IComputeBackend::FusedActivationDerivativeMultiply
  void FusedActivationDerivativeMultiply(float* dst, const float* a, float* activatedInOut,
                                         ActivationType dType, size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::FusedActivationDerivativeMultiplyFromPreact
  void FusedActivationDerivativeMultiplyFromPreact(float* dst, const float* a, const float* preact,
                                                    ActivationType dType, size_t n) noexcept override;
  /// @copydoc Deep::IComputeBackend::Fill
  void Fill(float* buf, size_t n, float value) noexcept override;

  /// @copydoc Deep::IComputeBackend::Im2Col
  void Im2Col(const float* input, int batchSize, int channels, int height, int width, int kernelH,
              int kernelW, int strideH, int strideW, int padH, int padW,
              float* columns) noexcept override;
  /// @copydoc Deep::IComputeBackend::Col2Im
  void Col2Im(const float* columns, int batchSize, int channels, int height, int width, int kernelH,
              int kernelW, int strideH, int strideW, int padH, int padW,
              float* outputImage) noexcept override;
  /// @copydoc Deep::IComputeBackend::MaxPool2DForward
  void MaxPool2DForward(const float* input, int batchSize, int channels, int height, int width,
                        int poolH, int poolW, int strideH, int strideW, float* output,
                        int* argmax) noexcept override;
  /// @copydoc Deep::IComputeBackend::MaxPool2DBackward
  void MaxPool2DBackward(const float* outputGrad, const int* argmax, int batchSize, int channels,
                         int height, int width, int outH, int outW,
                         float* inputGrad) noexcept override;
  /// @copydoc Deep::IComputeBackend::RepackForBatchedGemm
  void RepackForBatchedGemm(float* dst, const float* src, size_t batchSize, size_t rows,
                            size_t cols) noexcept override;
  /// @copydoc Deep::IComputeBackend::TryFusedConvForward
  bool TryFusedConvForward(const float* input, const float* W, float* output, float alpha,
                           int batchSize, int inChannels, int height, int width, int outChannels,
                           int kernelH, int kernelW, int strideH, int strideW, int padH,
                           int padW) noexcept override;
  /// @copydoc Deep::IComputeBackend::TryFusedConvBackwardData
  bool TryFusedConvBackwardData(const float* outputGrad, const float* W, float* inputGrad,
                                float alpha, int batchSize, int inChannels, int height, int width,
                                int outChannels, int kernelH, int kernelW, int strideH, int strideW,
                                int padH, int padW) noexcept override;
  /// @copydoc Deep::IComputeBackend::TryFusedConvBackwardFilter
  bool TryFusedConvBackwardFilter(const float* input, const float* outputGrad, float* dW,
                                  float alpha, float beta, int batchSize, int inChannels,
                                  int height, int width, int outChannels, int kernelH, int kernelW,
                                  int strideH, int strideW, int padH, int padW) noexcept override;
  /// @copydoc Deep::IComputeBackend::PrepareConvWorkspace
  void PrepareConvWorkspace(int batchSize, int inChannels, int height, int width, int outChannels,
                            int kernelH, int kernelW, int strideH, int strideW, int padH,
                            int padW) noexcept override;
  /// @copydoc Deep::IComputeBackend::AddBiasPerChannel
  void AddBiasPerChannel(float* buf, const float* bias, size_t channels,
                         size_t spatialSize) noexcept override;
  /// @copydoc Deep::IComputeBackend::AddBiasPerChannelNCHW
  void AddBiasPerChannelNCHW(float* buf, const float* bias, size_t batchSize, size_t channels,
                             size_t spatialSize) noexcept override;

  /// @copydoc Deep::IComputeBackend::ComputePrecisionWeightedErrorAndEnergy
  float ComputePrecisionWeightedErrorAndEnergy(float* e, const float* z, const float* mu,
                                               const float* p, size_t batchSize,
                                               size_t width) noexcept override;
  /// @copydoc Deep::IComputeBackend::ComputePrecisionWeightedError
  void ComputePrecisionWeightedError(float* e, const float* z, const float* mu, const float* p,
                                     size_t batchSize, size_t width) noexcept override;
  /// @copydoc Deep::IComputeBackend::AxpyBroadcastInto
  void AxpyBroadcastInto(float* y, const float* x, const float* factor, size_t batchSize,
                         size_t width, float alpha) noexcept override;
  /// @copydoc Deep::IComputeBackend::MultiplyBroadcastInto
  void MultiplyBroadcastInto(float* dst, const float* a, const float* factor, const float* b,
                             size_t batchSize, size_t width) noexcept override;
  /// @copydoc Deep::IComputeBackend::UpdatePrecisionFromError
  void UpdatePrecisionFromError(float* p, float* log_p, const float* e, size_t batchSize,
                                size_t width, float pr) noexcept override;
  /// @copydoc Deep::IComputeBackend::GetStreamHandle
  std::uintptr_t GetStreamHandle() const noexcept override;

private:
#ifdef DEEPITY_USE_CUDA
  cublasHandle_t handle;
  cudaStream_t stream;
  /// @brief One entry per graph slot, indexed by BeginGraphCapture()/
  /// EndGraphCapture()/ReplayGraph()'s @p slot.
  struct GraphSlot
  {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graphExec = nullptr;
    bool hasGraph = false;
  };
  std::vector<GraphSlot> graphSlots;
#ifdef DEEPITY_USE_CUDNN
  cudnnHandle_t cudnnHandle;
#endif
#endif
  float* onesVector = nullptr;
  size_t onesCapacity = 0;
  float* workspace = nullptr;
  /// @brief Current byte capacity of @p workspace.
  size_t workspaceBytes = 0;
  /// @brief Last value passed to SetAllowTF32().
  bool allowTF32 = true;

  /// @brief Single-float device scratch buffer for scalar reductions
  /// (ComputePrecisionWeightedErrorAndEnergy(), Sum()).
  float* scalarScratch = nullptr;

  /// @brief Device-resident pointer arrays for cublasSgemmBatched()'s
  /// Aarray/Barray/Carray parameters: cuBLAS requires these arrays
  /// THEMSELVES, not just what they point to, to live in device memory
  /// -- unlike CPUBackend's cblas_sgemm_batch (a host array of host
  /// pointers). Grown lazily, reused across calls, freed in the
  /// destructor, the same pattern onesVector/workspace already use.
  const float** batchedGemmDeviceA = nullptr;
  const float** batchedGemmDeviceB = nullptr;
  float** batchedGemmDeviceC = nullptr;
  /// @brief Pinned (cudaMallocHost) host staging buffers paired with
  /// the three device arrays above: MatMulBatched() copies the caller's
  /// host pointer arrays in here, then cudaMemcpyAsync()s to device on
  /// `stream`. Needed because a plain synchronous cudaMemcpy() (as
  /// CopyFromHost() uses) isn't safely capturable inside
  /// BeginGraphCapture()/EndGraphCapture(), and ordinary pageable host
  /// memory isn't a valid cudaMemcpyAsync() source during capture either.
  const float** batchedGemmHostA = nullptr;
  const float** batchedGemmHostB = nullptr;
  float** batchedGemmHostC = nullptr;
  /// @brief Current capacity, in pointer slots, of EACH of the six
  /// arrays above (they always grow together, all-or-nothing).
  size_t batchedGemmArrayCapacity = 0;

  /// @brief Grows (if needed) the six batched-GEMM pointer-array
  /// buffers above to hold at least @p batchCount pointers each.
  /// All-or-nothing: on any allocation failure, frees everything and
  /// returns false rather than leave a partially-grown, mismatched set.
  bool GrowBatchedGemmArrays(size_t batchCount) noexcept;

  /// @brief Device-resident problem-size/stride metadata for
  /// MatMulBatched()'s CUTLASS GemmGrouped dispatch path, used once
  /// M/N/K clear kMinSafeDim (see CUDABackendGemm.cu's own comment):
  /// cublasSgemmBatched() decomposes into roughly one GPU kernel
  /// instance per batch item internally, so its wall-clock time still
  /// scales with batchCount even though CPU-side dispatch is O(1) --
  /// confirmed via nsys (Total GPU Ops scaling ~linearly with depth,
  /// per-op time staying flat). CUTLASS's Grouped GEMM genuinely fuses
  /// the whole batch into one kernel launch instead. Stored as raw
  /// int/int64 arrays, not cutlass::gemm::GemmCoord*, so this header
  /// doesn't need to include CUTLASS -- CUDABackendGemm.cu
  /// reinterpret_casts groupedProblemSizes to GemmCoord*, guarded by a
  /// static_assert on GemmCoord's layout there. Unlike
  /// batchedGemmDeviceA/B/C (which may legitimately point to new
  /// buffers every call), these depend only on shape, which is
  /// constant for a given network's whole CUDA-graph-captured
  /// lifetime -- so they're only refilled when (M,N,K,lda,ldb,ldc,
  /// batchCount) changes from the cached values, not every call.
  int* groupedProblemSizes = nullptr;
  int64_t* groupedLda = nullptr;
  int64_t* groupedLdb = nullptr;
  int64_t* groupedLdc = nullptr;
  size_t groupedMetaCapacity = 0;
  int cachedGroupedM = -1, cachedGroupedN = -1, cachedGroupedK = -1;
  int64_t cachedGroupedLda = -1, cachedGroupedLdb = -1, cachedGroupedLdc = -1;
  size_t cachedGroupedBatchCount = 0;
  /// @brief CUTLASS GemmGrouped's own required scratch (tile-scheduling
  /// metadata), sized via GemmGroupedOp::get_workspace_size(). Grown
  /// lazily like everything else above.
  uint8_t* groupedWorkspace = nullptr;
  size_t groupedWorkspaceBytes = 0;

  /// @brief Grows/refills the grouped-GEMM metadata arrays above for
  /// the given shape, if (M,N,K,lda,ldb,ldc,batchCount) differs from
  /// the cached values. Returns false (leaving the cache untouched) on
  /// any allocation failure, so the caller can fall back safely.
  bool GrowGroupedGemmMetadata(size_t batchCount, int M, int N, int K, int64_t lda, int64_t ldb,
                                int64_t ldc) noexcept;
};
} // namespace Deep