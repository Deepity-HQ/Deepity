#pragma once
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/Tensor.h>
#include <deepity/utils/activations/ActivationMacros.h> // RESTRICT
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace Deep
{

/// @brief Templated CPU implementation of IComputeBackendT<T, AccumT>,
/// used by templated layers/networks (FullPCLayerT, SimplePCLayerT,
/// GaussSeidelPCLayerT, DirectKPPCLayerT, and the conv family).
/// Separate from, and not a drop-in replacement for, the non-templated
/// CPUBackend, which DiscriminativePCLayer and non-templated layers
/// still use.
///
/// For T=float, GEMM/activation/elementwise methods dispatch through
/// the same BLAS/Highway kernels as CPUBackend. For other T (double,
/// quantized formats), the same methods fall back to a plain scalar
/// loop.
///
/// Every "Try*" fused-path method (TryFusedMatMul, TryFusedConv*, ...)
/// always returns false: no fused CPU implementation is available
/// through this backend, so callers must use the corresponding
/// unfused sequence. FusedStateUpdate(), FusedStateUpdateMomentum(),
/// FusedActivationDerivativeMultiply(), and
/// FusedActivationDerivativeMultiplyFromPreact() are not implemented
/// and throw if called; see each method's own exception message.
template <typename T, typename AccumT = T>
class CPUBackendT : public IComputeBackendT<T, AccumT>
{
public:
  CPUBackendT() = default;
  ~CPUBackendT() override = default;

  void BeginGraphCapture(size_t slot = 0) noexcept override
  {
    (void)slot;
  }
  bool EndGraphCapture(size_t slot = 0) noexcept override
  {
    (void)slot;
    return true;
  }
  void ReplayGraph(size_t slot = 0) noexcept override
  {
    (void)slot;
  }
  void Synchronize() noexcept override {}
  void PrepareForBatchSize(size_t) noexcept override {}
  bool PrewarmBatchedGemmScratch(size_t) noexcept override { return true; }
  bool PrewarmGroupedGemmScratch(bool, bool, int, int, int, int64_t, int64_t, int64_t,
                                 size_t) noexcept override
  {
    return true;
  }
  void SetAllowTF32(bool) noexcept override {}

  T* Allocate(size_t numElements) override;
  void Free(T* ptr) noexcept override;
  void Zero(T* ptr, size_t numElements) noexcept override;
  void Copy(T* dst, const T* src, size_t numElements) noexcept override;
  void CopyFromHost(T* deviceDst, const T* hostSrc, size_t numElements) noexcept override;
  void CopyToHost(T* hostDst, const T* deviceSrc, size_t numElements) noexcept override;
  void RandomizeNormal(T* buf, size_t n, AccumT mean, AccumT stddev, uint32_t seed) noexcept override;
  void RandomizeUniform(T* buf, size_t n, AccumT min, AccumT max, uint32_t seed) noexcept override;

  void MatMul(bool transA, bool transB, int M, int N, int K, AccumT alpha, const T* A, int lda,
              const T* B, int ldb, AccumT beta, T* C, int ldc) noexcept override;
  void MatMulAccum(bool transA, bool transB, int M, int N, int K, AccumT alpha, const T* A,
                   int lda, const T* B, int ldb, AccumT beta, AccumT* C,
                   int ldc) noexcept override;
  void MatMulAccumMixed(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                        const AccumT* A, int lda, const T* B, int ldb, AccumT beta, AccumT* C,
                        int ldc) noexcept override;

  void SumRows(T* dst, const T* src, size_t batchSize, size_t width) noexcept override;
  void CenterRows(T* buf, size_t rows, size_t cols) noexcept override;
  void SumRowsBatched(T* dst, const T* src, size_t batchSize, size_t width,
                      size_t numLayers) noexcept override
  {
    for (size_t l = 0; l < numLayers; ++l)
      SumRows(dst + l * width, src + l * batchSize * width, batchSize, width);
  }
  void SumRowsAccum(AccumT* dst, const T* src, size_t batchSize, size_t width) noexcept override;
  AccumT Sum(const T* buf, size_t n) noexcept override;

  void Scale(T* buf, size_t n, AccumT alpha) noexcept override;
  void AxpyInto(T* y, const T* x, size_t n, AccumT alpha) noexcept override;
  void AxpyIntoAccum(AccumT* y, const T* x, size_t n, AccumT alpha) noexcept override;
  void AxpyIntoFromAccum(T* y, const AccumT* x, size_t n, AccumT alpha) noexcept override;
  void AddBiasBroadcast(T* buf, const T* bias, size_t batchSize, size_t width) noexcept override;
  void AddBiasBroadcastBatched(T* buf, const T* bias, size_t batchSize, size_t width,
                               size_t numLayers) noexcept override
  {
    for (size_t l = 0; l < numLayers; ++l)
      AddBiasBroadcast(buf + l * batchSize * width, bias + l * width, batchSize, width);
  }

  bool TryFusedMatMulBias(bool, bool, int, int, int, const T*, int, const T*, int, const T*, T*,
                          int) noexcept override
  {
    return false;
  }
  bool TryFusedMatMul(bool, bool, int, int, int, AccumT, const T*, int, const T*, int, T*,
                      int) noexcept override
  {
    return false;
  }
  bool TryFusedFeedbackUpdate(ActivationType, int, int, int, const T*, const T*, T*, const T*,
                              AccumT, T*, T*) noexcept override
  {
    return false;
  }
  bool TryFusedForwardPass(ActivationType, const T*, const T*, const T*, T*, int, int,
                           int) noexcept override
  {
    return false;
  }

  void Activation(ActivationType type, T* buf, size_t n) noexcept override;
  void ActivationInto(ActivationType type, T* dst, const T* src, size_t n) noexcept override;
  void ActivationDerivative(ActivationType type, T* buf, size_t n, bool activated) noexcept override;
  void ActivationDerivativeInto(ActivationType type, T* dst, const T* src, size_t n) noexcept override;
  void ActivationDerivativeIntoAccum(ActivationType type, AccumT* dst, const T* src,
                                     size_t n) noexcept override;

  void FusedStateUpdateMomentum(T*, T*, const T*, ActivationType, const T*, size_t, AccumT,
                                AccumT) noexcept override
  {
    std::fprintf(stderr,
                 "CPUBackendT::FusedStateUpdateMomentum: out of scope -- SimplePCLayer has no "
                 "momentum option, so SimplePCLayerT never calls this either.\n");
    std::abort();
  }
  void FusedStateUpdate(T*, const T*, ActivationType, const T*, size_t, AccumT) noexcept override
  {
    std::fprintf(stderr,
                 "CPUBackendT::FusedStateUpdate: out of scope -- SimplePCLayerT computes this "
                 "update manually (AccumT-precision feedback/derivative, see SimplePCLayerT.h's "
                 "own class comment), never through this T*-typed fused entry point.\n");
    std::abort();
  }

  AccumT ComputeErrorAndEnergy(T* RESTRICT e, const T* RESTRICT z, const T* RESTRICT mu,
                                size_t n) noexcept override;
  void ComputeError(T* RESTRICT e, const T* RESTRICT z, const T* RESTRICT mu,
                     size_t n) noexcept override;

  bool TryComputeErrorAndActivation(ActivationType, T*, T*, const T*, const T*,
                                    size_t) noexcept override
  {
    return false;
  }

  AccumT ComputeSoftmaxCrossEntropyErrorAndEnergy(T* e, const T* z, const T* mu, size_t batchSize,
                                                  size_t nextSize,
                                                  AccumT* rowEnergies) noexcept override;
  void ComputeSoftmaxCrossEntropyError(T* e, const T* z, const T* mu, size_t batchSize,
                                       size_t nextSize) noexcept override;

  void Im2Col(const T* input, int batchSize, int channels, int height, int width, int kernelH,
             int kernelW, int strideH, int strideW, int padH, int padW,
             T* columns) noexcept override;
  void Col2Im(const T* columns, int batchSize, int channels, int height, int width, int kernelH,
             int kernelW, int strideH, int strideW, int padH, int padW,
             T* outputImage) noexcept override;
  void MaxPool2DForward(const T* input, int batchSize, int channels, int height, int width,
                        int poolH, int poolW, int strideH, int strideW, T* output,
                        int* argmax) noexcept override;
  void MaxPool2DBackward(const T* outputGrad, const int* argmax, int batchSize, int channels,
                         int height, int width, int outH, int outW,
                         T* inputGrad) noexcept override;
  void RepackForBatchedGemm(T* dst, const T* src, size_t batchSize, size_t rows,
                           size_t cols) noexcept override;

  bool TryFusedConvForward(const T*, const T*, T*, AccumT, int, int, int, int, int, int, int, int,
                           int, int, int) noexcept override
  {
    return false;
  }
  bool TryFusedConvBackwardData(const T*, const T*, T*, AccumT, int, int, int, int, int, int, int,
                                int, int, int, int) noexcept override
  {
    return false;
  }
  bool TryFusedConvBackwardFilter(const T*, const T*, T*, AccumT, AccumT, int, int, int, int, int,
                                  int, int, int, int, int, int) noexcept override
  {
    return false;
  }
  void PrepareConvWorkspace(int, int, int, int, int, int, int, int, int, int, int) noexcept override
  {
  }

  void IncrementCounter(int* counter) noexcept override;
  void AdamStep(T* param, const AccumT* grad, AccumT* m, AccumT* v, size_t n, const int* t,
               const AccumT* lr, AccumT beta1 = AccumT(0.9), AccumT beta2 = AccumT(0.999),
               AccumT eps = AccumT(1e-8)) noexcept override;
  void AdamWStep(T* param, const AccumT* grad, AccumT* m, AccumT* v, size_t n, const int* t,
                const AccumT* lr, AccumT weightDecay, AccumT beta1 = AccumT(0.9),
                AccumT beta2 = AccumT(0.999), AccumT eps = AccumT(1e-8)) noexcept override;

  DeviceType GetDeviceType() const noexcept override
  {
    return DeviceType::DEVICE_CPU;
  }

  void MultiplyInto(T* dst, const T* a, const T* b, size_t n) noexcept override;
  void MultiplyIntoAccum(AccumT* dst, const T* a, const AccumT* b, size_t n) noexcept override;

  void FusedActivationDerivativeMultiply(T*, const T*, T*, ActivationType, size_t) noexcept override
  {
    std::fprintf(stderr,
                 "CPUBackendT::FusedActivationDerivativeMultiply: out of scope -- no current "
                 "layer (templated or not) calls this method anymore; "
                 "SimpleConvPCLayer/FullConvPCLayer each moved to an unfused "
                 "ActivationDerivativeInto()+MultiplyInto() pair instead.\n");
    std::abort();
  }
  void FusedActivationDerivativeMultiplyFromPreact(T*, const T*, const T*, ActivationType,
                                                    size_t) noexcept override
  {
    std::fprintf(stderr,
                 "CPUBackendT::FusedActivationDerivativeMultiplyFromPreact: out of scope, see "
                 "FusedActivationDerivativeMultiply()'s own message.\n");
    std::abort();
  }
  void Fill(T* buf, size_t n, AccumT value) noexcept override;
  void AddBiasPerChannel(T* buf, const T* bias, size_t channels,
                        size_t spatialSize) noexcept override;
  void AddBiasPerChannelNCHW(T* buf, const T* bias, size_t batchSize, size_t channels,
                             size_t spatialSize) noexcept override;

  AccumT ComputePrecisionWeightedErrorAndEnergy(T* e, const T* z, const T* mu, const T* p,
                                                size_t batchSize,
                                                size_t width) noexcept override;
  void ComputePrecisionWeightedError(T* e, const T* z, const T* mu, const T* p, size_t batchSize,
                                     size_t width) noexcept override;
  void AxpyBroadcastInto(T* y, const T* x, const T* factor, size_t batchSize, size_t width,
                        AccumT alpha) noexcept override;
  void MultiplyBroadcastInto(T* dst, const T* a, const T* factor, const T* b, size_t batchSize,
                             size_t width) noexcept override;
  void UpdatePrecisionFromError(T* p, T* log_p, const T* e, size_t batchSize, size_t width,
                                AccumT pr) noexcept override;
};

} // namespace Deep
