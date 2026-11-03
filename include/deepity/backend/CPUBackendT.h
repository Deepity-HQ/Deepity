#pragma once
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/Tensor.h>
#include <deepity/utils/activations/ActivationMacros.h> // RESTRICT
#include <stdexcept>

namespace Deep
{

/// @brief Templated CPU implementation of IComputeBackendT<T, AccumT>,
/// used by every templated layer/network in the "Templating Deepity's
/// layers/networks over a numeric type" plan: FullPCLayerT,
/// SimplePCLayerT, GaussSeidelPCLayerT, DirectKPPCLayerT, and the conv
/// family (SimpleConvPCLayerT/ConvPCLayerT/FullConvPCLayerT). Not used
/// by, and not a replacement for, the existing non-templated
/// CPUBackend -- DiscriminativePCLayer (not part of this plan) and the
/// non-templated originals of every layer above keep using CPUBackend
/// exactly as before.
///
/// Implements every IComputeBackendT method an actual templated layer
/// calls, with real per-T behavior: for T=float, every one of those
/// reuses the EXACT SAME dispatch this project's existing CPUBackend
/// uses (cblas_sgemm, the Highway-dispatched kernels in
/// CPUBackendHighway.*, or -- for the conv-only data-movement/broadcast
/// primitives below, which have no existing BLAS/Highway-dispatched
/// equivalent to call into -- a scalar loop whose per-element op order
/// matches the original's own scalar/AVX-intrinsic version exactly), so
/// e.g. FullPCLayer<float>'s and SimpleConvPCLayer<float>'s output is
/// bit-exact against their pre-templating originals -- see this
/// project's quantization plan's verification gate #1. For any other T
/// (double, and the quantized formats), those same methods fall back to
/// a plain, correct, non-Highway-vectorized scalar loop; genuinely
/// vectorizing them (via Highway's own T-generic `hn::ScalableTag<T>`
/// kernels) is flagged as a deliberate, documented follow-up, not
/// attempted in this pass.
///
/// The "Try*" oneDNN/libxsmm/Halide/cuDNN-flavored fused paths are
/// unreachable from every templated layer above (none of them call
/// any Try* method; see SimplePCLayerT.h's own class comment on why,
/// and note the non-templated CPUBackend's own TryFusedConv*() always
/// return false too -- there never was a CPU-side fused conv path to
/// match in the first place). They return false, their own
/// interface's well-defined "no fused path here" contract.
/// FusedActivationDerivativeMultiply[FromPreact] remain throw-stubs:
/// no current layer, templated or not, calls either method anymore
/// (see their own stub message).
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
  void SumRowsAccum(AccumT* dst, const T* src, size_t batchSize, size_t width) noexcept override;
  AccumT Sum(const T* buf, size_t n) noexcept override;

  void Scale(T* buf, size_t n, AccumT alpha) noexcept override;
  void AxpyInto(T* y, const T* x, size_t n, AccumT alpha) noexcept override;
  void AxpyIntoAccum(AccumT* y, const T* x, size_t n, AccumT alpha) noexcept override;
  void AxpyIntoFromAccum(T* y, const AccumT* x, size_t n, AccumT alpha) noexcept override;
  void AddBiasBroadcast(T* buf, const T* bias, size_t batchSize, size_t width) noexcept override;

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
    throw std::logic_error(
        "CPUBackendT::FusedStateUpdateMomentum: out of scope -- SimplePCLayer has no momentum "
        "option, so SimplePCLayerT never calls this either.");
  }
  void FusedStateUpdate(T*, const T*, ActivationType, const T*, size_t, AccumT) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::FusedStateUpdate: out of scope -- SimplePCLayerT computes this update "
        "manually (AccumT-precision feedback/derivative, see SimplePCLayerT.h's own class "
        "comment), never through this T*-typed fused entry point.");
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
    throw std::logic_error(
        "CPUBackendT::FusedActivationDerivativeMultiply: out of scope -- no current layer "
        "(templated or not) calls this method anymore; SimpleConvPCLayer/FullConvPCLayer each "
        "moved to an unfused ActivationDerivativeInto()+MultiplyInto() pair instead.");
  }
  void FusedActivationDerivativeMultiplyFromPreact(T*, const T*, const T*, ActivationType,
                                                    size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::FusedActivationDerivativeMultiplyFromPreact: out of scope, see "
        "FusedActivationDerivativeMultiply()'s own message.");
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
