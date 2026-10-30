#pragma once
#include <deepity/backend/IComputeBackend.h>
#include <deepity/backend/Tensor.h>
#include <stdexcept>

namespace Deep
{

/// @brief Templated CPU implementation of IComputeBackendT<T, AccumT>,
/// used exclusively by the new templated FullPCLayer<T>/FullPCNetwork<T>
/// path (see QuantFormat.h and the "Templating Deepity's layers/networks
/// over a numeric type" plan). Deliberately NOT used by, and NOT a
/// replacement for, the existing non-templated CPUBackend -- every
/// other layer (SimplePCLayer, DirectKPPCLayer, GaussSeidelPCLayer, the
/// conv family) keeps using CPUBackend exactly as before.
///
/// Only implements the ~20 IComputeBackendT methods FullPCLayer<T>/
/// FullPCNetwork<T> actually call (MatMul, the elementwise/activation
/// ops, Adam, graph-capture no-ops, ...) with real per-T behavior: for
/// T=float, every one of those reuses the EXACT SAME dispatch this
/// project's existing CPUBackend uses (cblas_sgemm, the Highway-
/// dispatched kernels in CPUBackendHighway.*), so FullPCLayer<float>'s
/// output is bit-exact against the pre-templating FullPCLayer -- see
/// this project's quantization plan's verification gate #1. For any
/// other T (double, and eventually the quantized formats), those same
/// methods fall back to a plain, correct, non-Highway-vectorized scalar
/// loop; genuinely vectorizing them (via Highway's own T-generic
/// `hn::ScalableTag<T>` kernels) is flagged as a deliberate, documented
/// follow-up, not attempted in this pass.
///
/// Every OTHER IComputeBackendT method -- the "Try*" fused paths
/// (oneDNN/libxsmm/Halide/cuDNN-flavored; FullPCLayer never calls any
/// of them, only SimplePCLayer/the conv family do) and the conv/
/// precision-weighted-layer-only ops (Im2Col, Col2Im, MaxPool2D*,
/// RepackForBatchedGemm, AddBiasPerChannel*, ComputePrecisionWeighted*,
/// AxpyBroadcastInto, MultiplyBroadcastInto, UpdatePrecisionFromError)
/// -- is unreachable from FullPCLayer<T>/FullPCNetwork<T>. The "Try*"
/// ones return false, their own interface's well-defined "no fused path
/// here" contract; the conv/precision-weighted ones throw
/// std::logic_error naming this class's scope, rather than shipping an
/// unexercised, unverified implementation of logic nothing in this pass
/// actually calls or tests (the conv family and DiscriminativePCLayer/
/// ConvPCLayer stay on the non-templated CPUBackend, matching the
/// plan's explicit non-goals).
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
        "CPUBackendT::FusedStateUpdateMomentum: out of scope -- only SimplePCLayer uses this, "
        "and SimplePCLayer stays on the non-templated CPUBackend in this templating pass.");
  }
  void FusedStateUpdate(T*, const T*, ActivationType, const T*, size_t, AccumT) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::FusedStateUpdate: out of scope -- only SimplePCLayer uses this, and "
        "SimplePCLayer stays on the non-templated CPUBackend in this templating pass.");
  }

  AccumT ComputeErrorAndEnergy(T* e, const T* z, const T* mu, size_t n) noexcept override;
  void ComputeError(T* e, const T* z, const T* mu, size_t n) noexcept override;

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

  void Im2Col(const T*, int, int, int, int, int, int, int, int, int, int, T*) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::Im2Col: out of scope -- the conv layer family stays on the "
        "non-templated CPUBackend in this templating pass.");
  }
  void Col2Im(const T*, int, int, int, int, int, int, int, int, int, int, T*) noexcept override
  {
    throw std::logic_error("CPUBackendT::Col2Im: out of scope, see Im2Col()'s own message.");
  }
  void MaxPool2DForward(const T*, int, int, int, int, int, int, int, int, T*,
                        int*) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::MaxPool2DForward: out of scope, see Im2Col()'s own message.");
  }
  void MaxPool2DBackward(const T*, const int*, int, int, int, int, int, int, T*) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::MaxPool2DBackward: out of scope, see Im2Col()'s own message.");
  }
  void RepackForBatchedGemm(T*, const T*, size_t, size_t, size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::RepackForBatchedGemm: out of scope, see Im2Col()'s own message.");
  }

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
        "CPUBackendT::FusedActivationDerivativeMultiply: out of scope -- only "
        "SimpleConvPCLayer/FullConvPCLayer use this, and the conv family stays on the "
        "non-templated CPUBackend in this templating pass.");
  }
  void FusedActivationDerivativeMultiplyFromPreact(T*, const T*, const T*, ActivationType,
                                                    size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::FusedActivationDerivativeMultiplyFromPreact: out of scope, see "
        "FusedActivationDerivativeMultiply()'s own message.");
  }
  void Fill(T*, size_t, AccumT) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::Fill: out of scope -- FullPCLayer<T>/FullPCNetwork<T> never call this "
        "in this templating pass.");
  }
  void AddBiasPerChannel(T*, const T*, size_t, size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::AddBiasPerChannel: out of scope, see Im2Col()'s own message.");
  }
  void AddBiasPerChannelNCHW(T*, const T*, size_t, size_t, size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::AddBiasPerChannelNCHW: out of scope, see Im2Col()'s own message.");
  }

  AccumT ComputePrecisionWeightedErrorAndEnergy(T*, const T*, const T*, const T*, size_t,
                                                size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::ComputePrecisionWeightedErrorAndEnergy: out of scope -- only "
        "DiscriminativePCLayer/ConvPCLayer use this, and they stay on the non-templated "
        "CPUBackend in this templating pass.");
  }
  void ComputePrecisionWeightedError(T*, const T*, const T*, const T*, size_t,
                                     size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::ComputePrecisionWeightedError: out of scope, see "
        "ComputePrecisionWeightedErrorAndEnergy()'s own message.");
  }
  void AxpyBroadcastInto(T*, const T*, const T*, size_t, size_t, AccumT) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::AxpyBroadcastInto: out of scope, see "
        "ComputePrecisionWeightedErrorAndEnergy()'s own message.");
  }
  void MultiplyBroadcastInto(T*, const T*, const T*, const T*, size_t, size_t) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::MultiplyBroadcastInto: out of scope, see "
        "ComputePrecisionWeightedErrorAndEnergy()'s own message.");
  }
  void UpdatePrecisionFromError(T*, T*, const T*, size_t, size_t, AccumT) noexcept override
  {
    throw std::logic_error(
        "CPUBackendT::UpdatePrecisionFromError: out of scope, see "
        "ComputePrecisionWeightedErrorAndEnergy()'s own message.");
  }
};

} // namespace Deep
