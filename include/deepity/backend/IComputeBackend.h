#pragma once
#include <cstddef>
#include <deepity/utils/Activations.h>
#include <deepity/backend/DeviceType.h>

namespace Deep
{

    class IComputeBackend
    {
    public:
        virtual ~IComputeBackend() = default;

        // Graphs

        virtual void BeginGraphCapture() noexcept = 0;
        /// @brief Ends capture and instantiates the captured graph.
        /// @return true if capture and instantiation both succeeded and
        /// ReplayGraph() is now safe to call; false otherwise. Callers
        /// MUST check this -- silently assuming success here was the
        /// cause of a real bug: a failed capture left ReplayGraph()
        /// permanently doing nothing on every subsequent call, since the
        /// caller had no way to know capture never actually happened.
        virtual bool EndGraphCapture() noexcept = 0;
        virtual void ReplayGraph() noexcept = 0;

        // Memory

        virtual float *Allocate(size_t numFloats) = 0;
        virtual void Free(float *ptr) noexcept = 0;
        virtual void Zero(float *ptr, size_t numFloats) noexcept = 0;
        virtual void Copy(float *dst, const float *src, size_t numFloats) noexcept = 0;
        virtual void CopyFromHost(float *deviceDst, const float *hostSrc, size_t numFloats) noexcept = 0;
        virtual void CopyToHost(float *hostDst, const float *deviceSrc, size_t numFloats) noexcept = 0;
        virtual void RandomizeNormal(float *buf, size_t n, float mean, float stddev, uint32_t seed) noexcept = 0;
        virtual void RandomizeUniform(float *buf, size_t n, float min, float max, uint32_t seed) noexcept = 0;

        /// @brief Must be called once, before Compile()'s first
        /// BeginGraphCapture(), for any backend that needs to prepare
        /// batch-size-dependent state (e.g. CUDABackend's cached all-ones
        /// vector for SumRows' GEMV). No-op on CPUBackend.
        virtual void PrepareForBatchSize(size_t batchSize) noexcept = 0;

        // GEMM

        virtual void MatMul(bool transA, bool transB,
                            int M, int N, int K,
                            float alpha, const float *A, int lda,
                            const float *B, int ldb,
                            float beta, float *C, int ldc) noexcept = 0;

        /// @brief dst[j] = sum over b in [0,batchSize) of src[b*width + j], for
        /// all j in [0,width). Replaces a batchSize-iteration loop of
        /// individual AxpyInto calls -- the reduction-direction counterpart to
        /// AddBiasBroadcast, still unfixed until now. On GPU this is one
        /// cublasSgemv call against a cached all-ones vector, reinterpreting
        /// src's row-major [batchSize,width] layout as column-major
        /// [width,batchSize] with no data movement (verified numerically).
        virtual void SumRows(float *dst, const float *src, size_t batchSize, size_t width) noexcept = 0;

        // Elementwise scalar ops

        virtual void Scale(float *buf, size_t n, float alpha) noexcept = 0;
        virtual void AxpyInto(float *y, const float *x, size_t n, float alpha) noexcept = 0;
        virtual void AddBiasBroadcast(float *buf, const float *bias, size_t batchSize, size_t width) noexcept = 0;

        // Activation

        virtual void Activation(ActivationType type, float *buf, size_t n) noexcept = 0;
        virtual void ActivationInto(ActivationType type, float *dst, const float *src, size_t n) noexcept = 0;
        virtual void ActivationDerivative(ActivationType type, float *buf, size_t n, bool activated) noexcept = 0;
        virtual void ActivationDerivativeInto(ActivationType type, float *dst, const float *src, size_t n) noexcept = 0;

        // Fused PC-specific ops

        virtual void FusedStateUpdate(float *z, const float *feedback, const float *deriv,
                                      const float *e, size_t n, float ir) noexcept = 0;
        virtual float ComputeErrorAndEnergy(float *e, const float *z, const float *mu, size_t n) noexcept = 0;
        virtual void ComputeError(float *e, const float *z, const float *mu, size_t n) noexcept = 0;

        // Optimizer

        virtual void IncrementCounter(int *counter) noexcept = 0;

        virtual void AdamStep(float *param, const float *grad, float *m, float *v,
                              size_t n, const int *t, const float *lr,
                              float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept = 0;
        virtual void AdamWStep(float *param, const float *grad, float *m, float *v,
                               size_t n, const int *t, const float *lr, float weightDecay,
                               float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept = 0;

        virtual DeviceType GetDeviceType() const noexcept = 0;
    };
}