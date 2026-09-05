#pragma once
#include <deepity/backend/IComputeBackend.h>

namespace Deep
{
    class CPUBackend : public IComputeBackend
    {
    public:
        CPUBackend() = default;
        ~CPUBackend() override = default;

        float *Allocate(size_t numFloats) override;
        void Free(float *ptr) noexcept override;
        void Zero(float *ptr, size_t numFloats) noexcept override;
        void Copy(float *dst, const float *src, size_t numFloats) noexcept override;
        void CopyFromHost(float *deviceDst, const float *hostSrc, size_t numFloats) noexcept override;
        void CopyToHost(float *hostDst, const float *deviceSrc, size_t numFloats) noexcept override;

        void MatMul(bool transA, bool transB, int M, int N, int K,
                    float alpha, const float *A, int lda,
                    const float *B, int ldb,
                    float beta, float *C, int ldc) noexcept override;

        void Scale(float *buf, size_t n, float alpha) noexcept override;
        void AxpyInto(float *y, const float *x, size_t n, float alpha) noexcept override;
        void Activation(ActivationType type, float *buf, size_t n) noexcept override;
        void ActivationInto(ActivationType type, float *dst, const float *src, size_t n) noexcept override;
        void ActivationDerivative(ActivationType type, float *buf, size_t n, bool activated) noexcept override;
        void ActivationDerivativeInto(ActivationType type, float *dst, const float *src, size_t n) noexcept override;

        void FusedStateUpdate(float *z, const float *feedback, const float *deriv,
                              const float *e, size_t n, float ir) noexcept override;
        float ComputeErrorAndEnergy(float *e, const float *z, const float *mu, size_t n) noexcept override;

        void AdamStep(float *param, const float *grad, float *m, float *v,
                      size_t n, int t, float lr,
                      float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept override;
        void AdamWStep(float *param, const float *grad, float *m, float *v,
                       size_t n, int t, float lr, float weightDecay,
                       float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept override;
    };
}