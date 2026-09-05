#include <deepity/backend/CPUBackend.h>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdexcept>

#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

namespace Deep
{
    float *CPUBackend::Allocate(size_t numFloats) noexcept
    {
#ifdef _WIN32
        return (float *)_aligned_alloc(numFloats * sizeof(float), 16);
#else
        return (float *)aligned_alloc(16, numFloats);
#endif
    }

    void CPUBackend::Free(float *ptr) noexcept
    {
#ifdef _WIN32
        _aligned_free(ptr);
#else
        free(ptr);
#endif
    }

    void CPUBackend::Zero(float *ptr, size_t numFloats) noexcept
    {
        memset(ptr, 0, numFloats * sizeof(float));
    }

    void CPUBackend::Copy(float *dst, const float *src, size_t numFloats) noexcept
    {
        memcpy(dst, src, numFloats * sizeof(float));
    }

    void CPUBackend::CopyFromHost(float *deviceDst, const float *hostSrc, size_t numFloats) noexcept
    {
        memcpy(deviceDst, hostSrc, numFloats);
    }

    void CPUBackend::CopyToHost(float *hostDst, const float *deviceSrc, size_t numFloats) noexcept
    {
        memcpy(hostDst, deviceSrc, numFloats);
    }

    void CPUBackend::MatMul(bool transA, bool transB, int M, int N, int K,
                            float alpha, const float *A, int lda,
                            const float *B, int ldb,
                            float beta, float *C, int ldc) noexcept
    { // Multithreading is the caller's job.
        cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans, transB ? CblasTrans : CblasNoTrans,
                    M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    }

    void CPUBackend::Scale(float *buf, size_t n, float alpha) noexcept
    {
        cblas_sscal(n, alpha, buf, 1);
    }

    void CPUBackend::AxpyInto(float *y, const float *x, size_t n, float alpha) noexcept
    {
        cblas_saxpy(n, alpha, x, 1, y, 1);
    }

    void CPUBackend::Activation(ActivationType type, float *buf, size_t n) noexcept
    {
        To_Fn(type)(buf, n);
    }

    void CPUBackend::ActivationInto(ActivationType type, float *dst, const float *src, size_t n) noexcept
    {
        cblas_scopy(n, src, 1, dst, 1);
        To_Fn(type)(dst, n);
    }

    void CPUBackend::ActivationDerivative(ActivationType type, float *buf, size_t n, bool activated) noexcept
    {
        To_dFn(type)(buf, n, activated);
    }

    void CPUBackend::ActivationDerivativeInto(ActivationType type, float *dst, const float *src, size_t n) noexcept
    {
        To_dFn2(type)(dst, src, n);
    }

    void CPUBackend::FusedStateUpdate(float *z, const float *feedback, const float *deriv,
                                      const float *e, size_t n, float ir) noexcept
    { // TODO: Unknown if this is how it should look?
#pragma omp parallel for schedule(static) if (n > 4 && !omp_in_parallel())
        for (size_t i = 0; i < n; ++i)
        {
            z[i] += ir * ((feedback[i] * deriv[i]) - e[i]);
        }
    }

    float CPUBackend::ComputeErrorAndEnergy(float *e, const float *z, const float *mu, size_t n) noexcept
    {
        float energy = 0.0f;

#pragma omp parallel for reduction(+ : energy) schedule(static) if (n > 256 && !omp_in_parallel())
        for (size_t i = 0; i < n; ++i)
        {
            float err = z[i] - mu[i];
            e[i] = err;
            energy += err * err;
        }

        return 0.5f * energy;
    }

    void CPUBackend::AdamStep(float *param, const float *grad, float *m, float *v,
                              size_t n, int t, float lr,
                              float beta1, float beta2, float eps) noexcept
    {
        // Precompute bias correction
        float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(t));
        float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(t));
        float step_size = lr * Sleef_sqrtf(beta2_t) / beta1_t;

#pragma omp parallel for schedule(static) if (n > 256 && !omp_in_parallel())
        for (size_t i = 0; i < n; ++i)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

            param[i] -= step_size * m[i] / (std::sqrt(v[i]) + eps);
        }
    }

    void CPUBackend::AdamWStep(float *param, const float *grad, float *m, float *v,
                               size_t n, int t, float lr, float weightDecay,
                               float beta1, float beta2, float eps) noexcept
    {
        float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(t));
        float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(t));
        float step_size = lr * Sleef_sqrtf(beta2_t) / beta1_t;

#pragma omp parallel for schedule(static) if (n > 256 && !omp_in_parallel())
        for (size_t i = 0; i < n; ++i)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

            param[i] -= lr * weightDecay * param[i];
            param[i] -= step_size * m[i] / (Sleef_sqrtf(v[i]) + eps);
        }
    }
}