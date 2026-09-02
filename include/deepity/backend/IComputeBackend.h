#pragma once
#include <cstddef>
#include <deepity/utils/Activations.h>

/**
 * @file IComputeBackend.h
 * @brief Abstract interface for every core numerical operation a PC
 * layer needs, so that CPUBackend and CUDABackend can be swapped behind
 * one pointer -- no #ifdef DEEPITY_USE_CUDA anywhere except inside
 * Backend.cpp's single factory function.
 *
 * Derived directly from what SimplePCLayer/DirectKPPCLayer actually call
 * today (CalculateState, ComputeMuOnly, UpdateState, UpdateWeights,
 * DirectFeedbackUpdate), not a generic "BLAS wrapper" -- a few operations
 * below are kept as single, explicit, FUSED methods (FusedStateUpdate,
 * ComputeErrorAndEnergy) specifically because decomposing them into
 * separate elementwise-multiply/subtract/saxpy calls would lose the
 * fusion benefit this codebase already relies on on CPU, and would cost
 * even more on GPU (extra global-memory round-trips between kernel
 * launches, where memory bandwidth is usually the real bottleneck).
 *
 * Device is fixed at Tensor/allocation time (see Tensor.h) -- nothing
 * here supports moving a live buffer between CPU and GPU after creation.
 *
 * @warning This is a first draft, not yet implemented by either
 * CPUBackend or CUDABackend. Several signatures are marked below as
 * open questions -- confirm/adjust before treating this as final.
 */

namespace Deep
{
    class IComputeBackend
    {
    public:
        virtual ~IComputeBackend() = default;

        // --- Memory -----------------------------------------------------

        virtual float *Allocate(size_t numFloats) = 0;
        virtual void Free(float *ptr) noexcept = 0;
        virtual void Zero(float *ptr, size_t numFloats) noexcept = 0;

        /// @brief Device-to-device copy (both buffers already live on
        /// this backend's device).
        virtual void Copy(float *dst, const float *src, size_t numFloats) noexcept = 0;

        /// @brief Host-to-device copy. No-op memcpy on CPUBackend;
        /// cudaMemcpyHostToDevice on CUDABackend. This is the ONLY thing
        /// needed for "load weights saved on CPU, run on GPU"; see the
        /// constructor-time initialization pattern discussed separately.
        virtual void CopyFromHost(float *deviceDst, const float *hostSrc, size_t numFloats) noexcept = 0;

        /// @brief Device-to-host copy, e.g. for Save()/inspection.
        virtual void CopyToHost(float *hostDst, const float *deviceSrc, size_t numFloats) noexcept = 0;

        // --- GEMM ---------------------------------------------------------

        virtual void MatMul(bool transA, bool transB,
                            int M, int N, int K,
                            float alpha, const float *A, int lda,
                            const float *B, int ldb,
                            float beta, float *C, int ldc) noexcept = 0;

        // --- Elementwise scalar ops ---------------------------------------

        /// @brief buf *= alpha (cblas_sscal equivalent). Used for weight
        /// decay (W *= 1-lambda) today.
        virtual void Scale(float *buf, size_t n, float alpha) noexcept = 0;

        /// @brief y += alpha * x (cblas_saxpy equivalent). Used for bias
        /// adds and Adam/AdamW's own internal accumulation today.
        virtual void AxpyInto(float *y, const float *x, size_t n, float alpha) noexcept = 0;

        // --- Activation -----------------------------------------------

        /// @brief In-place activation, matching Deep::relu/sigmoid/etc's
        /// existing single-buffer signature.
        virtual void Activation(ActivationType type, float *buf, size_t n) noexcept = 0;

        /// @brief Two-buffer activation: reads src, writes phi(src) into
        /// dst, src left untouched. Matches ComputeMuOnly()'s zF = phi(z)
        /// pattern without needing a separate scopy first.
        virtual void ActivationInto(ActivationType type, float *dst, const float *src, size_t n) noexcept = 0;

        /// @brief In-place derivative, matching Deep::dRelu/dSigmoid/etc's
        /// existing (buf, n, activated) signature.
        virtual void ActivationDerivative(ActivationType type, float *buf, size_t n, bool activated) noexcept = 0;

        /// @brief Two-buffer derivative: reads RAW src, writes f'(src)
        /// into dst. Matches the dReluInto/dSigmoidInto/etc family added
        /// to Activations.h this session.
        virtual void ActivationDerivativeInto(ActivationType type, float *dst, const float *src, size_t n) noexcept = 0;

        // --- Fused PC-specific ops -------

        /// @brief z[i] += ir * (feedback[i] * deriv[i] - e[i]), for all i
        /// in [0, n). Matches SimplePCLayer/DirectKPPCLayer's UpdateState()
        /// fused settling-step update exactly.
        virtual void FusedStateUpdate(float *z, const float *feedback, const float *deriv,
                                      const float *e, size_t n, float ir) noexcept = 0;

        /// @brief e[i] = z[i] - mu[i], for all i in [0, n); returns
        /// 0.5 * sum(e[i]^2). Matches CalculateState()'s error+energy
        /// computation exactly (the fused AVX loop from earlier tonight).
        virtual float ComputeErrorAndEnergy(float *e, const float *z, const float *mu, size_t n) noexcept = 0;

        // --- Optimizer ---------------------------------------------------

        virtual void AdamStep(float *param, const float *grad, float *m, float *v,
                              size_t n, int t, float lr,
                              float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept = 0;
        virtual void AdamWStep(float *param, const float *grad, float *m, float *v,
                               size_t n, int t, float lr, float weightDecay,
                               float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept = 0;
    };
}