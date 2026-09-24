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

        /// @brief Attempts a fused forward pass (GEMM + bias + activation in
        /// one kernel, via CUTLASS on GPU) for RELU or LINEAR activation types
        /// only -- see CUDABackend's implementation for why other activation
        /// types aren't attempted yet. Returns true if the fused path was used
        /// (mu is fully computed, including bias and activation); false if the
        /// caller should fall back to the existing MatMul+AddBiasBroadcast+
        /// Activation sequence (always false on CPUBackend -- CPU has no fused
        /// path, this is a GPU-only optimization).
        /// @param zF Activated input, shape [batchSize, size], row-major.
        /// @param W Weight matrix, shape [nextSize, size], row-major.
        /// @param bias Bias vector, shape [nextSize].
        /// @param mu Output, shape [batchSize, nextSize], row-major. Only
        /// written if this returns true.
        virtual bool TryFusedForwardPass(ActivationType actType,
                                        const float *zF, const float *W, const float *bias,
                                        float *mu, int batchSize, int size, int nextSize) noexcept = 0;

        // Convolution (im2col-based, ConvPCLayer family)

        /// @brief Rearranges a single (channels, height, width) input
        /// image into a (channels*kH*kW, outH*outW) column matrix --
        /// the standard im2col transform. NOT batch-aware: call once
        /// per batch item, with @p input and @p columns offset to that
        /// item's slice, matching Deep::Im2Col's own documented
        /// contract exactly (CPUBackend forwards to it directly).
        /// Positions outside the input (due to padding) are written as
        /// zero. @p columns is fully overwritten, not accumulated into.
        virtual void Im2Col(const float *input,
                            int channels, int height, int width,
                            int kernelH, int kernelW,
                            int strideH, int strideW,
                            int padH, int padW,
                            float *columns) noexcept = 0;

        /// @brief The adjoint of Im2Col(): scatters a
        /// (channels*kH*kW, outH*outW) column-gradient buffer back into
        /// a (channels, height, width) image. NOT batch-aware, same
        /// per-item-offset contract as Im2Col(). ACCUMULATES into
        /// @p outputImage (does not zero it first) -- caller must zero
        /// the destination if a fresh result is wanted, matching
        /// Deep::Col2Im's own contract exactly.
        virtual void Col2Im(const float *columns,
                            int channels, int height, int width,
                            int kernelH, int kernelW,
                            int strideH, int strideW,
                            int padH, int padW,
                            float *outputImage) noexcept = 0;

        /// @brief Repacks a [batchSize, rows, cols] tensor (batch-major)
        /// into [rows, batchSize, cols] (row-major, batch second) --
        /// i.e. dst[row][batch][col] = src[batch][row][col] for all
        /// row/batch/col, with the innermost `cols` dimension kept
        /// contiguous on both sides. Used by ConvPCLayer-family layers
        /// to reorganize per-batch im2col columns (and per-batch
        /// upstream-error columns) into the single flat layout a plain
        /// (non-batched) GEMM call needs, rather than one GEMM call per
        /// batch item. Same "port first, optimize" position as
        /// AddBiasBroadcast/SumRows: a future strided-batched-GEMM path
        /// could remove the need for this entirely, but this matches
        /// the existing, verified CPU repacking loops exactly for now.
        virtual void RepackForBatchedGemm(float *dst, const float *src,
                                          size_t batchSize, size_t rows, size_t cols) noexcept = 0;

        // Optimizer

        virtual void IncrementCounter(int *counter) noexcept = 0;

        virtual void AdamStep(float *param, const float *grad, float *m, float *v,
                              size_t n, const int *t, const float *lr,
                              float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept = 0;
        virtual void AdamWStep(float *param, const float *grad, float *m, float *v,
                               size_t n, const int *t, const float *lr, float weightDecay,
                               float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f) noexcept = 0;

        virtual DeviceType GetDeviceType() const noexcept = 0;

        virtual void MultiplyInto(float *dst, const float *a, const float *b, size_t n) noexcept = 0;
        virtual void Fill(float *buf, size_t n, float value) noexcept = 0;
        /// @brief Convolutional bias-add: buf[c*spatialSize + s] += bias[c] for
        /// all c in [0,channels), s in [0,spatialSize). Per-CHANNEL broadcast
        /// across spatial positions -- the transpose relationship to
        /// AddBiasBroadcast (which broadcasts a per-COLUMN bias across ROWS,
        /// the dense-layer convention). NOT batch-aware: matches Im2Col/
        /// Col2Im's contract exactly -- call once per batch item, with @p buf
        /// offset to that item's slice.
        virtual void AddBiasPerChannel(float *buf, const float *bias,
                                       size_t channels, size_t spatialSize) noexcept = 0;
    };
}
