#pragma once
#include <cstddef>
#include <cstdint>
#include <deepity/backend/DeviceType.h>
#include <deepity/utils/Activations.h>

namespace Deep
{

/// @brief Device-agnostic compute backend interface: every numerical
/// primitive a layer/network needs, implemented by CPUBackend (SIMD/BLAS)
/// and CUDABackend (kernels/cuBLAS/CUTLASS).
/// @tparam T Element type for this backend's data buffers (activations,
/// weights, errors, biases, ...).
/// @tparam AccumT Type used for optimizer accumulator state (AdamStep/
/// AdamWStep's grad/m/v/lr) and for scalar hyperparameters/reductions
/// (alpha, beta, eps, ir, Sum()'s return, ...). Defaults to T.
template <typename T, typename AccumT = T>
class IComputeBackendT
{
public:
  virtual ~IComputeBackendT() = default;

  // Graphs

  /// @brief Starts recording subsequent backend calls into a replayable
  /// graph instead of executing them immediately. Pair with
  /// EndGraphCapture().
  /// @param slot Which independent graph slot to (re)capture into. Each
  /// slot keeps its own graph, so independent call sequences (e.g. a
  /// training step vs. a prediction step) can each have their own
  /// without evicting one another.
  virtual void BeginGraphCapture(size_t slot = 0) noexcept = 0;
  /// @brief Ends capture and instantiates the captured graph into @p slot.
  /// @return true if ReplayGraph() is now safe to call for this slot;
  /// false otherwise. Callers must check this -- a failed capture leaves
  /// ReplayGraph() permanently doing nothing on every subsequent call for
  /// that slot.
  virtual bool EndGraphCapture(size_t slot = 0) noexcept = 0;
  /// @brief Launches the graph previously captured and instantiated into
  /// @p slot. No-op if EndGraphCapture() never returned true for that slot.
  /// Asynchronous: returns once the launch is enqueued, not once it
  /// completes. Call Synchronize() afterward if the result must be ready,
  /// e.g. before reading it back on the host.
  virtual void ReplayGraph(size_t slot = 0) noexcept = 0;
  /// @brief Blocks until all previously enqueued work on this backend has
  /// completed.
  virtual void Synchronize() noexcept = 0;

  // Memory

  /// @brief Allocates `numElements` elements of T on this backend's device.
  /// @return nullptr on allocation failure.
  virtual T* Allocate(size_t numElements) = 0;
  /// @brief Frees a buffer previously returned by Allocate(). No-op on nullptr.
  virtual void Free(T* ptr) noexcept = 0;
  /// @brief Zero-fills `numElements` elements of T starting at `ptr`.
  virtual void Zero(T* ptr, size_t numElements) noexcept = 0;
  /// @brief Copies `numElements` elements of T from `src` to `dst`, both on
  /// this backend's own device.
  virtual void Copy(T* dst, const T* src, size_t numElements) noexcept = 0;
  /// @brief Copies `numElements` elements of T from a host buffer into a
  /// device buffer.
  virtual void CopyFromHost(T* deviceDst, const T* hostSrc, size_t numElements) noexcept = 0;
  /// @brief Copies `numElements` elements of T from a device buffer into a
  /// host buffer.
  virtual void CopyToHost(T* hostDst, const T* deviceSrc, size_t numElements) noexcept = 0;
  /// @brief Fills `n` elements with samples from a normal distribution.
  /// @param buf Buffer to fill.
  /// @param n Number of elements to fill.
  /// @param mean Distribution mean.
  /// @param stddev Distribution standard deviation.
  /// @param seed RNG seed.
  virtual void RandomizeNormal(T* buf, size_t n, AccumT mean, AccumT stddev,
                               uint32_t seed) noexcept = 0;
  /// @brief Fills `n` elements with samples from a uniform distribution
  /// over [min, max).
  /// @param buf Buffer to fill.
  /// @param n Number of elements to fill.
  /// @param min Distribution lower bound (inclusive).
  /// @param max Distribution upper bound (exclusive).
  /// @param seed RNG seed.
  virtual void RandomizeUniform(T* buf, size_t n, AccumT min, AccumT max,
                                uint32_t seed) noexcept = 0;

  /// @brief Call once, before the first BeginGraphCapture(), for any
  /// backend that needs to prepare batch-size-dependent state.
  virtual void PrepareForBatchSize(size_t batchSize) noexcept = 0;

  // GEMM

  /// @brief C = alpha * op(A) * op(B) + beta * C, the standard BLAS
  /// SGEMM convention (op(X) = X^T if the matching transX is true).
  /// @param transA Whether to transpose A.
  /// @param transB Whether to transpose B.
  /// @param M Rows of op(A) and C.
  /// @param N Columns of op(B) and C.
  /// @param K Columns of op(A), rows of op(B).
  /// @param alpha Scalar multiplied into op(A)*op(B).
  /// @param A Left operand.
  /// @param lda,ldb,ldc Leading dimensions of A, B, C respectively.
  /// @param B Right operand.
  /// @param beta Scalar multiplied into the existing C before accumulating.
  /// @param C Output; also read if beta != 0.
  virtual void MatMul(bool transA, bool transB, int M, int N, int K, AccumT alpha, const T* A,
                      int lda, const T* B, int ldb, AccumT beta, T* C, int ldc) noexcept = 0;

  /// @brief Same operation as MatMul(), but writes into an AccumT* buffer
  /// instead of T*.
  virtual void MatMulAccum(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                           const T* A, int lda, const T* B, int ldb, AccumT beta, AccumT* C,
                           int ldc) noexcept = 0;

  /// @brief Same operation as MatMulAccum(), but A is already AccumT*
  /// instead of T*. Only B decodes from T; A is used as-is.
  virtual void MatMulAccumMixed(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                                const AccumT* A, int lda, const T* B, int ldb, AccumT beta,
                                AccumT* C, int ldc) noexcept = 0;

  /// @brief Whether MatMul() may use TF32 instead of full FP32 on tensor
  /// cores. CUDABackend only; a no-op on CPUBackend. Defaults to allowed.
  /// Turn off before a finite-difference gradient check, where TF32's
  /// reduced precision can mask or fail the check.
  virtual void SetAllowTF32(bool allow) noexcept = 0;

  /// @brief dst[j] = sum over b in [0,batchSize) of src[b*width + j], for
  /// all j in [0,width).
  virtual void SumRows(T* dst, const T* src, size_t batchSize, size_t width) noexcept = 0;

  /// @brief Same operation as SumRows(), but writes into an AccumT* buffer
  /// instead of T*.
  virtual void SumRowsAccum(AccumT* dst, const T* src, size_t batchSize,
                            size_t width) noexcept = 0;

  /// @brief Returns the plain (signed) sum of every element in buf.
  /// @warning Synchronous on CUDABackend: do not call this from inside a
  /// captured CUDA graph region.
  virtual AccumT Sum(const T* buf, size_t n) noexcept = 0;

  // Elementwise scalar ops

  /// @brief buf[i] *= alpha for all i in [0, n).
  virtual void Scale(T* buf, size_t n, AccumT alpha) noexcept = 0;
  /// @brief y[i] += alpha * x[i] for all i in [0, n), the standard AXPY.
  virtual void AxpyInto(T* y, const T* x, size_t n, AccumT alpha) noexcept = 0;
  /// @brief Same operation as AxpyInto(), but y is AccumT* and x stays T*.
  virtual void AxpyIntoAccum(AccumT* y, const T* x, size_t n, AccumT alpha) noexcept = 0;
  /// @brief Same operation as AxpyInto(), but x is AccumT* instead of T*.
  virtual void AxpyIntoFromAccum(T* y, const AccumT* x, size_t n, AccumT alpha) noexcept = 0;
  /// @brief buf[row*width + col] += bias[col] for every row in
  /// [0,batchSize) and col in [0,width): broadcasts a per-column bias
  /// across every row.
  virtual void AddBiasBroadcast(T* buf, const T* bias, size_t batchSize,
                                size_t width) noexcept = 0;

  /// @brief Attempts C = A @ B(^T) + bias in one fused kernel (CPU only,
  /// requires a oneDNN build).
  /// @return true if the fused path ran (C is fully computed); false if
  /// the caller should fall back to MatMul()+AddBiasBroadcast().
  /// @param transA/transB Same meaning as MatMul(). @param M/N/K/A/lda/
  /// B/ldb/C/ldc Same meaning and shapes as MatMul() (alpha=1, beta=0
  /// implied).
  /// @param bias Length-N vector, broadcast across all M rows of C
  /// (same convention as AddBiasBroadcast).
  virtual bool TryFusedMatMulBias(bool transA, bool transB, int M, int N, int K, const T* A,
                                  int lda, const T* B, int ldb, const T* bias, T* C,
                                  int ldc) noexcept = 0;

  /// @brief Attempts C = alpha * (A @ B(^T)) via a fused kernel (CPU only,
  /// requires a oneDNN build). No bias, beta=0 implied -- this cannot
  /// accumulate into an existing C.
  /// @return true if the fused path ran; false if the caller should fall
  /// back to MatMul() with the same alpha and beta=0.
  /// @param transA/transB/M/N/K/A/lda/B/ldb/C/ldc Same meaning and
  /// shapes as MatMul(). @param alpha Scalar multiplier on the product,
  /// same convention as MatMul().
  virtual bool TryFusedMatMul(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                              const T* A, int lda, const T* B, int ldb, T* C,
                              int ldc) noexcept = 0;

  /// @brief Fuses a settling layer's feedback GEMM with its own state
  /// update in one call:
  ///   feedback = eAbove @ W                    (M=batchSize,N=size,K=nextSize)
  ///   z += ir * (f'(z) * feedback - e)
  /// @return false if unsupported (e.g. this dType/shape combination has
  /// no fused path), in which case the caller falls back to its own
  /// separate TryFusedMatMul()/FusedStateUpdate() calls.
  /// @param dType Which activation's derivative to apply to z.
  /// @param batchSize/size/nextSize Standard layer shapes.
  /// @param eAbove (batchSize, nextSize) row-major, tightly packed.
  /// @param W (nextSize, size) row-major, tightly packed.
  /// @param z In/out, (batchSize, size) row-major, tightly packed. Must
  /// not alias `scratch` or `derivScratch`.
  /// @param e (batchSize, size) row-major, tightly packed.
  /// @param ir Inference rate, same meaning as FusedStateUpdate()'s ir.
  /// @param scratch Caller-owned (batchSize, size) scratch buffer; the
  /// fused result is written here before being copied into z.
  /// @param derivScratch Caller-owned (batchSize, size) scratch buffer
  /// holding f'(z).
  virtual bool TryFusedFeedbackUpdate(ActivationType dType, int batchSize, int size, int nextSize,
                                      const T* eAbove, const T* W, T* z,
                                      const T* e, AccumT ir, T* scratch,
                                      T* derivScratch) noexcept = 0;

  // Activation

  /// @brief Applies `type` to `buf` in place.
  virtual void Activation(ActivationType type, T* buf, size_t n) noexcept = 0;
  /// @brief Two-buffer variant of Activation(): reads src, writes into dst.
  virtual void ActivationInto(ActivationType type, T* dst, const T* src,
                              size_t n) noexcept = 0;
  /// @brief Applies `type`'s derivative to `buf` in place.
  /// @param type Which activation's derivative to apply.
  /// @param buf Array to derive in place.
  /// @param n Length of buf.
  /// @param activated If true, buf already holds the activated value;
  /// if false, buf holds the pre-activation value.
  virtual void ActivationDerivative(ActivationType type, T* buf, size_t n,
                                    bool activated) noexcept = 0;
  /// @brief Two-buffer variant of ActivationDerivative(): reads src
  /// (the pre-activation value), writes the derivative into dst.
  virtual void ActivationDerivativeInto(ActivationType type, T* dst, const T* src,
                                        size_t n) noexcept = 0;
  /// @brief Same operation as ActivationDerivativeInto(), but writes
  /// into an AccumT* buffer instead of T*.
  virtual void ActivationDerivativeIntoAccum(ActivationType type, AccumT* dst, const T* src,
                                             size_t n) noexcept = 0;

  // Fused PC-specific ops

  /// @brief One settling step's state update with momentum:
  ///   v = beta*v + (1-beta)*((feedback*deriv) - e)
  ///   z += ir*v
  /// @param v Momentum buffer, same shape as z, persistent across settling
  /// steps within one call, reset to zero at the start of each.
  /// @param beta EMA decay (0.9 is a common default; higher means more
  /// smoothing/inertia).
  /// @param dType Which activation's derivative to apply to z in place.
  virtual void FusedStateUpdateMomentum(T* z, T* v, const T* feedback,
                                        ActivationType dType, const T* e, size_t n, AccumT ir,
                                        AccumT beta) noexcept = 0;

  /// @brief One settling step's state update, fused into a single call:
  ///   z += ir * ((feedback * deriv) - e)
  /// @param dType Which activation's derivative to apply to z in place.
  virtual void FusedStateUpdate(T* z, const T* feedback, ActivationType dType,
                                const T* e, size_t n, AccumT ir) noexcept = 0;
  /// @brief Computes e = z - mu, then returns 0.5 * sum(e^2), the
  /// Gaussian error/energy used by every layer except a cross-entropy
  /// terminal.
  virtual AccumT ComputeErrorAndEnergy(T* e, const T* z, const T* mu,
                                       size_t n) noexcept = 0;
  /// @brief Same as ComputeErrorAndEnergy(), without computing (or
  /// returning) the energy, for callers that only need e.
  virtual void ComputeError(T* e, const T* z, const T* mu, size_t n) noexcept = 0;

  /// @brief Fuses e = z - mu with the forward activation zF = f(z) in one
  /// pass.
  /// @param activationType A forward-flavored ActivationType (e.g. SIGMOID).
  /// @param e Output, length n: z - mu.
  /// @param zF Output, length n: f(z).
  /// @param z Input, length n.
  /// @param mu Input, length n.
  /// @param n Element count (batchSize * size).
  /// @return false if this backend has no fused implementation, in
  /// which case the caller falls back to separate ComputeError()/
  /// ActivationInto() calls.
  virtual bool TryComputeErrorAndActivation(ActivationType activationType, T* e, T* zF,
                                            const T* z, const T* mu,
                                            size_t n) noexcept = 0;

  /// @brief Softmax cross-entropy variant of ComputeErrorAndEnergy, for
  /// the terminal layer only:
  ///   probs = softmax(mu), row-wise
  ///   e = z - probs
  ///   energy = -sum(z * log(probs + eps))
  /// mu holds the raw logits; softmax is applied internally.
  /// @param rowEnergies Caller-provided scratch, >= batchSize elements.
  /// Ignored by CPUBackend's own override, kept for interface parity.
  virtual AccumT ComputeSoftmaxCrossEntropyErrorAndEnergy(T* e, const T* z, const T* mu,
                                                          size_t batchSize, size_t nextSize,
                                                          AccumT* rowEnergies) noexcept = 0;

  /// @brief Same as ComputeSoftmaxCrossEntropyErrorAndEnergy(), without
  /// computing (or requiring `rowEnergies` for) the energy, for
  /// callers that only need e.
  virtual void ComputeSoftmaxCrossEntropyError(T* e, const T* z, const T* mu,
                                               size_t batchSize, size_t nextSize) noexcept = 0;

  /// @brief Attempts a fused forward pass (GEMM + bias + activation in
  /// one kernel) for RELU, LINEAR, GELU, TANH, or SIGMOID; anything else
  /// returns false immediately.
  /// @return true if the fused path was used (mu is fully computed,
  /// including bias and activation); false if the caller should fall
  /// back to MatMul()+AddBiasBroadcast()+Activation().
  /// @param actType Activation to fuse in.
  /// @param zF Activated input, shape [batchSize, size], row-major.
  /// @param W Weight matrix, shape [nextSize, size], row-major.
  /// @param bias Bias vector, shape [nextSize].
  /// @param mu Output, shape [batchSize, nextSize], row-major. Only
  /// written if this returns true.
  /// @param batchSize Number of rows in zF/mu.
  /// @param size Input width (columns of zF, columns of W).
  /// @param nextSize Output width (rows of W, columns of mu, length of bias).
  virtual bool TryFusedForwardPass(ActivationType actType, const T* zF, const T* W,
                                   const T* bias, T* mu, int batchSize, int size,
                                   int nextSize) noexcept = 0;

  // Convolution (im2col-based, ConvPCLayer family)

  /// @brief Rearranges a whole batch of (channels, height, width) input
  /// images into (channels*kH*kW, outH*outW) column matrices, the
  /// standard im2col transform, in one call. @p input is
  /// [batchSize, channels, height, width] and @p columns is
  /// [batchSize, channels*kH*kW, outH*outW], both contiguous and
  /// batch-major. Positions outside the input (due to padding) are
  /// written as zero. @p columns is fully overwritten, not accumulated
  /// into.
  virtual void Im2Col(const T* input, int batchSize, int channels, int height, int width,
                      int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                      T* columns) noexcept = 0;

  /// @brief The adjoint of Im2Col(): scatters a whole batch of
  /// (channels*kH*kW, outH*outW) column-gradient buffers back into
  /// (channels, height, width) images, in one call. Accumulates into
  /// @p outputImage (does not zero it first) -- zero the destination
  /// first for a fresh result.
  virtual void Col2Im(const T* columns, int batchSize, int channels, int height, int width,
                      int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                      T* outputImage) noexcept = 0;

  /// @brief Batched 2D max pooling, no padding. @p input is [batchSize,
  /// channels, height, width], @p output and @p argmax are [batchSize,
  /// channels, outH, outW]. @p argmax records, per output position, the
  /// winning input position's flat (row*width+col) offset within its
  /// channel plane, consumed by MaxPool2DBackward().
  virtual void MaxPool2DForward(const T* input, int batchSize, int channels, int height,
                                int width, int poolH, int poolW, int strideH, int strideW,
                                T* output, int* argmax) noexcept = 0;

  /// @brief The adjoint of MaxPool2DForward(): scatters a batch of
  /// [batchSize, channels, outH, outW] gradients back into [batchSize,
  /// channels, height, width] using the recorded @p argmax. Accumulates
  /// into @p inputGrad (does not zero it first).
  virtual void MaxPool2DBackward(const T* outputGrad, const int* argmax, int batchSize,
                                 int channels, int height, int width, int outH, int outW,
                                 T* inputGrad) noexcept = 0;

  /// @brief Repacks a [batchSize, rows, cols] tensor (batch-major) into
  /// [rows, batchSize, cols] (row-major, batch second), i.e.
  /// dst[row][batch][col] = src[batch][row][col], keeping the innermost
  /// `cols` dimension contiguous on both sides.
  virtual void RepackForBatchedGemm(T* dst, const T* src, size_t batchSize, size_t rows,
                                    size_t cols) noexcept = 0;

  /// @brief Attempts a single fused convolution forward pass: @p output =
  /// @p alpha * conv(@p input, @p W), in NCHW layout.
  /// @return false (without touching @p output) if this backend has no
  /// fused path, letting the caller fall back to the manual
  /// Im2Col()+RepackForBatchedGemm()+MatMul() sequence.
  /// @param W [outChannels, inChannels, kernelH, kernelW].
  /// @note Use AddBiasPerChannelNCHW(), not AddBiasPerChannel(), for the
  /// bias-add on this output layout.
  virtual bool TryFusedConvForward(const T* input, const T* W, T* output, AccumT alpha,
                                   int batchSize, int inChannels, int height, int width,
                                   int outChannels, int kernelH, int kernelW, int strideH,
                                   int strideW, int padH, int padW) noexcept = 0;

  /// @brief The adjoint of TryFusedConvForward() w.r.t. @p input.
  /// Overwrites @p inputGrad = @p alpha * convBackwardData(@p outputGrad,
  /// @p W) -- unlike Col2Im(), this does not accumulate; zero the
  /// destination first if accumulation is needed.
  /// @return false if this backend has no fused path.
  virtual bool TryFusedConvBackwardData(const T* outputGrad, const T* W, T* inputGrad,
                                        AccumT alpha, int batchSize, int inChannels, int height,
                                        int width, int outChannels, int kernelH, int kernelW,
                                        int strideH, int strideW, int padH, int padW) noexcept = 0;

  /// @brief The adjoint of TryFusedConvForward() w.r.t. @p W: @p dW =
  /// @p alpha * convBackwardFilter(@p input, @p outputGrad) + @p beta *
  /// @p dW, matching MatMul()'s own alpha/beta convention.
  /// @return false if this backend has no fused path.
  virtual bool TryFusedConvBackwardFilter(const T* input, const T* outputGrad, T* dW,
                                          AccumT alpha, AccumT beta, int batchSize, int inChannels,
                                          int height, int width, int outChannels, int kernelH,
                                          int kernelW, int strideH, int strideW, int padH,
                                          int padW) noexcept = 0;

  /// @brief Pre-grows this backend's conv workspace (used by
  /// TryFusedConvForward() and friends) to fit this shape. Must be
  /// called for every distinct conv shape before the first
  /// BeginGraphCapture(). No-op on backends with no such workspace.
  virtual void PrepareConvWorkspace(int batchSize, int inChannels, int height, int width,
                                    int outChannels, int kernelH, int kernelW, int strideH,
                                    int strideW, int padH, int padW) noexcept = 0;

  // Optimizer

  /// @brief Atomically increments *counter by 1.
  virtual void IncrementCounter(int* counter) noexcept = 0;

  /// @brief Adam update for `n` parameters in place.
  /// @param param Parameters to update, in storage type T.
  /// @param grad Gradient w.r.t. param, in accumulator type AccumT.
  /// @param m Adam's first-moment (momentum) buffer, persistent across calls.
  /// @param v Adam's second-moment (variance) buffer, persistent across calls.
  /// @param n Number of parameters.
  /// @param t Device-resident step count (read, not written here; see
  /// IncrementCounter()).
  /// @param lr Device-resident learning rate.
  /// @param beta1,beta2 Momentum/variance EMA decay rates.
  /// @param eps Denominator stabilizer.
  virtual void AdamStep(T* param, const AccumT* grad, AccumT* m, AccumT* v, size_t n, const int* t,
                        const AccumT* lr, AccumT beta1 = AccumT(0.9), AccumT beta2 = AccumT(0.999),
                        AccumT eps = AccumT(1e-8)) noexcept = 0;
  /// @brief Same as AdamStep(), plus decoupled weight decay (AdamW):
  /// param -= lr * weightDecay * param, applied before the Adam step itself.
  virtual void AdamWStep(T* param, const AccumT* grad, AccumT* m, AccumT* v, size_t n,
                         const int* t, const AccumT* lr, AccumT weightDecay,
                         AccumT beta1 = AccumT(0.9), AccumT beta2 = AccumT(0.999),
                         AccumT eps = AccumT(1e-8)) noexcept = 0;

  /// @brief Which device this backend instance runs on.
  virtual DeviceType GetDeviceType() const noexcept = 0;

  /// @brief Returns this backend's compute stream as an opaque,
  /// backend-defined integer handle (a cudaStream_t reinterpret_cast to
  /// std::uintptr_t, for CUDABackend), or 0 if this backend has no
  /// stream concept. Exists only so cross-framework GPU interop (e.g.
  /// importing a DLPack tensor from PyTorch/JAX) can ask the exporting
  /// framework to synchronize against the exact stream Deepity will
  /// read from, instead of racing it. No internal Deepity code path
  /// uses this, so every backend but CUDABackend keeps the default.
  virtual std::uintptr_t GetStreamHandle() const noexcept
  {
    return 0;
  }

  /// @brief dst[i] = a[i] * b[i] for all i in [0, n), elementwise product.
  virtual void MultiplyInto(T* dst, const T* a, const T* b, size_t n) noexcept = 0;
  /// @brief Same operation as MultiplyInto(), but dst and b are AccumT*
  /// (a stays T*).
  virtual void MultiplyIntoAccum(AccumT* dst, const T* a, const AccumT* b, size_t n) noexcept = 0;
  /// @brief Fuses an in-place "derivative from activated value" step with
  /// the elementwise multiply that follows it: dst[i] = a[i] *
  /// f'(activatedInOut[i]), then activatedInOut[i] is overwritten with
  /// that same derivative.
  /// @warning GELU's derivative can't be recovered from its own output,
  /// so this is approximate for dType=dGELU, same as
  /// ActivationDerivative(..., activated=true).
  /// @param dType A derivative-flavored ActivationType.
  virtual void FusedActivationDerivativeMultiply(T* dst, const T* a, T* activatedInOut,
                                                 ActivationType dType, size_t n) noexcept = 0;
  /// @brief Sibling of FusedActivationDerivativeMultiply() for callers
  /// that hold the pre-activation value instead of the activated one:
  /// dst[i] = a[i] * f'(preact[i]). preact is read-only (never
  /// overwritten). Exact for every activation type, including GELU.
  /// @param dType A derivative-flavored ActivationType.
  virtual void FusedActivationDerivativeMultiplyFromPreact(T* dst, const T* a,
                                                            const T* preact, ActivationType dType,
                                                            size_t n) noexcept = 0;
  /// @brief buf[i] = value for all i in [0, n).
  virtual void Fill(T* buf, size_t n, AccumT value) noexcept = 0;
  /// @brief Convolutional bias-add: buf[c*spatialSize + s] += bias[c] for
  /// all c in [0,channels), s in [0,spatialSize). Per-channel broadcast
  /// across spatial positions. For a shared bias across a batch, fold
  /// batchSize into @p spatialSize; for a per-batch-item bias, call once
  /// per item with @p buf offset to that item's slice.
  virtual void AddBiasPerChannel(T* buf, const T* bias, size_t channels,
                                 size_t spatialSize) noexcept = 0;

  /// @brief Convolutional bias-add for the batch-major NCHW layout:
  /// buf[b*channels*spatialSize + c*spatialSize + s] += bias[c] for all
  /// b in [0,batchSize), c in [0,channels), s in [0,spatialSize). Use
  /// this (not AddBiasPerChannel()) when batch is the outermost
  /// dimension and batchSize > 1.
  virtual void AddBiasPerChannelNCHW(T* buf, const T* bias, size_t batchSize,
                                     size_t channels, size_t spatialSize) noexcept = 0;

  // Precision-weighted PC ops (DiscriminativePCLayer/ConvPCLayer family)

  /// @brief Precision-weighted counterpart to ComputeErrorAndEnergy():
  /// e[b*width+i] = z[b*width+i] - mu[b*width+i] for every b in
  /// [0,batchSize), i in [0,width); returns
  /// sum_{b,i}(0.5*p'[i]*e^2 - 0.5*log(p'[i])), where p'[i] =
  /// max(p[i], 1e-8). @p p has length `width`, one precision per
  /// own-position, broadcast across every batch item (not length
  /// batchSize*width like e/z/mu).
  virtual AccumT ComputePrecisionWeightedErrorAndEnergy(T* e, const T* z, const T* mu,
                                                        const T* p, size_t batchSize,
                                                        size_t width) noexcept = 0;
  /// @brief Same as ComputePrecisionWeightedErrorAndEnergy(), without
  /// computing (or returning) the energy, for callers that only need e.
  virtual void ComputePrecisionWeightedError(T* e, const T* z, const T* mu,
                                             const T* p, size_t batchSize,
                                             size_t width) noexcept = 0;

  /// @brief y[b*width+i] += alpha * x[b*width+i] * factor[i], for every b
  /// in [0,batchSize), i in [0,width): AxpyInto with an added
  /// per-position factor broadcast across the batch dimension.
  virtual void AxpyBroadcastInto(T* y, const T* x, const T* factor, size_t batchSize,
                                 size_t width, AccumT alpha) noexcept = 0;

  /// @brief dst[b*width+i] = a[b*width+i] * factor[i] * b[b*width+i], for
  /// every batch item: a three-operand elementwise product with one
  /// operand (factor) broadcast across the batch dimension.
  virtual void MultiplyBroadcastInto(T* dst, const T* a, const T* factor,
                                     const T* b, size_t batchSize, size_t width) noexcept = 0;

  /// @brief Precision update from accumulated squared error: for each i
  /// in [0,width), grad = mean_{b in [0,batchSize)} of
  /// 0.5*(p[i]*e[b*width+i]^2 - 1); log_p[i] -= pr*grad, clamped to
  /// [-5,5]; p[i] = exp(log_p[i]).
  virtual void UpdatePrecisionFromError(T* p, T* log_p, const T* e, size_t batchSize,
                                        size_t width, AccumT pr) noexcept = 0;
};

/// @brief Non-templated alias: `IComputeBackend` is `IComputeBackendT<float,
/// float>`.
using IComputeBackend = IComputeBackendT<float, float>;

} // namespace Deep
