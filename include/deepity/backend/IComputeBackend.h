#pragma once
#include <cstddef>
#include <deepity/backend/DeviceType.h>
#include <deepity/utils/Activations.h>

namespace Deep
{

/// @brief Device-agnostic compute backend interface: every numerical
/// primitive a layer/network needs, implemented once each by CPUBackend
/// (SIMD/BLAS) and CUDABackend (kernels/cuBLAS/CUTLASS), so layer code
/// never branches on device directly.
class IComputeBackend
{
public:
  virtual ~IComputeBackend() = default;

  // Graphs

  /// @brief Starts recording subsequent backend calls into a replayable
  /// graph, instead of executing them immediately (CUDA graph capture;
  /// a no-op record on CPUBackend). Pair with EndGraphCapture().
  virtual void BeginGraphCapture() noexcept = 0;
  /// @brief Ends capture and instantiates the captured graph.
  /// @return true if capture and instantiation both succeeded and
  /// ReplayGraph() is now safe to call; false otherwise. Callers
  /// MUST check this, silently assuming success here was the
  /// cause of a real bug: a failed capture left ReplayGraph()
  /// permanently doing nothing on every subsequent call, since the
  /// caller had no way to know capture never actually happened.
  virtual bool EndGraphCapture() noexcept = 0;
  /// @brief Launches a graph previously captured and instantiated by
  /// BeginGraphCapture()/EndGraphCapture(). No-op if EndGraphCapture()
  /// never returned true. cudaGraphLaunch() is asynchronous: this
  /// returns as soon as the launch is ENQUEUED, not once it completes.
  /// Ordinarily that's fine (CUDA guarantees same-stream ordering, so a
  /// later call that needs the result just needs to be queued on the
  /// same stream too) -- but call Synchronize() afterward if you need
  /// the replayed work to have definitely finished, e.g. before reading
  /// results back on the host through any path that ISN'T itself a
  /// synchronous backend call.
  virtual void ReplayGraph() noexcept = 0;
  /// @brief Blocks until all previously enqueued work on this backend's
  /// stream has completed. A no-op on CPUBackend (every CPU call is
  /// already synchronous).
  virtual void Synchronize() noexcept = 0;

  // Memory

  /// @brief Allocates `numFloats` floats on this backend's device.
  /// @return nullptr on allocation failure.
  virtual float* Allocate(size_t numFloats) = 0;
  /// @brief Frees a buffer previously returned by Allocate(). No-op on nullptr.
  virtual void Free(float* ptr) noexcept = 0;
  /// @brief Zero-fills `numFloats` floats starting at `ptr`.
  virtual void Zero(float* ptr, size_t numFloats) noexcept = 0;
  /// @brief Copies `numFloats` floats from `src` to `dst`, both on this
  /// backend's own device.
  virtual void Copy(float* dst, const float* src, size_t numFloats) noexcept = 0;
  /// @brief Copies `numFloats` floats from a host buffer into a device
  /// buffer (a plain memcpy on CPUBackend).
  virtual void CopyFromHost(float* deviceDst, const float* hostSrc, size_t numFloats) noexcept = 0;
  /// @brief Copies `numFloats` floats from a device buffer into a host
  /// buffer (a plain memcpy on CPUBackend).
  virtual void CopyToHost(float* hostDst, const float* deviceSrc, size_t numFloats) noexcept = 0;
  /// @brief Fills `n` floats with samples from a normal distribution.
  /// @param buf Buffer to fill.
  /// @param n Number of floats to fill.
  /// @param mean Distribution mean.
  /// @param stddev Distribution standard deviation.
  /// @param seed RNG seed.
  virtual void RandomizeNormal(float* buf, size_t n, float mean, float stddev,
                               uint32_t seed) noexcept = 0;
  /// @brief Fills `n` floats with samples from a uniform distribution
  /// over [min, max).
  /// @param buf Buffer to fill.
  /// @param n Number of floats to fill.
  /// @param min Distribution lower bound (inclusive).
  /// @param max Distribution upper bound (exclusive).
  /// @param seed RNG seed.
  virtual void RandomizeUniform(float* buf, size_t n, float min, float max,
                                uint32_t seed) noexcept = 0;

  /// @brief Must be called once, before Compile()'s first
  /// BeginGraphCapture(), for any backend that needs to prepare
  /// batch-size-dependent state (e.g. CUDABackend's cached all-ones
  /// vector for SumRows' GEMV). No-op on CPUBackend.
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
  virtual void MatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
                      int lda, const float* B, int ldb, float beta, float* C, int ldc) noexcept = 0;

  /// @brief Whether MatMul() may use TF32 (10 mantissa bits, ~1e-3
  /// relative error per GEMM) instead of full FP32 on tensor cores --
  /// CUDABackend only; a no-op on CPUBackend, which has no such
  /// distinction. Defaults to allowed, matching PyTorch's own default,
  /// since every conv in this library lowers to im2col+MatMul and would
  /// otherwise run at a fraction of the GPU's peak throughput. Turn off
  /// before a finite-difference gradient check: TF32's error is large
  /// enough to either fail those checks for no real reason, or tempt
  /// loosening their tolerance until it stops catching real bugs.
  virtual void SetAllowTF32(bool allow) noexcept = 0;

  /// @brief dst[j] = sum over b in [0,batchSize) of src[b*width + j], for
  /// all j in [0,width). Replaces a batchSize-iteration loop of
  /// individual AxpyInto calls, the reduction-direction counterpart to
  /// AddBiasBroadcast, still unfixed until now. On GPU this is one
  /// cublasSgemv call against a cached all-ones vector, reinterpreting
  /// src's row-major [batchSize,width] layout as column-major
  /// [width,batchSize] with no data movement (verified numerically).
  virtual void SumRows(float* dst, const float* src, size_t batchSize, size_t width) noexcept = 0;

  /// @brief Returns the plain (signed) sum of every element in buf.
  /// Synchronous on CUDABackend (reads the reduction back to a host
  /// float before returning) -- same constraint as ComputeErrorAndEnergy
  /// and friends: never call this from inside a captured CUDA graph
  /// region, only from code that runs outside any
  /// BeginGraphCapture()/EndGraphCapture() pair.
  virtual float Sum(const float* buf, size_t n) noexcept = 0;

  // Elementwise scalar ops

  /// @brief buf[i] *= alpha for all i in [0, n).
  virtual void Scale(float* buf, size_t n, float alpha) noexcept = 0;
  /// @brief y[i] += alpha * x[i] for all i in [0, n), the standard AXPY.
  virtual void AxpyInto(float* y, const float* x, size_t n, float alpha) noexcept = 0;
  /// @brief buf[row*width + col] += bias[col] for every row in
  /// [0,batchSize) and col in [0,width), broadcasts a per-column bias
  /// across every row (the dense-layer convention).
  virtual void AddBiasBroadcast(float* buf, const float* bias, size_t batchSize,
                                size_t width) noexcept = 0;

  // Activation

  /// @brief Applies `type` to `buf` in place.
  virtual void Activation(ActivationType type, float* buf, size_t n) noexcept = 0;
  /// @brief Two-buffer variant of Activation(): reads src, writes into dst.
  virtual void ActivationInto(ActivationType type, float* dst, const float* src,
                              size_t n) noexcept = 0;
  /// @brief Applies `type`'s derivative to `buf` in place.
  /// @param type Which activation's derivative to apply.
  /// @param buf Array to derive in place.
  /// @param n Length of buf.
  /// @param activated If true, buf already holds the activated value;
  /// if false, buf holds the pre-activation value.
  virtual void ActivationDerivative(ActivationType type, float* buf, size_t n,
                                    bool activated) noexcept = 0;
  /// @brief Two-buffer variant of ActivationDerivative(): reads src
  /// (the pre-activation value), writes the derivative into dst.
  virtual void ActivationDerivativeInto(ActivationType type, float* dst, const float* src,
                                        size_t n) noexcept = 0;

  // Fused PC-specific ops

  /// @brief Same as FusedStateUpdate, but with momentum: the update
  /// direction (feedback*deriv - e) is EMA-smoothed into `v` (same
  /// shape as z, persistent across settling steps within one
  /// TrainStep(), reset to zero at the start of each call) before being
  /// applied to z. beta is the EMA decay (0.9 is a common default,
  /// higher = more smoothing/inertia).
  ///   v = beta*v + (1-beta)*((feedback*deriv) - e)
  ///   z += ir*v
  /// @param dType Which activation's derivative to apply to z in
  /// place, deriv is computed inline from z, element by element,
  /// rather than read from a separate precomputed buffer.
  virtual void FusedStateUpdateMomentum(float* z, float* v, const float* feedback,
                                        ActivationType dType, const float* e, size_t n, float ir,
                                        float beta) noexcept = 0;

  /// @brief One settling step's state update, fused into a single call:
  ///   z += ir * ((feedback * deriv) - e)
  /// Non-momentum counterpart to FusedStateUpdateMomentum() above.
  /// @param dType Which activation's derivative to apply to z in
  /// place, deriv is computed inline from z, element by element,
  /// rather than read from a separate precomputed buffer.
  virtual void FusedStateUpdate(float* z, const float* feedback, ActivationType dType, const float* e,
                                size_t n, float ir) noexcept = 0;
  /// @brief Computes e = z - mu, then returns 0.5 * sum(e^2), the
  /// Gaussian error/energy used by every layer except a cross-entropy
  /// terminal.
  virtual float ComputeErrorAndEnergy(float* e, const float* z, const float* mu,
                                      size_t n) noexcept = 0;
  /// @brief Same as ComputeErrorAndEnergy(), without computing (or
  /// returning) the energy, for callers that only need e.
  virtual void ComputeError(float* e, const float* z, const float* mu, size_t n) noexcept = 0;

  /// @brief Softmax cross-entropy variant of ComputeErrorAndEnergy, for
  /// the terminal layer only. Unlike the Gaussian version, this needs
  /// batchSize/nextSize separately (not just a flat n) since softmax is
  /// a row-wise, not element-wise, operation.
  ///   probs = softmax(mu), row-wise
  ///   e = z - probs         (z is the clamped target; matches the
  ///                           existing e := -dF/dmu convention exactly,
  ///                           since d(CE)/d(logits) = probs - y)
  ///   energy = -sum(z * log(probs + eps))  (eps=1e-8, avoids log(0))
  /// mu holds the RAW LOGITS (unchanged from the Gaussian case, still
  /// a*W\@phi(z)+b), softmax is applied here, not baked into mu itself.
  /// `rowEnergies` is caller-provided scratch, >= batchSize floats,
  /// NOT allocated internally (this runs inside CalculateState(), which
  /// runs inside TrainStep()'s CUDA-graph-captured region; dynamic
  /// allocation there is unsafe under capture, same reasoning as the
  /// CUTLASS workspace). CPUBackend's own override ignores this
  /// parameter entirely, kept in the shared interface so callers pass
  /// the same arguments to either backend.
  virtual float ComputeSoftmaxCrossEntropyErrorAndEnergy(float* e, const float* z, const float* mu,
                                                         size_t batchSize, size_t nextSize,
                                                         float* rowEnergies) noexcept = 0;

  /// @brief Same as ComputeSoftmaxCrossEntropyErrorAndEnergy(), without
  /// computing (or requiring `rowEnergies` for) the energy, for
  /// callers that only need e.
  virtual void ComputeSoftmaxCrossEntropyError(float* e, const float* z, const float* mu,
                                               size_t batchSize, size_t nextSize) noexcept = 0;

  /// @brief Attempts a fused forward pass (GEMM + bias + activation in
  /// one kernel, via CUTLASS on GPU) for RELU, LINEAR, GELU, TANH, or
  /// SIGMOID activation types; anything else returns false immediately.
  /// Returns true if the fused path was used (mu is fully computed,
  /// including bias and activation); false if the caller should fall
  /// back to the existing MatMul+AddBiasBroadcast+Activation sequence
  /// (always false on CPUBackend, CPU has no fused path, this is a
  /// GPU-only optimization).
  /// @warning As of this writing, every existing call site
  /// (FullPCLayer::ComputeMuOnly(), DirectKPPCLayer::ComputeMuOnly())
  /// passes actType=LINEAR unconditionally -- their real activation is
  /// applied via a separate ActivationInto() pass on the INPUT before
  /// this GEMM runs, not fused into its epilogue. The RELU/GELU/TANH/
  /// SIGMOID branches below are real and GPU-tested-correct in
  /// isolation, but exercising them for an actual layer requires moving
  /// that activation into this call instead (see CUDABackendGemm.cu's
  /// file comment before doing so: it's a data-flow change to a settling
  /// loop, not just a new branch).
  /// @param actType Activation to fuse in; RELU, LINEAR, GELU, TANH, and
  /// SIGMOID take the fused path, everything else returns false
  /// immediately.
  /// @param zF Activated input, shape [batchSize, size], row-major.
  /// @param W Weight matrix, shape [nextSize, size], row-major.
  /// @param bias Bias vector, shape [nextSize].
  /// @param mu Output, shape [batchSize, nextSize], row-major. Only
  /// written if this returns true.
  /// @param batchSize Number of rows in zF/mu.
  /// @param size Input width (columns of zF, columns of W).
  /// @param nextSize Output width (rows of W, columns of mu, length of bias).
  virtual bool TryFusedForwardPass(ActivationType actType, const float* zF, const float* W,
                                   const float* bias, float* mu, int batchSize, int size,
                                   int nextSize) noexcept = 0;

  // Convolution (im2col-based, ConvPCLayer family)

  /// @brief Rearranges a whole batch of (channels, height, width) input
  /// images into (channels*kH*kW, outH*outW) column matrices, the
  /// standard im2col transform, in one call. @p input is
  /// [batchSize, channels, height, width] and @p columns is
  /// [batchSize, channels*kH*kW, outH*outW], both contiguous and
  /// batch-major (the layout RepackForBatchedGemm() already expects).
  /// On CPUBackend this is a thin loop over Deep::Im2Col() (which
  /// genuinely only knows how to do one image at a time); on
  /// CUDABackend it's one kernel launch for the whole batch instead of
  /// @p batchSize separate launches -- the whole reason this takes a
  /// batch dimension at all rather than mirroring Deep::Im2Col()'s
  /// single-image contract. Positions outside the input (due to
  /// padding) are written as zero. @p columns is fully overwritten,
  /// not accumulated into.
  virtual void Im2Col(const float* input, int batchSize, int channels, int height, int width,
                      int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                      float* columns) noexcept = 0;

  /// @brief The adjoint of Im2Col(): scatters a whole batch of
  /// (channels*kH*kW, outH*outW) column-gradient buffers back into
  /// (channels, height, width) images, in one call. Same
  /// [batchSize, ...] batch-major layout and one-launch-per-batch
  /// rationale as Im2Col() above. ACCUMULATES into @p outputImage
  /// (does not zero it first), caller must zero the destination if a
  /// fresh result is wanted, matching Deep::Col2Im's own contract.
  virtual void Col2Im(const float* columns, int batchSize, int channels, int height, int width,
                      int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                      float* outputImage) noexcept = 0;

  /// @brief Batched 2D max pooling, no padding. @p input is [batchSize,
  /// channels, height, width], @p output and @p argmax are [batchSize,
  /// channels, outH, outW] where outH/outW = Deep::PoolOutDim(height/
  /// width, poolH/poolW, strideH/strideW). @p argmax records, per output
  /// position, the winning input position's flat (row*width+col) offset
  /// within its channel plane -- MaxPool2DBackward()'s only input besides
  /// the incoming gradient, no pre-activation buffer needed (a hard max's
  /// backward is exact from the argmax alone).
  virtual void MaxPool2DForward(const float* input, int batchSize, int channels, int height,
                                int width, int poolH, int poolW, int strideH, int strideW,
                                float* output, int* argmax) noexcept = 0;

  /// @brief The adjoint of MaxPool2DForward(): scatters a batch of
  /// [batchSize, channels, outH, outW] gradients back into [batchSize,
  /// channels, height, width] using the recorded @p argmax. ACCUMULATES
  /// into @p inputGrad (does not zero it first), matching
  /// Deep::MaxPool2DBackward's own contract.
  virtual void MaxPool2DBackward(const float* outputGrad, const int* argmax, int batchSize,
                                 int channels, int height, int width, int outH, int outW,
                                 float* inputGrad) noexcept = 0;

  /// @brief Repacks a [batchSize, rows, cols] tensor (batch-major)
  /// into [rows, batchSize, cols] (row-major, batch second),
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
  virtual void RepackForBatchedGemm(float* dst, const float* src, size_t batchSize, size_t rows,
                                    size_t cols) noexcept = 0;

  // Optimizer

  /// @brief Atomically increments *counter by 1. Device-resident so it
  /// can be read/updated by the same kernel/graph across settling steps
  /// (e.g. Adam's step count `t`) without a host round-trip.
  virtual void IncrementCounter(int* counter) noexcept = 0;

  /// @brief Adam update for `n` parameters in place.
  /// @param param Parameters to update.
  /// @param grad Gradient w.r.t. param.
  /// @param m Adam's first-moment (momentum) buffer, persistent across calls.
  /// @param v Adam's second-moment (variance) buffer, persistent across calls.
  /// @param n Number of parameters.
  /// @param t Device-resident step count (read, not written, here,
  /// see IncrementCounter()).
  /// @param lr Device-resident learning rate.
  /// @param beta1,beta2 Momentum/variance EMA decay rates.
  /// @param eps Denominator stabilizer.
  virtual void AdamStep(float* param, const float* grad, float* m, float* v, size_t n, const int* t,
                        const float* lr, float beta1 = 0.9f, float beta2 = 0.999f,
                        float eps = 1e-8f) noexcept = 0;
  /// @brief Same as AdamStep(), plus decoupled weight decay (AdamW):
  /// param -= lr * weightDecay * param, applied before the Adam step itself.
  virtual void AdamWStep(float* param, const float* grad, float* m, float* v, size_t n,
                         const int* t, const float* lr, float weightDecay, float beta1 = 0.9f,
                         float beta2 = 0.999f, float eps = 1e-8f) noexcept = 0;

  /// @brief Which device this backend instance runs on.
  virtual DeviceType GetDeviceType() const noexcept = 0;

  /// @brief dst[i] = a[i] * b[i] for all i in [0, n), elementwise product.
  virtual void MultiplyInto(float* dst, const float* a, const float* b, size_t n) noexcept = 0;
  /// @brief Fuses an in-place "derivative from activated value" step with
  /// the elementwise multiply that immediately follows it in every known
  /// caller: dst[i] = a[i] * f'(activatedInOut[i]), then
  /// activatedInOut[i] is overwritten with that same derivative,
  /// exactly matching (and replacing) the two separate calls
  /// ActivationDerivative(dType, activatedInOut, n, /*activated=*/true)
  /// followed by MultiplyInto(dst, a, activatedInOut, n). The overwrite
  /// is not a discardable side effect: at least one caller
  /// (SimpleConvPCLayer::UpdateWeights()) reads activatedInOut again
  /// afterward expecting it to hold the derivative from the last
  /// UpdateState() call, not the original activated value.
  /// @warning Same GELU caveat as ActivationDerivativeFromActivatedScalar
  /// (see Activations.h): GELU's derivative can't be recovered from its
  /// own output, so dGELU here reproduces the same wrong-for-GELU answer
  /// ActivationDerivative(..., activated=true) already silently gives
  /// today, rather than pretending to fix a deeper design issue.
  /// @param dType A derivative-flavored ActivationType.
  virtual void FusedActivationDerivativeMultiply(float* dst, const float* a, float* activatedInOut,
                                                 ActivationType dType, size_t n) noexcept = 0;
  /// @brief buf[i] = value for all i in [0, n).
  virtual void Fill(float* buf, size_t n, float value) noexcept = 0;
  /// @brief Convolutional bias-add: buf[c*spatialSize + s] += bias[c] for
  /// all c in [0,channels), s in [0,spatialSize). Per-CHANNEL broadcast
  /// across spatial positions, the transpose relationship to
  /// AddBiasBroadcast (which broadcasts a per-COLUMN bias across ROWS,
  /// the dense-layer convention). Batch-aware for the common case of one
  /// shared @p bias (fold batchSize into @p spatialSize, as ConvPCLayer's
  /// family already does for the ordinary per-conv-layer bias); NOT
  /// batch-aware when a call site instead needs a DIFFERENT bias per
  /// batch item (e.g. FullConvPCLayer::DirectFeedbackUpdate()'s
  /// per-image projChannel), which still has to call this once per item
  /// with @p buf offset to that item's slice.
  virtual void AddBiasPerChannel(float* buf, const float* bias, size_t channels,
                                 size_t spatialSize) noexcept = 0;

  // Precision-weighted PC ops (DiscriminativePCLayer/ConvPCLayer family,
  // a per-position "confidence" p broadcast across the batch dimension,
  // matching Gaussian PC theory's inverse-variance interpretation of p)

  /// @brief Precision-weighted counterpart to ComputeErrorAndEnergy():
  /// e[b*width+i] = z[b*width+i] - mu[b*width+i] for every b in
  /// [0,batchSize), i in [0,width); returns
  /// sum_{b,i}(0.5*p'[i]*e^2 - 0.5*log(p'[i])), where p'[i] =
  /// max(p[i], 1e-8) (the same floor DiscriminativePCLayer/ConvPCLayer's
  /// existing CPU code already applies, guarding against log(0)). `p` has
  /// length `width`, one precision per own-position, broadcast across
  /// every batch item, NOT length batchSize*width like e/z/mu.
  virtual float ComputePrecisionWeightedErrorAndEnergy(float* e, const float* z, const float* mu,
                                                       const float* p, size_t batchSize,
                                                       size_t width) noexcept = 0;
  /// @brief Same as ComputePrecisionWeightedErrorAndEnergy(), without
  /// computing (or returning) the energy, for callers that only need e.
  /// On CUDABackend this additionally means: no blocking
  /// cudaStreamSynchronize() -- the ...AndEnergy() variant must
  /// synchronously read the energy reduction back to a host float, which
  /// is illegal while a CUDA graph is being captured, exactly why this
  /// counterpart exists (mirrors ComputeError() alongside
  /// ComputeErrorAndEnergy() above).
  virtual void ComputePrecisionWeightedError(float* e, const float* z, const float* mu,
                                             const float* p, size_t batchSize,
                                             size_t width) noexcept = 0;

  /// @brief y[b*width+i] += alpha * x[b*width+i] * factor[i], for every b
  /// in [0,batchSize), i in [0,width), AxpyInto with an added
  /// per-position factor broadcast across the batch dimension. Covers
  /// dz_dt[idx] -= p[i]*e[idx] (alpha=-1, factor=p) without a temporary
  /// buffer for the elementwise product.
  virtual void AxpyBroadcastInto(float* y, const float* x, const float* factor, size_t batchSize,
                                 size_t width, float alpha) noexcept = 0;

  /// @brief dst[b*width+i] = a[b*width+i] * factor[i] * b[b*width+i], for
  /// every batch item, a three-operand elementwise product with one
  /// operand (factor) broadcast across the batch dimension. Covers
  /// bottom_up_cols[idx] = e_above[idx] * p_above[i] * mu[idx], the same
  /// shape needed in both UpdateState() and UpdateWeights() for every
  /// precision-weighted layer.
  virtual void MultiplyBroadcastInto(float* dst, const float* a, const float* factor, const float* b,
                                     size_t batchSize, size_t width) noexcept = 0;

  /// @brief Precision update from accumulated squared error, fused:
  /// for each i in [0,width), grad = mean_{b in [0,batchSize)} of
  /// 0.5*(p[i]*e[b*width+i]^2 - 1); log_p[i] -= pr*grad, clamped to
  /// [-5,5]; p[i] = exp(log_p[i]). Matches
  /// DiscriminativePCLayer::UpdatePrecision()/ConvPCLayer::UpdatePrecision()'s
  /// existing CPU formula exactly, including the clamp bounds.
  virtual void UpdatePrecisionFromError(float* p, float* log_p, const float* e, size_t batchSize,
                                        size_t width, float pr) noexcept = 0;
};
} // namespace Deep
