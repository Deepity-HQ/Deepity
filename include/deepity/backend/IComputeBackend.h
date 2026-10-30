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
///
/// @tparam T Element type for this backend's data buffers (activations,
/// weights, errors, biases, ...). `IComputeBackend` (see the alias
/// below) is `IComputeBackendT<float, float>`, bit-for-bit identical to
/// this interface before it was templated -- every existing caller
/// (SimplePCLayer, DirectKPPCLayer, GaussSeidelPCLayer, the conv
/// family) keeps using that exact spelling, unchanged.
/// @tparam AccumT Wider type used for optimizer accumulator state
/// (AdamStep/AdamWStep's grad/m/v/lr) and for scalar
/// hyperparameters/reductions (alpha, beta, eps, ir, Sum()'s return,
/// ...) -- "accumulate wider than you store/compute the forward pass
/// at," so a sub-8-bit T never gets Adam-stepped in its own
/// low-precision representation. Defaults to T for every existing
/// (non-templated) caller, where it's simply unused distinction
/// (AccumT=T=float there); FullPCLayer<T, AccumT=double> is the only
/// place this is ever spelled as something other than T.
template <typename T, typename AccumT = T>
class IComputeBackendT
{
public:
  virtual ~IComputeBackendT() = default;

  // Graphs

  /// @brief Starts recording subsequent backend calls into a replayable
  /// graph, instead of executing them immediately (CUDA graph capture;
  /// a no-op record on CPUBackend). Pair with EndGraphCapture().
  /// @param slot Which independent graph slot to (re)capture into. Each
  /// slot keeps its own graph/instantiated-executable, so a network
  /// that needs two DIFFERENT, independently-replayable operations
  /// (e.g. FullPCNetwork::TrainStep()'s forward+settle+update vs.
  /// Predict()'s forward+settle-only, called interleaved every step)
  /// captures each into its own slot instead of one evicting the
  /// other's graph and forcing a recapture on every single call.
  /// Defaults to 0, the only slot every other existing caller uses.
  virtual void BeginGraphCapture(size_t slot = 0) noexcept = 0;
  /// @brief Ends capture and instantiates the captured graph into
  /// @p slot (see BeginGraphCapture()'s own doc comment).
  /// @return true if capture and instantiation both succeeded and
  /// ReplayGraph() is now safe to call; false otherwise. Callers
  /// MUST check this, silently assuming success here was the
  /// cause of a real bug: a failed capture left ReplayGraph()
  /// permanently doing nothing on every subsequent call, since the
  /// caller had no way to know capture never actually happened.
  virtual bool EndGraphCapture(size_t slot = 0) noexcept = 0;
  /// @brief Launches the graph previously captured and instantiated
  /// into @p slot by BeginGraphCapture()/EndGraphCapture(). No-op if
  /// EndGraphCapture() never returned true for that slot.
  /// cudaGraphLaunch() is asynchronous: this returns as soon as the
  /// launch is ENQUEUED, not once it completes. Ordinarily that's fine
  /// (CUDA guarantees same-stream ordering, so a later call that needs
  /// the result just needs to be queued on the same stream too) -- but
  /// call Synchronize() afterward if you need the replayed work to have
  /// definitely finished, e.g. before reading results back on the host
  /// through any path that ISN'T itself a synchronous backend call.
  virtual void ReplayGraph(size_t slot = 0) noexcept = 0;
  /// @brief Blocks until all previously enqueued work on this backend's
  /// stream has completed. A no-op on CPUBackend (every CPU call is
  /// already synchronous).
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
  /// device buffer (a plain memcpy on CPUBackend).
  virtual void CopyFromHost(T* deviceDst, const T* hostSrc, size_t numElements) noexcept = 0;
  /// @brief Copies `numElements` elements of T from a device buffer into a
  /// host buffer (a plain memcpy on CPUBackend).
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
  virtual void MatMul(bool transA, bool transB, int M, int N, int K, AccumT alpha, const T* A,
                      int lda, const T* B, int ldb, AccumT beta, T* C, int ldc) noexcept = 0;

  /// @brief Same operation as MatMul(), but writes into an AccumT*
  /// buffer instead of T* -- for a layer's own Adam/AdamW gradient
  /// computation (grad_W, grad_Psi, ...), which must land in the wider
  /// AccumT accumulator, not get rounded down to storage precision T
  /// mid-computation (exactly the gap this method closes: MatMul()
  /// itself can only ever write a T* C, so it alone can't produce a
  /// loss-free gradient when T is a lossy storage format with
  /// AccumT != T, e.g. Fp4E2M1Value). For T == AccumT (every float/
  /// double instantiation) this is numerically identical to MatMul()
  /// with that same buffer, just spelled through the AccumT-typed
  /// overload the optimizer path calls either way.
  virtual void MatMulAccum(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                           const T* A, int lda, const T* B, int ldb, AccumT beta, AccumT* C,
                           int ldc) noexcept = 0;

  /// @brief Same operation as MatMulAccum(), but A is already AccumT* --
  /// for a layer's transient settling/gradient scratch (e.g. the
  /// activation-derivative-weighted error signal that feeds both
  /// UpdateState()'s feedback GEMM and UpdateWeights()'s gradient GEMM),
  /// which must never round-trip through storage precision T even
  /// transiently: for a lossy T (e.g. Fp4E2M1Value), an activation
  /// derivative's natural range commonly falls entirely below T's
  /// coarsest resolution (sigmoid's derivative maxes out at 0.25, below
  /// Fp4E2M1Value's smallest nonzero magnitude of 0.5), so writing it
  /// into a T* buffer even once rounds it to exactly zero and kills the
  /// gradient outright -- this is that fix. Only B (the real, stored
  /// parameter, e.g. W) decodes from T; A is used as-is.
  virtual void MatMulAccumMixed(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                                const AccumT* A, int lda, const T* B, int ldb, AccumT beta,
                                AccumT* C, int ldc) noexcept = 0;

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
  virtual void SumRows(T* dst, const T* src, size_t batchSize, size_t width) noexcept = 0;

  /// @brief Same operation as SumRows(), but writes into an AccumT*
  /// buffer instead of T* -- see MatMulAccum()'s own doc comment for
  /// why this exists (a layer's bias gradient, grad_b, needs the same
  /// AccumT-accumulating treatment grad_W does).
  virtual void SumRowsAccum(AccumT* dst, const T* src, size_t batchSize,
                            size_t width) noexcept = 0;

  /// @brief Returns the plain (signed) sum of every element in buf.
  /// Synchronous on CUDABackend (reads the reduction back to a host
  /// value before returning) -- same constraint as ComputeErrorAndEnergy
  /// and friends: never call this from inside a captured CUDA graph
  /// region, only from code that runs outside any
  /// BeginGraphCapture()/EndGraphCapture() pair.
  virtual AccumT Sum(const T* buf, size_t n) noexcept = 0;

  // Elementwise scalar ops

  /// @brief buf[i] *= alpha for all i in [0, n).
  virtual void Scale(T* buf, size_t n, AccumT alpha) noexcept = 0;
  /// @brief y[i] += alpha * x[i] for all i in [0, n), the standard AXPY.
  virtual void AxpyInto(T* y, const T* x, size_t n, AccumT alpha) noexcept = 0;
  /// @brief Same operation as AxpyInto(), but y is AccumT* and x stays
  /// T* -- see MatMulAccumMixed()'s own doc comment for why a settling/
  /// gradient scratch accumulator must never round-trip through T.
  virtual void AxpyIntoAccum(AccumT* y, const T* x, size_t n, AccumT alpha) noexcept = 0;
  /// @brief Same operation as AxpyInto(), but x is AccumT* -- for
  /// applying an AccumT-precision settling/gradient delta onto a real,
  /// stored T-typed buffer (z, W, b, ...), which is the one place this
  /// delta is *meant* to finally round-trip through storage precision.
  virtual void AxpyIntoFromAccum(T* y, const AccumT* x, size_t n, AccumT alpha) noexcept = 0;
  /// @brief buf[row*width + col] += bias[col] for every row in
  /// [0,batchSize) and col in [0,width), broadcasts a per-column bias
  /// across every row (the dense-layer convention).
  virtual void AddBiasBroadcast(T* buf, const T* bias, size_t batchSize,
                                size_t width) noexcept = 0;

  /// @brief Attempts C = A @ B(^T) + bias in one fused kernel (CPU only,
  /// via oneDNN's matmul-with-bias primitive, DEEPITY_USE_ONEDNN build
  /// only) -- avoids AddBiasBroadcast's separate, full memory pass over
  /// C after MatMul(). Returns true if the fused path ran (C is fully
  /// computed); false if the caller should fall back to the existing
  /// MatMul()+AddBiasBroadcast() sequence (always false when oneDNN
  /// isn't compiled in, and on CUDABackend -- that backend's own fused
  /// path is TryFusedForwardPass, which also fuses the activation).
  /// @note Deliberately separate from TryFusedForwardPass: this one
  /// stops at the bias-add, so callers that need the pre-activation
  /// value (e.g. FullPCLayer's preact, for GELU's exact derivative)
  /// keep getting it from a normal Activation()/ActivationInto() call
  /// afterward, unlike TryFusedForwardPass's fused activation which
  /// would discard it. (SimplePCLayer itself has no such need under
  /// its own mu=W\@phi(z)+b convention -- the activation it needs lives
  /// on the GEMM's INPUT, applied before this call, not on its output
  /// -- but this method stays activation-free regardless, for any
  /// caller that does need the snapshot.)
  /// @param transA/transB Same meaning as MatMul(). @param M/N/K/A/lda/
  /// B/ldb/C/ldc Same meaning and shapes as MatMul() (alpha=1, beta=0
  /// implied -- this is specifically the forward-layer GEMM+bias
  /// pattern, not a general GEMM).
  /// @param bias Length-N vector, broadcast across all M rows of C
  /// (same convention as AddBiasBroadcast).
  virtual bool TryFusedMatMulBias(bool transA, bool transB, int M, int N, int K, const T* A,
                                  int lda, const T* B, int ldb, const T* bias, T* C,
                                  int ldc) noexcept = 0;

  /// @brief Attempts C = alpha * (A @ B(^T)) via oneDNN's matmul
  /// primitive instead of MatMul()'s cblas_sgemm (CPU only,
  /// DEEPITY_USE_ONEDNN build only) -- no bias, beta=0 implied (this is
  /// NOT a general GEMM: it cannot accumulate into an existing C, unlike
  /// MatMul()). Exists to test whether oneDNN's own kernel simply beats
  /// cblas_sgemm's dispatch overhead at the small per-settling-step
  /// shapes PC layers use, independent of any bias-fusion benefit.
  /// Returns true if the fused path ran; false if the caller should
  /// fall back to MatMul() with the same alpha and beta=0 (always false
  /// when oneDNN isn't compiled in, and on CUDABackend).
  /// @param transA/transB/M/N/K/A/lda/B/ldb/C/ldc Same meaning and
  /// shapes as MatMul(). @param alpha Scalar multiplier on the product,
  /// same convention as MatMul().
  virtual bool TryFusedMatMul(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                              const T* A, int lda, const T* B, int ldb, T* C,
                              int ldc) noexcept = 0;

  /// @brief Fuses SimplePCLayer::UpdateState()'s feedback GEMM directly
  /// into its own elementwise state update -- the GEMM's output is never
  /// separately read back and re-walked by its own pass:
  ///   feedback = e_above @ W                   (M=batchSize,N=size,K=nextSize)
  ///   z += ir * (f'(z) * feedback - e)
  /// Primary mechanism (CPUBackend, DEEPITY_USE_ONEDNN): a dnnl::matmul
  /// primitive with the elementwise update welded on as post-ops
  /// (binary_mul/binary_add), evaluated in registers on each output tile
  /// right as oneDNN's own GEMM microkernel produces it -- same kernel,
  /// same thread pool as every other matmul in this file, just with the
  /// epilogue attached. Works for every ActivationType oneDNN can be
  /// handed a precomputed derivative for (i.e. all of them, via
  /// ActivationDerivativeInto()), unlike the Halide fallback below.
  /// Secondary fallback (DEEPITY_USE_HALIDE only, off by default): an
  /// ahead-of-time Halide kernel (src/backend/halide/
  /// FeedbackUpdateGenerator.cpp) that was this project's first attempt at
  /// this fusion -- kept only because it's verified correct, not because
  /// it's fast: measured slower than oneDNN's separate GEMM+elementwise
  /// pair even after a properly-tiled schedule, because its own thread
  /// pool oversubscribes CPU cores alongside oneDNN's (see
  /// DEEPITY_USE_HALIDE's CMakeLists.txt comment for the full writeup).
  /// Only ActivationType::SIGMOID and nextSize>=64 reach it. Any dType
  /// neither path handles, or every path failing/disabled, returns false
  /// so the caller falls back to its own separate TryFusedMatMul()/
  /// FusedStateUpdate() calls, same convention as TryFusedMatMul() above.
  /// @param dType Which activation's derivative to apply to z, matching
  /// FusedStateUpdate()'s own dType parameter.
  /// @param batchSize/size/nextSize Shapes as named in SimplePCLayer.
  /// @param eAbove (batchSize, nextSize) row-major, tightly packed.
  /// @param W (nextSize, size) row-major, tightly packed.
  /// @param z In/out, (batchSize, size) row-major, tightly packed. Must
  /// NOT alias scratch/derivScratch (each is both read and written
  /// across the two mechanisms above; aliasing silently produces wrong
  /// results -- verified empirically with the Halide kernel, see
  /// FeedbackUpdateGenerator.cpp's file comment).
  /// @param e (batchSize, size) row-major, tightly packed.
  /// @param ir Inference rate, same meaning as FusedStateUpdate()'s ir.
  /// @param scratch Caller-owned (batchSize, size) scratch buffer (e.g.
  /// SimplePCLayer's own feedbackScratch, otherwise unused on this
  /// path) -- the fused result is written here, then copied into z, to
  /// satisfy the no-aliasing requirement above without a per-call heap
  /// allocation.
  /// @param derivScratch Caller-owned (batchSize, size) scratch buffer
  /// holding f'(z), precomputed once by the oneDNN path (the only
  /// elementwise precompute pass this path needs -- e and z feed the
  /// post-op chain directly, see CPUBackend.cpp's implementation) and
  /// bound as a post-op's runtime operand. Unused on the Halide path.
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
  /// into an AccumT* buffer instead of T* -- see MatMulAccumMixed()'s
  /// own doc comment: an activation derivative's natural range commonly
  /// undershoots a lossy T's coarsest resolution entirely, so this must
  /// land in AccumT precision at the point of computation, never T, even
  /// transiently.
  virtual void ActivationDerivativeIntoAccum(ActivationType type, AccumT* dst, const T* src,
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
  virtual void FusedStateUpdateMomentum(T* z, T* v, const T* feedback,
                                        ActivationType dType, const T* e, size_t n, AccumT ir,
                                        AccumT beta) noexcept = 0;

  /// @brief One settling step's state update, fused into a single call:
  ///   z += ir * ((feedback * deriv) - e)
  /// Non-momentum counterpart to FusedStateUpdateMomentum() above.
  /// @param dType Which activation's derivative to apply to z in
  /// place, deriv is computed inline from z, element by element,
  /// rather than read from a separate precomputed buffer.
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

  /// @brief Fuses ComputeError()'s e = z - mu with the forward
  /// activation zF = f(z) that SimplePCLayer::ComputeMuOnly() computes
  /// right afterward in the common (non-cross-entropy, no-energy-needed)
  /// settling step -- both read the same z, so this halves that read and
  /// pays one OpenMP fork/join instead of two, see
  /// SimplePCLayer::CalculateState()'s call site for the exact condition
  /// under which this applies instead of the separate ComputeError()/
  /// ActivationInto() pair. Unlike TryFusedMatMul() et al., this isn't a
  /// GEMM-adjacent fusion -- oneDNN's post-ops only transform a
  /// primitive's *output*, and the activation here runs strictly before
  /// the mu GEMM reads zF, so there's no post-op slot to weld it onto;
  /// this is a plain elementwise loop fusion instead.
  /// @param activationType A forward-flavored ActivationType (e.g.
  /// SIGMOID), matching ComputeMuOnly()'s own activationType.
  /// @param e Output, length n: z - mu.
  /// @param zF Output, length n: f(z).
  /// @param z Input, length n.
  /// @param mu Input, length n.
  /// @param n Element count (batchSize * size).
  /// @return false if this backend has no fused implementation, in
  /// which case the caller falls back to separate ComputeError()/
  /// ActivationInto() calls, same convention as TryFusedMatMul() above.
  virtual bool TryComputeErrorAndActivation(ActivationType activationType, T* e, T* zF,
                                            const T* z, const T* mu,
                                            size_t n) noexcept = 0;

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
  /// `rowEnergies` is caller-provided scratch, >= batchSize elements,
  /// NOT allocated internally (this runs inside CalculateState(), which
  /// runs inside TrainStep()'s CUDA-graph-captured region; dynamic
  /// allocation there is unsafe under capture, same reasoning as the
  /// CUTLASS workspace). CPUBackend's own override ignores this
  /// parameter entirely, kept in the shared interface so callers pass
  /// the same arguments to either backend.
  virtual AccumT ComputeSoftmaxCrossEntropyErrorAndEnergy(T* e, const T* z, const T* mu,
                                                          size_t batchSize, size_t nextSize,
                                                          AccumT* rowEnergies) noexcept = 0;

  /// @brief Same as ComputeSoftmaxCrossEntropyErrorAndEnergy(), without
  /// computing (or requiring `rowEnergies` for) the energy, for
  /// callers that only need e.
  virtual void ComputeSoftmaxCrossEntropyError(T* e, const T* z, const T* mu,
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
  virtual bool TryFusedForwardPass(ActivationType actType, const T* zF, const T* W,
                                   const T* bias, T* mu, int batchSize, int size,
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
  virtual void Im2Col(const T* input, int batchSize, int channels, int height, int width,
                      int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                      T* columns) noexcept = 0;

  /// @brief The adjoint of Im2Col(): scatters a whole batch of
  /// (channels*kH*kW, outH*outW) column-gradient buffers back into
  /// (channels, height, width) images, in one call. Same
  /// [batchSize, ...] batch-major layout and one-launch-per-batch
  /// rationale as Im2Col() above. ACCUMULATES into @p outputImage
  /// (does not zero it first), caller must zero the destination if a
  /// fresh result is wanted, matching Deep::Col2Im's own contract.
  virtual void Col2Im(const T* columns, int batchSize, int channels, int height, int width,
                      int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                      T* outputImage) noexcept = 0;

  /// @brief Batched 2D max pooling, no padding. @p input is [batchSize,
  /// channels, height, width], @p output and @p argmax are [batchSize,
  /// channels, outH, outW] where outH/outW = Deep::PoolOutDim(height/
  /// width, poolH/poolW, strideH/strideW). @p argmax records, per output
  /// position, the winning input position's flat (row*width+col) offset
  /// within its channel plane -- MaxPool2DBackward()'s only input besides
  /// the incoming gradient, no pre-activation buffer needed (a hard max's
  /// backward is exact from the argmax alone).
  virtual void MaxPool2DForward(const T* input, int batchSize, int channels, int height,
                                int width, int poolH, int poolW, int strideH, int strideW,
                                T* output, int* argmax) noexcept = 0;

  /// @brief The adjoint of MaxPool2DForward(): scatters a batch of
  /// [batchSize, channels, outH, outW] gradients back into [batchSize,
  /// channels, height, width] using the recorded @p argmax. ACCUMULATES
  /// into @p inputGrad (does not zero it first), matching
  /// Deep::MaxPool2DBackward's own contract.
  virtual void MaxPool2DBackward(const T* outputGrad, const int* argmax, int batchSize,
                                 int channels, int height, int width, int outH, int outW,
                                 T* inputGrad) noexcept = 0;

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
  virtual void RepackForBatchedGemm(T* dst, const T* src, size_t batchSize, size_t rows,
                                    size_t cols) noexcept = 0;

  /// @brief Attempts a single fused convolution forward pass, skipping
  /// Im2Col()/RepackForBatchedGemm()/MatMul() entirely: @p output =
  /// @p alpha * conv(@p input, @p W), in the same [batchSize, outChannels,
  /// outH, outW] NCHW layout the Im2Col()+RepackForBatchedGemm()+MatMul()
  /// fallback's own un-repack produces. Activation() is layout-agnostic
  /// (elementwise), but the bias-add is NOT: use AddBiasPerChannelNCHW(),
  /// not AddBiasPerChannel() (which needs channel as the OUTERMOST
  /// dimension, wrong for this layout whenever batchSize > 1 -- see its
  /// own doc comment). Returns false (without touching @p output) when this
  /// backend has no fused path -- always on CPUBackend, or CUDABackend
  /// built without cuDNN -- letting the caller fall back to the manual
  /// Im2Col()+RepackForBatchedGemm()+MatMul() sequence it already has.
  /// @param W [outChannels, inChannels, kernelH, kernelW], the exact
  /// same memory every ConvPCLayer-family layer already stores its flat
  /// [outChannels, inChannels*kernelH*kernelW] weights in.
  virtual bool TryFusedConvForward(const T* input, const T* W, T* output, AccumT alpha,
                                   int batchSize, int inChannels, int height, int width,
                                   int outChannels, int kernelH, int kernelW, int strideH,
                                   int strideW, int padH, int padW) noexcept = 0;

  /// @brief The adjoint of TryFusedConvForward() w.r.t. @p input, skipping
  /// the transposed MatMul()+RepackForBatchedGemm()+Col2Im() sequence.
  /// OVERWRITES @p inputGrad = @p alpha * convBackwardData(@p outputGrad,
  /// @p W) -- unlike Col2Im()'s own accumulate contract, every known
  /// caller here Zero()s its destination immediately beforehand anyway.
  /// Same false-means-unsupported contract as TryFusedConvForward().
  virtual bool TryFusedConvBackwardData(const T* outputGrad, const T* W, T* inputGrad,
                                        AccumT alpha, int batchSize, int inChannels, int height,
                                        int width, int outChannels, int kernelH, int kernelW,
                                        int strideH, int strideW, int padH, int padW) noexcept = 0;

  /// @brief The adjoint of TryFusedConvForward() w.r.t. @p W, skipping
  /// Im2Col()+RepackForBatchedGemm()+MatMul(). @p dW = @p alpha *
  /// convBackwardFilter(@p input, @p outputGrad) + @p beta * @p dW,
  /// matching MatMul()'s own alpha/beta convention -- a caller that
  /// fuses its weight UPDATE directly into this GEMM (SGD: beta=1 into
  /// W itself) and one that wants a fresh gradient buffer (Adam: beta=0
  /// into grad_W) both work unchanged. Same false-means-unsupported
  /// contract as TryFusedConvForward().
  virtual bool TryFusedConvBackwardFilter(const T* input, const T* outputGrad, T* dW,
                                          AccumT alpha, AccumT beta, int batchSize, int inChannels,
                                          int height, int width, int outChannels, int kernelH,
                                          int kernelW, int strideH, int strideW, int padH,
                                          int padW) noexcept = 0;

  /// @brief Pre-grows CUDABackend's shared cuDNN conv workspace (see
  /// TryFusedConvForward() and friends) to fit this shape's algorithm
  /// workspace requirement. MUST be called for every distinct conv
  /// shape BEFORE Compile()'s first BeginGraphCapture() -- same timing
  /// constraint, and same reason, as PrepareForBatchSize(): growing the
  /// workspace means cudaMalloc()/cudaFree(), which are illegal (not
  /// stream-ordered) inside an active graph capture and poison the
  /// whole captured region if TryFusedConvForward() et al. are left to
  /// grow it lazily on first use instead. No-op on CPUBackend, which
  /// has no such workspace.
  virtual void PrepareConvWorkspace(int batchSize, int inChannels, int height, int width,
                                    int outChannels, int kernelH, int kernelW, int strideH,
                                    int strideW, int padH, int padW) noexcept = 0;

  // Optimizer

  /// @brief Atomically increments *counter by 1. Device-resident so it
  /// can be read/updated by the same kernel/graph across settling steps
  /// (e.g. Adam's step count `t`) without a host round-trip.
  virtual void IncrementCounter(int* counter) noexcept = 0;

  /// @brief Adam update for `n` parameters in place. `param` stays in
  /// this backend's storage type T; the optimizer's own accumulator
  /// state (grad/m/v/lr) is always AccumT, regardless of T -- a
  /// sub-8-bit T's weights are never Adam-stepped directly in their own
  /// low-precision representation, only through this wider shadow
  /// accumulator (see FullPCLayer<T, AccumT>'s own doc comment).
  /// @param param Parameters to update, in storage type T.
  /// @param grad Gradient w.r.t. param, in accumulator type AccumT.
  /// @param m Adam's first-moment (momentum) buffer, persistent across calls.
  /// @param v Adam's second-moment (variance) buffer, persistent across calls.
  /// @param n Number of parameters.
  /// @param t Device-resident step count (read, not written, here,
  /// see IncrementCounter()).
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

  /// @brief dst[i] = a[i] * b[i] for all i in [0, n), elementwise product.
  virtual void MultiplyInto(T* dst, const T* a, const T* b, size_t n) noexcept = 0;
  /// @brief Same operation as MultiplyInto(), but dst and b are AccumT*
  /// (a stays T*) -- see MatMulAccumMixed()'s own doc comment: b is
  /// typically an activation derivative already computed in AccumT
  /// precision via ActivationDerivativeIntoAccum(), and the product with
  /// a (e.g. the layer-above error, still T*) must stay in that
  /// precision too rather than round-trip through storage type T.
  virtual void MultiplyIntoAccum(AccumT* dst, const T* a, const AccumT* b, size_t n) noexcept = 0;
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
  virtual void FusedActivationDerivativeMultiply(T* dst, const T* a, T* activatedInOut,
                                                 ActivationType dType, size_t n) noexcept = 0;
  /// @brief Sibling of FusedActivationDerivativeMultiply() above, for
  /// callers that hold the PRE-activation value instead of the
  /// activated one: dst[i] = a[i] * f'(preact[i]). preact is read-only
  /// here (never overwritten) -- there is no reuse-the-overwrite caller
  /// for this variant.
  /// @warning Deliberately NOT a drop-in replacement for
  /// FusedActivationDerivativeMultiply(): that sibling is wrong for
  /// GELU (no derivative closed form from the activated value alone,
  /// see its own GELU caveat above). This one is exact for every
  /// activation type, including GELU, because it reads preact
  /// directly -- the same correctness distinction
  /// FullConvPCLayer/SimpleConvPCLayer's UpdateWeights() already had to
  /// un-fuse their own call sites over.
  /// @param dType A derivative-flavored ActivationType.
  virtual void FusedActivationDerivativeMultiplyFromPreact(T* dst, const T* a,
                                                            const T* preact, ActivationType dType,
                                                            size_t n) noexcept = 0;
  /// @brief buf[i] = value for all i in [0, n).
  virtual void Fill(T* buf, size_t n, AccumT value) noexcept = 0;
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
  virtual void AddBiasPerChannel(T* buf, const T* bias, size_t channels,
                                 size_t spatialSize) noexcept = 0;

  /// @brief Convolutional bias-add for the batch-MAJOR NCHW layout:
  /// buf[b*channels*spatialSize + c*spatialSize + s] += bias[c] for all
  /// b in [0,batchSize), c in [0,channels), s in [0,spatialSize). NOT
  /// the same contract as AddBiasPerChannel() above (which needs
  /// channel as the OUTERMOST dimension, i.e. [channels, batchSize*
  /// spatialSize] -- correct for lgRepacked's layout, wrong for mu's).
  /// This is what TryFusedConvForward()'s own [batchSize, outChannels,
  /// outH, outW] output actually needs: that layout keeps batch
  /// outermost, so AddBiasPerChannel() with spatialSize folded to
  /// batchSize*trueSpatialSize silently adds bias[c] to the wrong
  /// elements whenever batchSize > 1 (confirmed: every gradient-check
  /// test in this codebase uses batchSize=1, which collapses the two
  /// layouts into the same thing and could never have caught this).
  virtual void AddBiasPerChannelNCHW(T* buf, const T* bias, size_t batchSize,
                                     size_t channels, size_t spatialSize) noexcept = 0;

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
  virtual AccumT ComputePrecisionWeightedErrorAndEnergy(T* e, const T* z, const T* mu,
                                                        const T* p, size_t batchSize,
                                                        size_t width) noexcept = 0;
  /// @brief Same as ComputePrecisionWeightedErrorAndEnergy(), without
  /// computing (or returning) the energy, for callers that only need e.
  /// On CUDABackend this additionally means: no blocking
  /// cudaStreamSynchronize() -- the ...AndEnergy() variant must
  /// synchronously read the energy reduction back to a host value, which
  /// is illegal while a CUDA graph is being captured, exactly why this
  /// counterpart exists (mirrors ComputeError() alongside
  /// ComputeErrorAndEnergy() above).
  virtual void ComputePrecisionWeightedError(T* e, const T* z, const T* mu,
                                             const T* p, size_t batchSize,
                                             size_t width) noexcept = 0;

  /// @brief y[b*width+i] += alpha * x[b*width+i] * factor[i], for every b
  /// in [0,batchSize), i in [0,width), AxpyInto with an added
  /// per-position factor broadcast across the batch dimension. Covers
  /// dz_dt[idx] -= p[i]*e[idx] (alpha=-1, factor=p) without a temporary
  /// buffer for the elementwise product.
  virtual void AxpyBroadcastInto(T* y, const T* x, const T* factor, size_t batchSize,
                                 size_t width, AccumT alpha) noexcept = 0;

  /// @brief dst[b*width+i] = a[b*width+i] * factor[i] * b[b*width+i], for
  /// every batch item, a three-operand elementwise product with one
  /// operand (factor) broadcast across the batch dimension. Covers
  /// bottom_up_cols[idx] = e_above[idx] * p_above[i] * mu[idx], the same
  /// shape needed in both UpdateState() and UpdateWeights() for every
  /// precision-weighted layer.
  virtual void MultiplyBroadcastInto(T* dst, const T* a, const T* factor,
                                     const T* b, size_t batchSize, size_t width) noexcept = 0;

  /// @brief Precision update from accumulated squared error, fused:
  /// for each i in [0,width), grad = mean_{b in [0,batchSize)} of
  /// 0.5*(p[i]*e[b*width+i]^2 - 1); log_p[i] -= pr*grad, clamped to
  /// [-5,5]; p[i] = exp(log_p[i]). Matches
  /// DiscriminativePCLayer::UpdatePrecision()/ConvPCLayer::UpdatePrecision()'s
  /// existing CPU formula exactly, including the clamp bounds.
  virtual void UpdatePrecisionFromError(T* p, T* log_p, const T* e, size_t batchSize,
                                        size_t width, AccumT pr) noexcept = 0;
};

/// @brief Non-templated alias preserving every existing caller's
/// behavior exactly: `IComputeBackend` is `IComputeBackendT<float,
/// float>`, bit-for-bit identical to this interface before it was
/// templated. CPUBackend/CUDABackend still implement this exact
/// spelling; new templated code (FullPCLayer<T>/FullPCNetwork<T>) uses
/// IComputeBackendT<T, AccumT> directly, implemented by the new
/// CPUBackendT<T, AccumT>/CUDABackendT<T, AccumT>.
using IComputeBackend = IComputeBackendT<float, float>;

} // namespace Deep
