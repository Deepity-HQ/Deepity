/**
 * @file CUDABackendGemm.cu
 * @brief CUDABackend's matrix-multiply paths: the plain cuBLAS MatMul/
 * SumRows pair, and the CUTLASS-fused GEMM+bias+activation path used by
 * TryFusedForwardPass. Mirrors the "GEMM" section of IComputeBackend.h (plus
 * the GEMM-flavored half of "Fused PC-specific ops"). Split out of the
 * former monolithic CUDABackend.cu, see CUDABackendCore.cu,
 * CUDABackendElementwise.cu, CUDABackendActivations.cu,
 * CUDABackendFusedOps.cu, CUDABackendOptimizer.cu, CUDABackendConv.cu for
 * the rest.
 *
 * @note TryFusedForwardPass() dispatches RELU/LINEAR/GELU/TANH/SIGMOID
 * to a fused CUTLASS GEMM, but every current caller passes LINEAR
 * unconditionally and applies its real activation as a separate,
 * unfused pass on the GEMM's INPUT beforehand (see
 * IComputeBackend::TryFusedForwardPass's @warning). Actually fusing a
 * layer's activation into this GEMM means restructuring that caller to
 * feed this GEMM the pre-activation belief instead and let the epilogue
 * activate the OUTPUT -- a settling-loop data-flow change, not a new
 * branch here, and unverified end-to-end since this file has no way to
 * be exercised by a caller yet.
 */
#include <deepity/backend/CUDABackend.h>

#ifdef DEEPITY_USE_CUDA
#include <cstring>
#include <iostream>

#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
// Must be defined before activation.h (pulled in by
// linear_combination_sigmoid.h below) is first included anywhere in this
// translation unit: it selects Sigmoid<T>'s tanh-identity formula
// (sigmoid(x) = 0.5*tanh(x/2)+0.5) over the default 1/(1+exp(-x)). This
// engine's own CUDABackendActivations.cu sigmoidKernelInto() uses the
// same tanh-identity (and the same tanh.approx.f32 PTX instruction CUTLASS's
// fast_tanh() uses on sm_75+), so without this, GemmFwdSigmoid below would
// silently compute a numerically different (though mathematically
// equivalent) result than the unfused activation path it's meant to
// replace -- exactly the kind of fused/unfused divergence this project's
// predictive-coding settling loop can't tolerate.
#ifndef CUTLASS_USE_TANH_FOR_SIGMOID
#define CUTLASS_USE_TANH_FOR_SIGMOID
#endif
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm/device/gemm_grouped.h>
#include <cutlass/gemm/kernel/default_gemm_grouped.h>
#include <cutlass/gemm/kernel/gemm_grouped.h>

namespace Deep
{
// GELU_taylor, not CUTLASS's erf-exact GELU: this engine's own GELU
// (Activations.h / CUDABackendActivations.cu's GeluKernelInto) is the
// tanh approximation 0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x^2))),
// bit-for-bit CUTLASS's GELU_taylor (same constants, same fast_tanh()).
// Using plain GELU here would fuse a DIFFERENT activation than the one
// every unfused call site actually computes.
using LinearCombinationGELUTaylor = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::GELU_taylor, float, 128 / cutlass::sizeof_bits<float>::value, float,
    float, cutlass::epilogue::thread::ScaleType::Default,
    cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;
using LinearCombinationGELUTaylorScalar = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::GELU_taylor, float, 1, float, float,
    cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;

// cutlass::epilogue::thread::Tanh<float> calls fast_tanh(), the same
// tanh.approx.f32 PTX instruction (sm_75+) this engine's own tanhKernelInto()
// uses -- no dedicated linear_combination_tanh.h ships in CUTLASS, so this
// goes through the generic epilogue directly instead of a ready-made alias.
using LinearCombinationTanh = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::Tanh, float, 128 / cutlass::sizeof_bits<float>::value, float, float,
    cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;
using LinearCombinationTanhScalar = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::Tanh, float, 1, float, float,
    cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;

using GemmFwdRelu = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationRelu<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdLinear = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 128 / cutlass::sizeof_bits<float>::value,
                                                 float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdReluScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdLinearScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdGELU = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationGELUTaylor,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdGELUScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationGELUTaylorScalar,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdTanh = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationTanh,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdTanhScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationTanhScalar,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

// CUTLASS's own ready-made alias: with CUTLASS_USE_TANH_FOR_SIGMOID
// defined above, this is Sigmoid<T>'s tanh-identity branch, matching
// CUDABackendActivations.cu's sigmoidKernelInto() exactly (see the
// #define's comment up top).
using GemmFwdSigmoid = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationSigmoid<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdSigmoidScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationSigmoid<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

// --- Grouped GEMM: MatMulBatched()'s CUTLASS fast path -----------------
//
// cublasSgemmBatched() (below) decomposes into roughly one GPU kernel
// instance per batch item -- confirmed via nsys (depth-scaling benchmark's
// "Total GPU Ops" scales ~linearly with depth while per-op time stays
// flat, ~20-23us regardless of depth). So its CPU-side dispatch is O(1)
// (one cudaGraphLaunch() per step once graph-captured) but GPU-side
// EXECUTION still pays one kernel-launch's worth of fixed overhead per
// layer, which is why wall-clock time still trends toward O(depth) once
// depth is large enough for that per-item overhead to dominate.
// CUTLASS's Grouped GEMM genuinely fuses the whole batch into one kernel
// launch (the batch/group dimension lives inside the kernel's own grid),
// so GPU-side execution, not just CPU-side dispatch, becomes O(1) in the
// item count.
//
// Same (RowMajor, ColumnMajor)-means-(not transposed, transposed) mapping
// as GemmFwdLinear/GemmFwdRelu/etc. above: a [r,c] row-major buffer and a
// [c,r] column-major buffer over the same bytes/stride are identical, so
// "transpose this operand" is exactly "flip its Layout tag", independent
// of the other operand. Only the three (transA,transB) combinations
// DirectKPPCNetwork.cpp's batched calls actually use are instantiated;
// (true,true) isn't needed today and falls back to the small-dim path
// below rather than add a fourth untested instantiation.
template <typename LayoutA, typename LayoutB>
using GemmGroupedKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
    float, LayoutA, cutlass::ComplexTransform::kNone, 1, float, LayoutB,
    cutlass::ComplexTransform::kNone, 1, float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, cutlass::gemm::GemmShape<128, 128, 16>,
    cutlass::gemm::GemmShape<64, 64, 16>, cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 128 / cutlass::sizeof_bits<float>::value,
                                                  float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4,
    cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;

template <typename LayoutA, typename LayoutB>
using GemmGroupedOp = cutlass::gemm::device::GemmGrouped<GemmGroupedKernel<LayoutA, LayoutB>>;

// A not transposed, B transposed -- the only combo DirectKPPCNetwork.cpp
// currently uses twice (lines 111/175 as of this writing).
using GemmGroupedNT = GemmGroupedOp<cutlass::layout::RowMajor, cutlass::layout::ColumnMajor>;
// A transposed, B not.
using GemmGroupedTN = GemmGroupedOp<cutlass::layout::ColumnMajor, cutlass::layout::RowMajor>;
// Neither transposed.
using GemmGroupedNN = GemmGroupedOp<cutlass::layout::RowMajor, cutlass::layout::RowMajor>;

static_assert(sizeof(cutlass::gemm::GemmCoord) == 3 * sizeof(int),
              "cutlass::gemm::GemmCoord's layout changed -- "
              "CUDABackend::GrowGroupedGemmMetadata's int* reinterpret_cast to "
              "GemmCoord* (chosen so CUDABackend.h doesn't need to include CUTLASS) "
              "is no longer safe; store a real GemmCoord* there instead.");

namespace
{
// Fills the per-item problem-size/leading-dimension arrays GemmGrouped
// needs. Every item shares the same (M,N,K,lda,ldb,ldc) in this engine's
// usage (MatMulBatched's own contract -- a uniform-width batched run), so
// this is only ever called once per distinct shape, not once per
// TrainStep(); see GrowGroupedGemmMetadata's own caching.
__global__ void FillGroupedGemmMetadataKernel(int* problemSizes, int64_t* lda, int64_t* ldb,
                                               int64_t* ldc, int M, int N, int K, int64_t ldaVal,
                                               int64_t ldbVal, int64_t ldcVal, size_t batchCount)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= batchCount)
    return;
  problemSizes[i * 3 + 0] = M;
  problemSizes[i * 3 + 1] = N;
  problemSizes[i * 3 + 2] = K;
  lda[i] = ldaVal;
  ldb[i] = ldbVal;
  ldc[i] = ldcVal;
}
} // namespace

bool CUDABackend::GrowGroupedGemmMetadata(size_t batchCount, int M, int N, int K, int64_t lda,
                                           int64_t ldb, int64_t ldc) noexcept
{
  if (groupedMetaCapacity >= batchCount && cachedGroupedM == M && cachedGroupedN == N &&
      cachedGroupedK == K && cachedGroupedLda == lda && cachedGroupedLdb == ldb &&
      cachedGroupedLdc == ldc && cachedGroupedBatchCount == batchCount)
    return true; // Shape unchanged since the last call -- nothing to refill.

  if (groupedMetaCapacity < batchCount)
  {
    // Plain cudaMalloc/cudaFree, not the stream-ordered allocator: a
    // cudaMallocAsync captured into a graph here produced a graph that
    // itself replayed fine structurally but failed cudaGraphLaunch with
    // "invalid argument" on an A100 (confirmed on real hardware,
    // mechanism not fully understood) -- not worth the residual risk.
    // This growth is instead guaranteed to never run during capture at
    // all: CUDABackend::PrewarmGroupedGemmScratch() (called from
    // DirectKPPCNetwork::Compile(), mirroring PrewarmBatchedGemmScratch()
    // immediately above it) forces it to happen for every shape this
    // network will ever use, before that network's first captured
    // TrainStep()/Predict(). See that function's own comment.
    if (groupedProblemSizes)
      cudaFree(groupedProblemSizes);
    if (groupedLda)
      cudaFree(groupedLda);
    if (groupedLdb)
      cudaFree(groupedLdb);
    if (groupedLdc)
      cudaFree(groupedLdc);
    groupedProblemSizes = nullptr;
    groupedLda = groupedLdb = groupedLdc = nullptr;
    groupedMetaCapacity = 0;

    bool ok = cudaMalloc(&groupedProblemSizes, batchCount * 3 * sizeof(int)) == cudaSuccess;
    ok = ok && cudaMalloc(&groupedLda, batchCount * sizeof(int64_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&groupedLdb, batchCount * sizeof(int64_t)) == cudaSuccess;
    ok = ok && cudaMalloc(&groupedLdc, batchCount * sizeof(int64_t)) == cudaSuccess;
    if (!ok)
    {
      if (groupedProblemSizes)
        cudaFree(groupedProblemSizes);
      if (groupedLda)
        cudaFree(groupedLda);
      if (groupedLdb)
        cudaFree(groupedLdb);
      if (groupedLdc)
        cudaFree(groupedLdc);
      groupedProblemSizes = nullptr;
      groupedLda = groupedLdb = groupedLdc = nullptr;
      return false;
    }
    groupedMetaCapacity = batchCount;
  }

  // CUTLASS wants each operand's physical row stride regardless of its
  // Layout tag -- exactly the caller's own lda/ldb/ldc (which may be
  // padded wider than the bare M/N/K, e.g. DirectKPPCNetwork's
  // psiStride padding -- see that class's own comment), not something
  // to recompute from M/N/K here. No swap needed, unlike cuBLAS's
  // column-major-only M<->N<->lda<->ldb dance.
  const int threads = 256;
  const int blocks = (int)((batchCount + threads - 1) / threads);
  FillGroupedGemmMetadataKernel<<<blocks, threads, 0, stream>>>(
      groupedProblemSizes, groupedLda, groupedLdb, groupedLdc, M, N, K, lda, ldb, ldc, batchCount);

  cachedGroupedM = M;
  cachedGroupedN = N;
  cachedGroupedK = K;
  cachedGroupedLda = lda;
  cachedGroupedLdb = ldb;
  cachedGroupedLdc = ldc;
  cachedGroupedBatchCount = batchCount;
  return true;
}

// Builds Arguments, grows groupedWorkspace if needed, and runs the
// grouped GEMM on `stream`. Returns false on any failure so the caller
// falls back to cublasSgemmBatched() rather than silently skip the
// multiply.
//
// @note threadblockCount is a scheduling hint for the device-side
// ProblemVisitor (how many threadblocks' worth of tile-assignment slots
// to launch), not a correctness parameter -- too low under-utilizes the
// GPU, too high just wastes a few idle threadblocks that exit
// immediately once no tiles remain. CUTLASS's own grouped_gemm example
// computes this from a precomputed host-side problem list via
// Op::sufficient(); every item here shares one shape, so this uses a
// flat occupancy-based estimate instead (threadblocks-per-SM for this
// tile config, times a generous SM-count upper bound) rather than add a
// host-side array just to call sufficient(). Unverified until profiled
// on real hardware -- see this function's caller for the fallback this
// exists to make safe if the estimate is wrong.
template <typename Op>
bool RunGemmGrouped(int* problemSizesRaw, int64_t* lda, int64_t* ldb, int64_t* ldc,
                    const float* const* A_array, const float* const* B_array,
                    float* const* C_array, float alpha, float beta, int batchCount,
                    cudaStream_t stream, uint8_t*& workspace, size_t& workspaceBytes) noexcept
{
  auto* problemSizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(problemSizesRaw);
  typename Op::EpilogueOutputOp::Params epilogueParams(alpha, beta);
  constexpr int kThreadblockCountEstimate = 512;

  // CUTLASS's Arguments wants plain T** (no const anywhere in the outer
  // pointers) for every operand, A/B included, even though it only
  // reads through ptr_A/ptr_B -- the same type is reused symmetrically
  // for C/D, which genuinely get written. const_cast removes const at
  // every level here (A_array/B_array/C_array are T* const*/const T*
  // const*), not just the outer one; that's well-formed (only ever
  // removing cv-qualification, never adding or mismatching it).
  typename Op::Arguments args(problemSizes, batchCount, kThreadblockCountEstimate, epilogueParams,
                              const_cast<float**>(A_array), const_cast<float**>(B_array),
                              const_cast<float**>(C_array), const_cast<float**>(C_array), lda, ldb,
                              ldc, ldc,
                              nullptr); // host_problem_sizes: unused under kDeviceOnly schedule.

  Op gemmOp;
  size_t neededBytes = Op::get_workspace_size(args);
  if (neededBytes > workspaceBytes)
  {
    // Plain cudaMalloc/cudaFree -- see GrowGroupedGemmMetadata's own
    // comment. PrewarmGroupedGemmScratch() below sizes this workspace
    // for every shape a network will ever use before that network's
    // first captured call, so this growth path should never actually
    // run during capture; it's not relied upon to be capture-safe.
    if (workspace)
      cudaFree(workspace);
    workspace = nullptr;
    workspaceBytes = 0;
    if (neededBytes > 0 && cudaMalloc(&workspace, neededBytes) != cudaSuccess)
      return false;
    workspaceBytes = neededBytes;
  }

  if (gemmOp.initialize(args, workspace, stream) != cutlass::Status::kSuccess)
    return false;
  return gemmOp(stream) == cutlass::Status::kSuccess;
}

// Sizes groupedWorkspace for one (transA,transB,M,N,K,lda,ldb,ldc,
// batchCount) shape, without launching anything -- Op::get_workspace_size()
// is a pure function of problem metadata (problem_count, threadblock_count,
// compile-time tile config), never the data pointers, so A/B/C/D are
// nullptr here on purpose: no device buffer needs to exist yet for this
// call to be meaningful.
template <typename Op>
static bool SizeGroupedGemmWorkspace(int* problemSizesRaw, int64_t* lda, int64_t* ldb,
                                     int64_t* ldc, int batchCount, cudaStream_t stream,
                                     uint8_t*& workspace, size_t& workspaceBytes) noexcept
{
  auto* problemSizes = reinterpret_cast<cutlass::gemm::GemmCoord*>(problemSizesRaw);
  typename Op::EpilogueOutputOp::Params epilogueParams(1.0f, 0.0f);
  constexpr int kThreadblockCountEstimate = 512;

  typename Op::Arguments args(problemSizes, batchCount, kThreadblockCountEstimate, epilogueParams,
                              static_cast<float**>(nullptr), static_cast<float**>(nullptr),
                              static_cast<float**>(nullptr), static_cast<float**>(nullptr), lda,
                              ldb, ldc, ldc, nullptr);

  size_t neededBytes = Op::get_workspace_size(args);
  if (neededBytes <= workspaceBytes)
    return true;

  if (workspace)
    cudaFree(workspace);
  workspace = nullptr;
  workspaceBytes = 0;
  if (cudaMalloc(&workspace, neededBytes) != cudaSuccess)
    return false;
  workspaceBytes = neededBytes;
  return true;
}

bool CUDABackend::PrewarmGroupedGemmScratch(bool transA, bool transB, int M, int N, int K,
                                             int64_t lda, int64_t ldb, int64_t ldc,
                                             size_t batchCount) noexcept
{
  if (!GrowGroupedGemmMetadata(batchCount, M, N, K, lda, ldb, ldc))
    return false;

  if (!transA && transB)
    return SizeGroupedGemmWorkspace<GemmGroupedNT>(groupedProblemSizes, groupedLda, groupedLdb,
                                                   groupedLdc, (int)batchCount, stream,
                                                   groupedWorkspace, groupedWorkspaceBytes);
  if (transA && !transB)
    return SizeGroupedGemmWorkspace<GemmGroupedTN>(groupedProblemSizes, groupedLda, groupedLdb,
                                                   groupedLdc, (int)batchCount, stream,
                                                   groupedWorkspace, groupedWorkspaceBytes);
  if (!transA && !transB)
    return SizeGroupedGemmWorkspace<GemmGroupedNN>(groupedProblemSizes, groupedLda, groupedLdb,
                                                   groupedLdc, (int)batchCount, stream,
                                                   groupedWorkspace, groupedWorkspaceBytes);
  return true; // (true,true): never dispatched to CUTLASS, nothing to prewarm.
}

void CUDABackend::MatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
                         int lda, const float* B, int ldb, float beta, float* C, int ldc) noexcept
{
  if (!A || !B || !C)
    return;

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemm(this->handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

bool CUDABackend::GrowBatchedGemmArrays(size_t batchCount) noexcept
{
  if (batchedGemmArrayCapacity >= batchCount)
    return true;

  if (batchedGemmDeviceA)
    cudaFree(batchedGemmDeviceA);
  if (batchedGemmDeviceB)
    cudaFree(batchedGemmDeviceB);
  if (batchedGemmDeviceC)
    cudaFree(batchedGemmDeviceC);
  if (batchedGemmHostA)
    cudaFreeHost(batchedGemmHostA);
  if (batchedGemmHostB)
    cudaFreeHost(batchedGemmHostB);
  if (batchedGemmHostC)
    cudaFreeHost(batchedGemmHostC);
  batchedGemmDeviceA = nullptr;
  batchedGemmDeviceB = nullptr;
  batchedGemmDeviceC = nullptr;
  batchedGemmHostA = nullptr;
  batchedGemmHostB = nullptr;
  batchedGemmHostC = nullptr;
  batchedGemmArrayCapacity = 0;

  const size_t ptrBytes = batchCount * sizeof(const float*);
  bool ok = cudaMalloc(&batchedGemmDeviceA, ptrBytes) == cudaSuccess;
  ok = ok && cudaMalloc(&batchedGemmDeviceB, ptrBytes) == cudaSuccess;
  ok = ok && cudaMalloc(&batchedGemmDeviceC, ptrBytes) == cudaSuccess;
  ok = ok && cudaMallocHost(&batchedGemmHostA, ptrBytes) == cudaSuccess;
  ok = ok && cudaMallocHost(&batchedGemmHostB, ptrBytes) == cudaSuccess;
  ok = ok && cudaMallocHost(&batchedGemmHostC, ptrBytes) == cudaSuccess;

  if (!ok)
  {
    if (batchedGemmDeviceA)
      cudaFree(batchedGemmDeviceA);
    if (batchedGemmDeviceB)
      cudaFree(batchedGemmDeviceB);
    if (batchedGemmDeviceC)
      cudaFree(batchedGemmDeviceC);
    if (batchedGemmHostA)
      cudaFreeHost(batchedGemmHostA);
    if (batchedGemmHostB)
      cudaFreeHost(batchedGemmHostB);
    if (batchedGemmHostC)
      cudaFreeHost(batchedGemmHostC);
    batchedGemmDeviceA = nullptr;
    batchedGemmDeviceB = nullptr;
    batchedGemmDeviceC = nullptr;
    batchedGemmHostA = nullptr;
    batchedGemmHostB = nullptr;
    batchedGemmHostC = nullptr;
    return false;
  }

  batchedGemmArrayCapacity = batchCount;
  return true;
}

bool CUDABackend::PrewarmBatchedGemmScratch(size_t batchCount) noexcept
{
  return GrowBatchedGemmArrays(batchCount);
}

void CUDABackend::MatMulBatched(bool transA, bool transB, int M, int N, int K, float alpha,
                                const float* const* A_array, int lda, const float* const* B_array,
                                int ldb, float beta, float* const* C_array, int ldc,
                                int batchCount) noexcept
{
  if (batchCount <= 0)
    return;
  if (batchCount == 1)
  {
    MatMul(transA, transB, M, N, K, alpha, A_array[0], lda, B_array[0], ldb, beta, C_array[0], ldc);
    return;
  }

  // cublasSgemmBatched's cuBLASLt heuristic (confirmed via compute-sanitizer
  // on an A100: cublasLtSSSMatmul picking ampere_sgemm_128x128_tn, a
  // 128x128-tile kernel) writes past the end of the output buffer whenever
  // any of M/N/K is small -- OUT_DIM=10 reproduced it directly, writing up
  // to ~30 floats past a tightly-packed [OUT_DIM, HIDDEN] allocation. Same
  // class of bug as OpenBLAS's cblas_sgemm_batch small-matrix crash on the
  // CPU side (CPUBackend::MatMulBatched's own kMinSafeDim), and the same
  // shape it hits: a small terminal/output dimension is completely
  // ordinary for a PC network, not an edge case to carve out. Fall back to
  // the interface's own default (a plain MatMul() loop) below this
  // threshold rather than ship a landmine.
  constexpr int kMinSafeDim = 128;
  if (M < kMinSafeDim || N < kMinSafeDim || K < kMinSafeDim)
  {
    IComputeBackend::MatMulBatched(transA, transB, M, N, K, alpha, A_array, lda, B_array, ldb,
                                   beta, C_array, ldc, batchCount);
    return;
  }

  if (!GrowBatchedGemmArrays(static_cast<size_t>(batchCount)))
  {
    std::cerr << "CUDABackend::MatMulBatched: pointer-array scratch allocation failed, "
                 "falling back to per-layer MatMul().\n";
    IComputeBackend::MatMulBatched(transA, transB, M, N, K, alpha, A_array, lda, B_array, ldb,
                                   beta, C_array, ldc, batchCount);
    return;
  }

  // cuBLAS needs Aarray/Barray/Carray THEMSELVES in device memory, unlike
  // CPUBackend's cblas_sgemm_batch (a host array of host pointers). Stage
  // into pinned host memory first, then cudaMemcpyAsync() on `stream` --
  // plain synchronous cudaMemcpy() (as CopyFromHost() uses) isn't safely
  // capturable inside BeginGraphCapture()/EndGraphCapture().
  std::memcpy(batchedGemmHostA, A_array, batchCount * sizeof(const float*));
  std::memcpy(batchedGemmHostB, B_array, batchCount * sizeof(const float*));
  std::memcpy(batchedGemmHostC, C_array, batchCount * sizeof(float*));
  cudaMemcpyAsync(batchedGemmDeviceA, batchedGemmHostA, batchCount * sizeof(const float*),
                 cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(batchedGemmDeviceB, batchedGemmHostB, batchCount * sizeof(const float*),
                 cudaMemcpyHostToDevice, stream);
  cudaMemcpyAsync(batchedGemmDeviceC, batchedGemmHostC, batchCount * sizeof(float*),
                 cudaMemcpyHostToDevice, stream);

  // Try CUTLASS's Grouped GEMM first -- genuinely fuses the whole batch
  // into one GPU-side kernel launch, unlike cublasSgemmBatched() below
  // (see this file's own comment on GemmGroupedOp for why that matters).
  // Only the three (transA,transB) combos actually used today are
  // instantiated; (true,true) and any CUTLASS-side failure (e.g. a
  // workspace allocation failure, or an unsupported shape this first
  // attempt didn't anticipate) fall through to the cuBLAS path below
  // rather than skip the multiply -- unverified against real hardware
  // yet, so this fallback is load-bearing, not defensive boilerplate.
  bool cutlassOk = false;
  if (GrowGroupedGemmMetadata(static_cast<size_t>(batchCount), M, N, K, lda, ldb, ldc))
  {
    if (!transA && transB)
      cutlassOk = RunGemmGrouped<GemmGroupedNT>(groupedProblemSizes, groupedLda, groupedLdb,
                                                groupedLdc, batchedGemmDeviceA, batchedGemmDeviceB,
                                                batchedGemmDeviceC, alpha, beta, batchCount, stream,
                                                groupedWorkspace, groupedWorkspaceBytes);
    else if (transA && !transB)
      cutlassOk = RunGemmGrouped<GemmGroupedTN>(groupedProblemSizes, groupedLda, groupedLdb,
                                                groupedLdc, batchedGemmDeviceA, batchedGemmDeviceB,
                                                batchedGemmDeviceC, alpha, beta, batchCount, stream,
                                                groupedWorkspace, groupedWorkspaceBytes);
    else if (!transA && !transB)
      cutlassOk = RunGemmGrouped<GemmGroupedNN>(groupedProblemSizes, groupedLda, groupedLdb,
                                                groupedLdc, batchedGemmDeviceA, batchedGemmDeviceB,
                                                batchedGemmDeviceC, alpha, beta, batchCount, stream,
                                                groupedWorkspace, groupedWorkspaceBytes);
  }
  if (cutlassOk)
    return;

  // Same row-major -> column-major swap as the plain MatMul() above.
  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemmBatched(handle, opB, opA, N, M, K, &alpha, batchedGemmDeviceB, ldb, batchedGemmDeviceA,
                     lda, &beta, batchedGemmDeviceC, ldc, batchCount);
}

void CUDABackend::SetAllowTF32(bool allow) noexcept
{
  this->allowTF32 = allow;
  cublasSetMathMode(this->handle, allow ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
}

void CUDABackend::SumRows(float* dst, const float* src, size_t batchSize, size_t width) noexcept
{
  if (!dst || !src)
    return;

  if (!onesVector || onesCapacity < batchSize)
    PrepareForBatchSize(batchSize);

  float alpha = 1.0f, beta = 0.0f;
  cublasSgemv(
      handle, CUBLAS_OP_N, width, batchSize, &alpha, src, width, onesVector, 1, &beta, dst, 1);
}

void CUDABackend::SumRowsBatched(float* dst, const float* src, size_t batchSize, size_t width,
                                 size_t numLayers) noexcept
{
  if (!dst || !src || numLayers == 0)
    return;

  if (!onesVector || onesCapacity < batchSize)
    PrepareForBatchSize(batchSize);

  // Same cublasSgemv() as SumRows() above, but cublasSgemvStridedBatched:
  // a mature, standard cuBLAS primitive (unlike MatMulBatched()'s own
  // CUTLASS Grouped GEMM adventure above) for exactly this "numLayers
  // independent problems, one dispatch" shape. x (onesVector) is the
  // same buffer reused for every layer, so its stride is 0; A (src) and
  // y (dst) each advance by one layer's worth of data per problem.
  float alpha = 1.0f, beta = 0.0f;
  cublasSgemvStridedBatched(handle, CUBLAS_OP_N, (int)width, (int)batchSize, &alpha, src,
                            (int)width, (long long)(batchSize * width), onesVector, 1, 0, &beta,
                            dst, 1, (long long)width, (int)numLayers);
}

bool CUDABackend::TryFusedForwardPass(ActivationType actType, const float* zF, const float* W,
                                      const float* bias, float* mu, int batchSize, int size,
                                      int nextSize) noexcept
{
  int split_k_slices = 1;
  bool wide = (nextSize % 4 == 0);

  if (actType == ActivationType::RELU)
  {
    if (wide)
    {
      GemmFwdRelu gemm_op;
      GemmFwdRelu::Arguments args({batchSize, nextSize, size},
                                  {zF, size},
                                  {W, size},
                                  {bias, 0},
                                  {mu, nextSize},
                                  {1.0f, 1.0f},
                                  split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdReluScalar gemm_op;
    GemmFwdReluScalar::Arguments args({batchSize, nextSize, size},
                                      {zF, size},
                                      {W, size},
                                      {bias, 0},
                                      {mu, nextSize},
                                      {1.0f, 1.0f},
                                      split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }
  else if (actType == ActivationType::LINEAR)
  {
    if (wide)
    {
      GemmFwdLinear gemm_op;
      GemmFwdLinear::Arguments args({batchSize, nextSize, size},
                                    {zF, size},
                                    {W, size},
                                    {bias, 0},
                                    {mu, nextSize},
                                    {1.0f, 1.0f},
                                    split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdLinearScalar gemm_op;
    GemmFwdLinearScalar::Arguments args({batchSize, nextSize, size},
                                        {zF, size},
                                        {W, size},
                                        {bias, 0},
                                        {mu, nextSize},
                                        {1.0f, 1.0f},
                                        split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }
  else if (actType == ActivationType::GELU)
  {
    if (wide)
    {
      GemmFwdGELU gemm_op;
      GemmFwdGELU::Arguments args({batchSize, nextSize, size},
                                  {zF, size},
                                  {W, size},
                                  {bias, 0},
                                  {mu, nextSize},
                                  {1.0f, 1.0f},
                                  split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdGELUScalar gemm_op;
    GemmFwdGELUScalar::Arguments args({batchSize, nextSize, size},
                                      {zF, size},
                                      {W, size},
                                      {bias, 0},
                                      {mu, nextSize},
                                      {1.0f, 1.0f},
                                      split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }
  else if (actType == ActivationType::TANH)
  {
    if (wide)
    {
      GemmFwdTanh gemm_op;
      GemmFwdTanh::Arguments args({batchSize, nextSize, size},
                                  {zF, size},
                                  {W, size},
                                  {bias, 0},
                                  {mu, nextSize},
                                  {1.0f, 1.0f},
                                  split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdTanhScalar gemm_op;
    GemmFwdTanhScalar::Arguments args({batchSize, nextSize, size},
                                      {zF, size},
                                      {W, size},
                                      {bias, 0},
                                      {mu, nextSize},
                                      {1.0f, 1.0f},
                                      split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }
  else if (actType == ActivationType::SIGMOID)
  {
    if (wide)
    {
      GemmFwdSigmoid gemm_op;
      GemmFwdSigmoid::Arguments args({batchSize, nextSize, size},
                                     {zF, size},
                                     {W, size},
                                     {bias, 0},
                                     {mu, nextSize},
                                     {1.0f, 1.0f},
                                     split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdSigmoidScalar gemm_op;
    GemmFwdSigmoidScalar::Arguments args({batchSize, nextSize, size},
                                         {zF, size},
                                         {W, size},
                                         {bias, 0},
                                         {mu, nextSize},
                                         {1.0f, 1.0f},
                                         split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }

  return false;
}
} // namespace Deep

#endif
