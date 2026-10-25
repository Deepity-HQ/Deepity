#include "CPUBackendLibxsmm.h"

#include <algorithm>
#include <libxsmm.h>
#include <map>
#include <omp.h>
#include <tuple>

namespace Deep
{
namespace
{
// Keyed on every shape parameter, since dispatching JIT-compiles a
// kernel for that exact shape (expensive); invoking a cached one is
// cheap. A function-local static would not be safe here: the row-
// blocked loop below reads this map from multiple OpenMP threads at
// once, and concurrent std::map lookups are only safe with no
// concurrent writes -- so every shape a call might need is dispatched
// (written) single-threaded first, strictly before any parallel region
// that reads it. See Dispatch()'s callers for where that ordering is
// enforced.
std::map<std::tuple<bool, bool, int, int, int, int, int, int>, libxsmm_gemmfunction> g_cache;

// Row-major C(M,N)=opA(A)@opB(B) via libxsmm's column-major API: swap
// operands and M<->N, keep each operand's own transpose flag in its new
// slot (standard row-major-via-column-major identity; verified against
// a reference implementation for all 3 of SimplePCLayer's real GEMM
// shapes before landing this).
libxsmm_gemmfunction Dispatch(bool transA, bool transB, int M, int N, int K, int lda, int ldb,
                              int ldc)
{
  const auto key = std::make_tuple(transA, transB, M, N, K, lda, ldb, ldc);
  auto it = g_cache.find(key);
  if (it == g_cache.end())
  {
    const libxsmm_gemm_shape shape = libxsmm_create_gemm_shape(
        N, M, K, ldb, lda, ldc, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32,
        LIBXSMM_DATATYPE_F32);
    const int flags =
        LIBXSMM_GEMM_FLAGS(transB ? 'T' : 'N', transA ? 'T' : 'N') | LIBXSMM_GEMM_FLAG_BETA_0;
    it = g_cache
             .emplace(key, libxsmm_dispatch_gemm(shape, (libxsmm_bitfield)flags,
                                                 (libxsmm_bitfield)LIBXSMM_GEMM_PREFETCH_NONE))
             .first;
  }
  return it->second;
}

// rows/N/ldc describe only the slice of C this one block just wrote --
// called from inside the same per-thread loop iteration as Run() below,
// never as its own pass, so a caller's bias-add rides for free on
// memory that's already hot in this thread's cache.
void AddBiasRows(float* C, const float* bias, int rows, int N, int ldc)
{
  for (int r = 0; r < rows; ++r)
  {
    float* row = C + (size_t)r * ldc;
    for (int c = 0; c < N; ++c)
      row[c] += bias[c];
  }
}

void Run(libxsmm_gemmfunction kernel, const float* A, const float* B, float* C, const float* bias,
         int rows, int N, int ldc)
{
  libxsmm_gemm_param param;
  param.a.primary = const_cast<float*>(B); // swapped: libxsmm's "a" is our B
  param.b.primary = const_cast<float*>(A); // libxsmm's "b" is our A
  param.c.primary = C;
  kernel(&param);
  if (bias != nullptr)
    AddBiasRows(C, bias, rows, N, ldc);
}
} // namespace

bool TryLibxsmmMatMul(bool transA, bool transB, int M, int N, int K, const float* A, int lda,
                      const float* B, int ldb, float* C, int ldc, const float* bias) noexcept
{
  // libxsmm's dispatched kernels are deliberately single-threaded per
  // call -- parallelism is the caller's job (see libxsmm's own docs on
  // wrapping the dispatch loop in OpenMP). Without this, a single call
  // loses MKL's automatic multi-core cblas_sgemm and regresses badly at
  // our "medium" (not tiny) shapes -- measured ~3x slower end to end
  // before this was added. Row-blocking across threads needs A's
  // M-indexed dimension to be contiguous per row, i.e. transA==false --
  // true for every real caller today (UpdateState()'s feedback GEMM,
  // the only alpha==1 call site -- see CPUBackend::TryFusedMatMul).
  // transA==true falls through to the single-dispatch path below.
  if (!transA)
  {
    const int numThreads = std::min(omp_get_max_threads(), M);
    if (numThreads > 1)
    {
      const int chunk = (M + numThreads - 1) / numThreads;
      const int lastRows = M - (numThreads - 1) * chunk;

      // Dispatch (and cache-insert) both possible block shapes here,
      // single-threaded, before the parallel region below ever reads
      // g_cache -- see g_cache's own comment for why this ordering
      // matters.
      const libxsmm_gemmfunction fullBlock = Dispatch(false, transB, chunk, N, K, lda, ldb, ldc);
      const libxsmm_gemmfunction tailBlock =
          (lastRows != chunk) ? Dispatch(false, transB, lastRows, N, K, lda, ldb, ldc) : fullBlock;
      if (fullBlock == nullptr || tailBlock == nullptr)
        return false; // not dispatchable at all -- caller falls back to MatMul()

#pragma omp parallel for schedule(static)
      for (int t = 0; t < numThreads; ++t)
      {
        const int startRow = t * chunk;
        const int rows = (t == numThreads - 1) ? lastRows : chunk;
        Run(rows == chunk ? fullBlock : tailBlock, A + (size_t)startRow * lda, B,
            C + (size_t)startRow * ldc, bias, rows, N, ldc);
      }
      return true;
    }
  }

  // Fallback: single dispatch, no row-blocking (transA==true, or M too
  // small to be worth splitting across threads).
  const libxsmm_gemmfunction kernel = Dispatch(transA, transB, M, N, K, lda, ldb, ldc);
  if (kernel == nullptr)
    return false; // not dispatchable for this shape -- caller falls back to MatMul()
  Run(kernel, A, B, C, bias, M, N, ldc);
  return true;
}
} // namespace Deep
