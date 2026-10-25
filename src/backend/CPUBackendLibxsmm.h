#pragma once

/**
 * @file CPUBackendLibxsmm.h
 * @brief Declares the libxsmm-backed small-matrix GEMM used by
 * CPUBackend::TryFusedMatMul(). Deliberately a separate CMake target
 * (`DeepityLibxsmm`, see CMakeLists.txt) that never links or sees MKL:
 * libxsmm's CMake build (unlike its GNU Make build) hardcodes its
 * generated config to LP64 (32-bit libxsmm_blasint) with no way to
 * request ILP64 through this build path, and libxsmm_typedefs.h refuses
 * to compile at all if it sees MKL_ILP64 defined alongside that (a
 * defensive check against actually calling into an ILP64 BLAS through
 * libxsmm's own BLAS-wrapper layer, which this file never does). Rather
 * than downgrade the whole project off MKL_ILP64 to accommodate that,
 * this translation unit is kept out of MKL_ILP64's reach entirely, the
 * same way CPUBackendHighway.cpp is kept out of the project's pinned
 * -march flag.
 */

namespace Deep
{
/// @brief Attempts C = A @ B(^T) via libxsmm's JIT-dispatched small-GEMM
/// kernel (row-major, same calling convention as CPUBackend::MatMul()).
/// Returns false if libxsmm can't dispatch a kernel for this shape (the
/// caller falls back to MatMul()) -- notably, libxsmm's kernels bake
/// alpha=1 in at dispatch time, so this always returns false for any
/// other alpha; callers needing alpha!=1 should try oneDNN first.
/// @param transA/transB/M/N/K/A/lda/B/ldb/C/ldc Same meaning and shapes
/// as MatMul(). Alpha=1 and beta=0 are implied.
/// @param bias When non-null, a length-N vector broadcast-added to
/// every row of C (same convention as AddBiasBroadcast), fused into
/// the SAME per-row-block parallel loop that computes that block's
/// GEMM output -- not a separate pass. libxsmm has no bias epilogue of
/// its own, unlike oneDNN's fused path; doing the add as its own
/// AddBiasBroadcast() call afterward would need a second OpenMP region
/// and a second full memory pass over C, which measurably regressed
/// the mu GEMM call site (bias add dominates far less than the GEMM
/// itself, but a second pass + region-entry overhead was still enough
/// to erase libxsmm's GEMM win there).
bool TryLibxsmmMatMul(bool transA, bool transB, int M, int N, int K, const float* A, int lda,
                      const float* B, int ldb, float* C, int ldc,
                      const float* bias = nullptr) noexcept;
} // namespace Deep
