#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deepity/backend/CPUBackend.h>
#include <deepity/utils/Im2Col.h>
#include <deepity/utils/MaxPool2D.h>
#include <stdexcept>

#include "CPUBackendHighway.h"

#ifdef _WIN32
#include <malloc.h>
#else
#include <cstdlib>
#endif

#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <random>

#ifdef DEEPITY_USE_ONEDNN
#include <map>
#include <oneapi/dnnl/dnnl.hpp>
#include <tuple>
#endif

#ifdef DEEPITY_USE_LIBXSMM
#include "CPUBackendLibxsmm.h"
#endif

#ifdef DEEPITY_USE_HALIDE
#include "HalideBuffer.h"
#include "deepity_feedback_update_sigmoid.h"
#endif

namespace Deep
{
#ifdef DEEPITY_USE_ONEDNN
struct CPUBackend::OneDnnMatMulCache
{
  dnnl::engine engine{dnnl::engine::kind::cpu, 0};
  dnnl::stream stream{engine};

  struct Entry
  {
    dnnl::memory::desc srcMd, weightsMd, biasMd, dstMd;
    dnnl::matmul prim;
  };
  // Keyed on every shape parameter TryFusedMatMulBias() takes, since
  // building a dnnl::matmul JIT-compiles a kernel for that exact shape
  // (expensive); executing a cached one is cheap. Every real caller
  // (SimplePCLayer::ComputeMuOnly()) uses the SAME shape on every
  // settling step of every training iteration, so this amortizes to
  // once per distinct shape a network actually has.
  //
  // Not mutex-guarded: every current caller invokes backend methods
  // sequentially from one thread (the training loop), never concurrently
  // across layers -- see SimplePCNetwork::TrainStepWithProjection.
  std::map<std::tuple<bool, bool, int, int, int, int, int, int>, Entry> cache;

  struct NoBiasEntry
  {
    dnnl::memory::desc srcMd, weightsMd, dstMd;
    dnnl::matmul prim;
  };
  // Separate from `cache` above: TryFusedMatMul() has no bias operand
  // and instead folds `alpha` in as a post-op, so its primitive_desc
  // shape (3 operands, not 4) and cache key (alpha included) both
  // differ from TryFusedMatMulBias()'s.
  std::map<std::tuple<bool, bool, int, int, int, int, int, int, float>, NoBiasEntry> noBiasCache;

  struct FeedbackFusedEntry
  {
    dnnl::memory::desc srcMd, weightsMd, dstMd;
    dnnl::matmul prim;
  };
  // Keyed on (batchSize, size, nextSize) only: TryFusedFeedbackUpdate()'s
  // operands are always non-transposed and tightly packed (see
  // SimplePCLayer::UpdateState(), the only caller), unlike
  // TryFusedMatMul{,Bias}() above which also vary transA/transB/strides.
  std::map<std::tuple<int, int, int>, FeedbackFusedEntry> feedbackFusedCache;
};
#endif

CPUBackend::CPUBackend()
{
#ifdef DEEPITY_USE_ONEDNN
  m_oneDnnCache = std::make_unique<OneDnnMatMulCache>();
#endif
}

CPUBackend::~CPUBackend() = default;

float* CPUBackend::Allocate(size_t numFloats)
{
#ifdef _WIN32
  return (float*)_aligned_malloc(numFloats * sizeof(float), 16);
#else
  return (float*)aligned_alloc(16, numFloats * sizeof(float));
#endif
}

void CPUBackend::Free(float* ptr) noexcept
{
#ifdef _WIN32
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

void CPUBackend::Zero(float* ptr, size_t numFloats) noexcept
{
  memset(ptr, 0, numFloats * sizeof(float));
}

void CPUBackend::Copy(float* dst, const float* src, size_t numFloats) noexcept
{
  memcpy(dst, src, numFloats * sizeof(float));
}

void CPUBackend::CopyFromHost(float* deviceDst, const float* hostSrc, size_t numFloats) noexcept
{
  memcpy(deviceDst, hostSrc, numFloats * sizeof(float));
}

void CPUBackend::CopyToHost(float* hostDst, const float* deviceSrc, size_t numFloats) noexcept
{
  memcpy(hostDst, deviceSrc, numFloats * sizeof(float));
}

void CPUBackend::RandomizeNormal(float* buf, size_t n, float mean, float stddev,
                                 uint32_t seed) noexcept
{
  std::mt19937 seedGenerator(seed);
  std::uniform_int_distribution<uint32_t> seedDist;

  std::vector<uint32_t> seeds(omp_get_max_threads());
  for (auto& s : seeds)
    s = seedDist(seedGenerator);

#pragma omp parallel if (!omp_in_parallel())
  {
    std::mt19937 rng(seeds[omp_get_thread_num()]);
    std::normal_distribution<float> dist(mean, stddev);

#pragma omp for
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i)
      buf[i] = dist(rng);
  }
}

void CPUBackend::RandomizeUniform(float* buf, size_t n, float min, float max,
                                  uint32_t seed) noexcept
{
  std::mt19937 seedGenerator(seed);
  std::uniform_int_distribution<uint32_t> seedDist;

  std::vector<uint32_t> seeds(omp_get_max_threads());
  for (auto& s : seeds)
    s = seedDist(seedGenerator);

#pragma omp parallel if (!omp_in_parallel())
  {
    std::mt19937 rng(seeds[omp_get_thread_num()]);
    std::uniform_real_distribution<float> dist(min, max);

#pragma omp for
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i)
      buf[i] = dist(rng);
  }
}

namespace
{
// TEMPORARY profiling accumulator -- see DebugGetGemmSeconds()'s declaration.
double g_debugGemmSeconds = 0.0;
long g_debugGemmCalls = 0;
} // namespace

double DebugGetGemmSeconds() noexcept { return g_debugGemmSeconds; }
long DebugGetGemmCalls() noexcept { return g_debugGemmCalls; }

void CPUBackend::MatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
                        int lda, const float* B, int ldb, float beta, float* C, int ldc) noexcept
{ // Multithreading is the caller's job.
  auto t0 = std::chrono::steady_clock::now();
  cblas_sgemm(CblasRowMajor,
              transA ? CblasTrans : CblasNoTrans,
              transB ? CblasTrans : CblasNoTrans,
              M,
              N,
              K,
              alpha,
              A,
              lda,
              B,
              ldb,
              beta,
              C,
              ldc);
  auto t1 = std::chrono::steady_clock::now();
  g_debugGemmSeconds += std::chrono::duration<double>(t1 - t0).count();
  ++g_debugGemmCalls;
}

void CPUBackend::SumRows(float* dst, const float* src, size_t batchSize, size_t width) noexcept
{
  memset(dst, 0, width * sizeof(float));
  for (size_t b = 0; b < batchSize; ++b)
    cblas_saxpy(width, 1.0f, src + b * width, 1, dst, 1);
}

float CPUBackend::Sum(const float* buf, size_t n) noexcept
{
  float total = 0.0f;
#pragma omp parallel for schedule(static) reduction(+ : total) if (n > 4096 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
    total += buf[i];
  return total;
}

void CPUBackend::Scale(float* buf, size_t n, float alpha) noexcept
{
  cblas_sscal(n, alpha, buf, 1);
}

void CPUBackend::AxpyInto(float* y, const float* x, size_t n, float alpha) noexcept
{
  cblas_saxpy(n, alpha, x, 1, y, 1);
}

void CPUBackend::AddBiasBroadcast(float* buf, const float* bias, size_t batchSize,
                                  size_t width) noexcept
{
#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
  for (size_t b = 0; b < batchSize; ++b)
    cblas_saxpy(width, 1.0f, bias, 1, buf + b * width, 1);
}

bool CPUBackend::TryFusedMatMulBias(bool transA, bool transB, int M, int N, int K, const float* A,
                                    int lda, const float* B, int ldb, const float* bias, float* C,
                                    int ldc) noexcept
{
#ifndef DEEPITY_USE_ONEDNN
  (void)transA, (void)transB, (void)M, (void)N, (void)K, (void)A, (void)lda, (void)B, (void)ldb,
      (void)bias, (void)C, (void)ldc;
#else
  // Row-major operand with logical shape (R,C): contiguous (ld>=C) when
  // not transposed, strides {ld,1}; when transposed, the array is
  // physically (C,R) row-major with ld>=R, so logical dim R is the
  // contiguous one -- strides {1,ld}. Matches cblas_sgemm's own
  // RowMajor+Trans semantics exactly (see MatMul() above), just
  // expressed as explicit strides instead of a transpose flag, since
  // dnnl::matmul has no transpose flag of its own.
  auto operandMd = [](int R, int C, int ld, bool transposed)
  {
    return dnnl::memory::desc(
        {R, C}, dnnl::memory::data_type::f32,
        transposed ? dnnl::memory::dims{1, ld} : dnnl::memory::dims{ld, 1});
  };

  try
  {
    auto key = std::make_tuple(transA, transB, M, N, K, lda, ldb, ldc);
    auto it = m_oneDnnCache->cache.find(key);
    if (it == m_oneDnnCache->cache.end())
    {
      OneDnnMatMulCache::Entry entry;
      entry.srcMd = operandMd(M, K, lda, transA);
      entry.weightsMd = operandMd(K, N, ldb, transB);
      entry.dstMd = operandMd(M, N, ldc, false);
      entry.biasMd =
          dnnl::memory::desc({1, N}, dnnl::memory::data_type::f32, dnnl::memory::format_tag::ab);
      dnnl::matmul::primitive_desc pd(m_oneDnnCache->engine, entry.srcMd, entry.weightsMd,
                                      entry.biasMd, entry.dstMd);
      entry.prim = dnnl::matmul(pd);
      it = m_oneDnnCache->cache.emplace(key, std::move(entry)).first;
    }
    const auto& entry = it->second;
    const auto& eng = m_oneDnnCache->engine;

    // User formats were given explicitly above (never format_tag::any),
    // so dnnl executes directly on this memory, no implicit reorder.
    dnnl::memory srcMem(entry.srcMd, eng, const_cast<float*>(A));
    dnnl::memory weightsMem(entry.weightsMd, eng, const_cast<float*>(B));
    dnnl::memory biasMem(entry.biasMd, eng, const_cast<float*>(bias));
    dnnl::memory dstMem(entry.dstMd, eng, C);

    entry.prim.execute(m_oneDnnCache->stream,
                       {{DNNL_ARG_SRC, srcMem},
                        {DNNL_ARG_WEIGHTS, weightsMem},
                        {DNNL_ARG_BIAS, biasMem},
                        {DNNL_ARG_DST, dstMem}});
    m_oneDnnCache->stream.wait();
    return true;
  }
  catch (const dnnl::error&)
  {
    // falls through to libxsmm/false below, same as !DEEPITY_USE_ONEDNN
  }
#endif
#ifdef DEEPITY_USE_LIBXSMM
  // Fallback, not first choice: oneDNN's dnnl::matmul above already
  // internally parallelizes and consistently beat a hand-rolled
  // OpenMP-row-blocked libxsmm dispatch loop at every shape this
  // project actually uses when measured head to head (~13% slower
  // overall on a full MNIST profiling pass) -- see CPUBackendLibxsmm.h.
  // Kept as a fallback for builds without oneDNN, where libxsmm still
  // beats plain cblas_sgemm's per-call dispatch overhead.
  if (TryLibxsmmMatMul(transA, transB, M, N, K, A, lda, B, ldb, C, ldc, bias))
    return true;
#endif
  return false; // caller falls back to MatMul()+AddBiasBroadcast()
}

bool CPUBackend::TryFusedMatMul(bool transA, bool transB, int M, int N, int K, float alpha,
                                const float* A, int lda, const float* B, int ldb, float* C,
                                int ldc) noexcept
{
#ifndef DEEPITY_USE_ONEDNN
  (void)transA, (void)transB, (void)M, (void)N, (void)K, (void)alpha, (void)A, (void)lda,
      (void)B, (void)ldb, (void)C, (void)ldc;
#else
  auto operandMd = [](int R, int C, int ld, bool transposed)
  {
    return dnnl::memory::desc(
        {R, C}, dnnl::memory::data_type::f32,
        transposed ? dnnl::memory::dims{1, ld} : dnnl::memory::dims{ld, 1});
  };

  try
  {
    auto key = std::make_tuple(transA, transB, M, N, K, lda, ldb, ldc, alpha);
    auto it = m_oneDnnCache->noBiasCache.find(key);
    if (it == m_oneDnnCache->noBiasCache.end())
    {
      OneDnnMatMulCache::NoBiasEntry entry;
      entry.srcMd = operandMd(M, K, lda, transA);
      entry.weightsMd = operandMd(K, N, ldb, transB);
      entry.dstMd = operandMd(M, N, ldc, false);

      // alpha folded in as a post-op (dst = 1*(A@B) + 0, then
      // eltwise_linear scales it by alpha) rather than a primitive
      // attribute, since dnnl::matmul has no alpha scalar of its own --
      // see MatMul()'s cblas_sgemm call for the primitive this mirrors.
      dnnl::post_ops postOps;
      postOps.append_eltwise(dnnl::algorithm::eltwise_linear, alpha, 0.0f);
      dnnl::primitive_attr attr;
      attr.set_post_ops(postOps);

      dnnl::matmul::primitive_desc pd(m_oneDnnCache->engine, entry.srcMd, entry.weightsMd,
                                      entry.dstMd, attr);
      entry.prim = dnnl::matmul(pd);
      it = m_oneDnnCache->noBiasCache.emplace(key, std::move(entry)).first;
    }
    const auto& entry = it->second;
    const auto& eng = m_oneDnnCache->engine;

    dnnl::memory srcMem(entry.srcMd, eng, const_cast<float*>(A));
    dnnl::memory weightsMem(entry.weightsMd, eng, const_cast<float*>(B));
    dnnl::memory dstMem(entry.dstMd, eng, C);

    entry.prim.execute(
        m_oneDnnCache->stream,
        {{DNNL_ARG_SRC, srcMem}, {DNNL_ARG_WEIGHTS, weightsMem}, {DNNL_ARG_DST, dstMem}});
    m_oneDnnCache->stream.wait();
    return true;
  }
  catch (const dnnl::error&)
  {
    // falls through to libxsmm/false below, same as !DEEPITY_USE_ONEDNN
  }
#endif
#ifdef DEEPITY_USE_LIBXSMM
  // Fallback, not first choice -- see TryFusedMatMulBias()'s matching
  // comment. libxsmm's JIT'd kernels also bake alpha=1 in at dispatch
  // time (no runtime scalar), so this only ever applies when the
  // caller's alpha is exactly 1 (true for UpdateState()'s feedback
  // GEMM, not for UpdateWeights()'s grad_scale=-1 GEMM).
  if (alpha == 1.0f && TryLibxsmmMatMul(transA, transB, M, N, K, A, lda, B, ldb, C, ldc))
    return true;
#endif
  return false; // caller falls back to MatMul()
}

bool CPUBackend::TryFusedFeedbackUpdate(ActivationType dType, int batchSize, int size,
                                        int nextSize, const float* eAbove, const float* W,
                                        float* z, const float* e, float ir, float* scratch,
                                        float* derivScratch) noexcept
{
#ifdef DEEPITY_USE_ONEDNN
  // Primary path: weld the elementwise update onto the feedback GEMM as
  // post-ops on the SAME dnnl::matmul primitive, evaluated in registers
  // on each output tile right as oneDNN's own GEMM microkernel produces
  // it -- same kernel, same thread pool as every other matmul in this
  // file, no separate elementwise pass and no second thread pool
  // competing with this one (see this method's @copydoc in
  // IComputeBackend.h for why that second-thread-pool problem is exactly
  // what sank the Halide fallback below). This mirrors what XLA's CPU
  // backend does for jpc/JAX's matmul+elementwise fusions (confirmed via
  // strings in jaxlib's own binary: PopulateOneDnnPostOps,
  // OneDnnFusionConfig).
  try
  {
    size_t n = (size_t)batchSize * size;

    // derivScratch = f'(z) is the only elementwise precompute pass this
    // needs -- e and z both feed the post-op chain directly below
    // (unscaled), with the ir scale and the -e subtraction folded into
    // the chain itself instead of a second/third precompute pass.
    // ActivationDerivativeInto() handles every ActivationType this
    // project has (see To_dFn2()'s switch), unlike the Halide kernel
    // below which only ever covered SIGMOID.
    ActivationDerivativeInto(dType, derivScratch, z, n);

    auto key = std::make_tuple(batchSize, size, nextSize);
    auto it = m_oneDnnCache->feedbackFusedCache.find(key);
    if (it == m_oneDnnCache->feedbackFusedCache.end())
    {
      OneDnnMatMulCache::FeedbackFusedEntry entry;
      entry.srcMd = dnnl::memory::desc({batchSize, nextSize}, dnnl::memory::data_type::f32,
                                       dnnl::memory::dims{nextSize, 1});
      entry.weightsMd = dnnl::memory::desc({nextSize, size}, dnnl::memory::data_type::f32,
                                           dnnl::memory::dims{size, 1});
      entry.dstMd = dnnl::memory::desc({batchSize, size}, dnnl::memory::data_type::f32,
                                       dnnl::memory::dims{size, 1});

      // dst = feedback; dst = dst*derivScratch; dst = dst-e; dst = ir*dst;
      // dst = dst+z  ==>  dst = z + ir*(f'(z)*feedback - e), exactly
      // FusedStateUpdate()'s formula.
      dnnl::post_ops postOps;
      postOps.append_binary(dnnl::algorithm::binary_mul, entry.dstMd);
      postOps.append_binary(dnnl::algorithm::binary_sub, entry.dstMd);
      postOps.append_eltwise(dnnl::algorithm::eltwise_linear, ir, 0.0f);
      postOps.append_binary(dnnl::algorithm::binary_add, entry.dstMd);
      dnnl::primitive_attr attr;
      attr.set_post_ops(postOps);

      dnnl::matmul::primitive_desc pd(m_oneDnnCache->engine, entry.srcMd, entry.weightsMd,
                                      entry.dstMd, attr);
      entry.prim = dnnl::matmul(pd);
      it = m_oneDnnCache->feedbackFusedCache.emplace(key, std::move(entry)).first;
    }
    const auto& entry = it->second;
    const auto& eng = m_oneDnnCache->engine;

    dnnl::memory srcMem(entry.srcMd, eng, const_cast<float*>(eAbove));
    dnnl::memory weightsMem(entry.weightsMd, eng, const_cast<float*>(W));
    // scratch (never z itself -- see this method's no-aliasing @copydoc)
    // is the primitive's dst; one plain copy below finishes the
    // in-place-from-the-caller's-perspective contract.
    dnnl::memory dstMem(entry.dstMd, eng, scratch);
    dnnl::memory derivMem(entry.dstMd, eng, derivScratch);
    dnnl::memory eMem(entry.dstMd, eng, const_cast<float*>(e));
    dnnl::memory zMem(entry.dstMd, eng, const_cast<float*>(z));

    // Post-op 2 (eltwise_linear) has no runtime memory operand -- its
    // scale is baked into the post-op descriptor above -- so the binary
    // post-ops' indices in the chain are 0, 1, 3, not 0, 1, 2.
    entry.prim.execute(
        m_oneDnnCache->stream,
        {{DNNL_ARG_SRC, srcMem},
         {DNNL_ARG_WEIGHTS, weightsMem},
         {DNNL_ARG_DST, dstMem},
         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1, derivMem},
         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(1) | DNNL_ARG_SRC_1, eMem},
         {DNNL_ARG_ATTR_MULTIPLE_POST_OP(3) | DNNL_ARG_SRC_1, zMem}});
    m_oneDnnCache->stream.wait();

    std::memcpy(z, scratch, n * sizeof(float));
    return true;
  }
  catch (const dnnl::error&)
  {
    // falls through to Halide/false below
  }
#endif

#ifdef DEEPITY_USE_HALIDE
  // Secondary fallback only: this project's first attempt at this
  // fusion, kept because it's verified correct, not because it's fast
  // -- see this method's @copydoc in IComputeBackend.h. Only SIGMOID has
  // an AOT kernel, and only for nextSize>=64 (a short reduction doesn't
  // amortize the tiled schedule's overhead).
  if (dType == ActivationType::dSIGMOID && nextSize >= 64)
  {
    Halide::Runtime::Buffer<const float> eAboveBuf(eAbove, nextSize, batchSize);
    Halide::Runtime::Buffer<const float> wBuf(W, size, nextSize);
    Halide::Runtime::Buffer<const float> zInBuf(z, size, batchSize);
    Halide::Runtime::Buffer<const float> eBuf(e, size, batchSize);
    Halide::Runtime::Buffer<float> scratchBuf(scratch, size, batchSize);

    if (deepity_feedback_update_sigmoid(eAboveBuf, wBuf, zInBuf, eBuf, ir, scratchBuf) == 0)
    {
      std::memcpy(z, scratch, (size_t)batchSize * size * sizeof(float));
      return true;
    }
  }
#endif

  (void)dType, (void)batchSize, (void)size, (void)nextSize, (void)eAbove, (void)W, (void)z,
      (void)e, (void)ir, (void)scratch, (void)derivScratch;
  return false;
}

void CPUBackend::Activation(ActivationType type, float* buf, size_t n) noexcept
{
  To_Fn(type)(buf, n);
}

void CPUBackend::ActivationInto(ActivationType type, float* dst, const float* src,
                                size_t n) noexcept
{
  cblas_scopy(n, src, 1, dst, 1);
  To_Fn(type)(dst, n);
}

void CPUBackend::ActivationDerivative(ActivationType type, float* buf, size_t n,
                                      bool activated) noexcept
{
  To_dFn(type)(buf, n, activated);
}

void CPUBackend::ActivationDerivativeInto(ActivationType type, float* dst, const float* src,
                                          size_t n) noexcept
{
  To_dFn2(type)(dst, src, n);
}

namespace
{
// dGELU, dSIGMOID and dTANH need a transcendental (SLEEF tanh or
// std::exp) when evaluated from the raw pre-activation z, as
// FusedStateUpdate/FusedStateUpdateMomentum do -- see
// CPUBackendHighway.cpp's file comment. Those three stay on the scalar
// path below; everything else routes through the Highway dispatch.
bool NeedsTranscendentalDerivative(ActivationType dType) noexcept
{
  return dType == ActivationType::dGELU || dType == ActivationType::dSIGMOID ||
         dType == ActivationType::dTANH;
}
} // namespace

void CPUBackend::FusedStateUpdateMomentum(float* z, float* v, const float* feedback,
                                          ActivationType dType, const float* e, size_t n, float ir,
                                          float beta) noexcept
{
  if (!NeedsTranscendentalDerivative(dType))
  {
    FusedStateUpdateMomentumDispatch(z, v, feedback, dType, e, n, ir, beta);
    return;
  }

#pragma omp parallel for schedule(static) if (n > 4 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    float deriv = ActivationDerivativeScalar(dType, z[i]);
    float update = (feedback[i] * deriv) - e[i];
    v[i] = beta * v[i] + (1.0f - beta) * update;
    z[i] += ir * v[i];
  }
}

void CPUBackend::FusedStateUpdate(float* z, const float* feedback, ActivationType dType,
                                  const float* e, size_t n, float ir) noexcept
{
  if (!NeedsTranscendentalDerivative(dType))
  {
    FusedStateUpdateDispatch(z, feedback, dType, e, n, ir);
    return;
  }

#pragma omp parallel for schedule(static) if (n > 4 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    float deriv = ActivationDerivativeScalar(dType, z[i]);
    z[i] += ir * ((feedback[i] * deriv) - e[i]);
  }
}

float CPUBackend::ComputeErrorAndEnergy(float* e, const float* z, const float* mu,
                                        size_t n) noexcept
{
  float energy = 0.0f;

  // See ComputeError() above for the empirical basis of this threshold.
#pragma omp parallel for reduction(+ : energy) schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    float err = z[i] - mu[i];
    e[i] = err;
    energy += err * err;
  }

  return 0.5f * energy;
}

void CPUBackend::ComputeError(float* e, const float* z, const float* mu, size_t n) noexcept
{
  // OpenMP fork/join costs a fixed ~10us on this kind of trivial
  // elementwise loop regardless of n, so parallelizing below ~100k
  // elements is a net loss (measured on this engine's own settling
  // loop: a 256x64 ComputeError call was paying ~8.5us of thread
  // overhead for ~1.6us of actual serial work).
#pragma omp parallel for schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
    e[i] = z[i] - mu[i];
}

bool CPUBackend::TryComputeErrorAndActivation(ActivationType activationType, float* e, float* zF,
                                              const float* z, const float* mu,
                                              size_t n) noexcept
{
  // Plain scalar loop, not the hand-vectorized AVX/SLEEF path
  // ActivationInto() uses -- same tradeoff FusedStateUpdate() already
  // accepts for its own transcendental-derivative cases (see
  // NeedsTranscendentalDerivative()'s comment above): this trades some
  // per-call vector throughput for reading z once instead of twice and
  // paying one fork/join instead of two, which is the actual win at the
  // layer sizes this engine settles on (see ComputeError()'s comment on
  // where that fork/join cost dominates).
#pragma omp parallel for schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    e[i] = z[i] - mu[i];
    zF[i] = ActivationScalar(activationType, z[i]);
  }
  return true;
}

float CPUBackend::ComputeSoftmaxCrossEntropyErrorAndEnergy(
    float* e, const float* z, const float* mu, size_t batchSize, size_t nextSize,
    float* /*rowEnergies, unused on CPU*/) noexcept
{
  float energy = 0.0f;
  const float eps = 1e-8f;

#pragma omp parallel for reduction(+ : energy)                                                     \
    schedule(static) if (batchSize > 8 && !omp_in_parallel())
  for (size_t b = 0; b < batchSize; ++b)
  {
    size_t base = b * nextSize;

    float max_val = mu[base];
    for (size_t j = 1; j < nextSize; ++j)
      max_val = std::max(max_val, mu[base + j]);

    float sum_exp = 0.0f;
    for (size_t j = 0; j < nextSize; ++j)
    {
      float ex = std::exp(mu[base + j] - max_val);
      e[base + j] = ex; // e temporarily holds unnormalized exp
      sum_exp += ex;
    }

    float row_energy = 0.0f;
    for (size_t j = 0; j < nextSize; ++j)
    {
      float prob = e[base + j] / sum_exp;
      e[base + j] = prob; // e now holds probs
      row_energy -= z[base + j] * std::log(prob + eps);
    }
    energy += row_energy;

    for (size_t j = 0; j < nextSize; ++j)
      e[base + j] = z[base + j] - e[base + j]; // e now holds the final error
  }

  return energy;
}

void CPUBackend::ComputeSoftmaxCrossEntropyError(float* e, const float* z, const float* mu,
                                                 size_t batchSize, size_t nextSize) noexcept
{
#pragma omp parallel for schedule(static) if (batchSize > 8 && !omp_in_parallel())
  for (size_t b = 0; b < batchSize; ++b)
  {
    size_t base = b * nextSize;

    float max_val = mu[base];
    for (size_t j = 1; j < nextSize; ++j)
      max_val = std::max(max_val, mu[base + j]);

    float sum_exp = 0.0f;
    for (size_t j = 0; j < nextSize; ++j)
    {
      float ex = std::exp(mu[base + j] - max_val);
      e[base + j] = ex;
      sum_exp += ex;
    }

    for (size_t j = 0; j < nextSize; ++j)
      e[base + j] /= sum_exp; // e now holds probs

    for (size_t j = 0; j < nextSize; ++j)
      e[base + j] = z[base + j] - e[base + j]; // e now holds the final error
  }
}

void CPUBackend::IncrementCounter(int* counter) noexcept
{
  ++(*counter);
}

void CPUBackend::AdamStep(float* param, const float* grad, float* m, float* v, size_t n,
                          const int* t, const float* lr, float beta1, float beta2,
                          float eps) noexcept
{
  // Precompute bias correction
  float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(*t));
  float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(*t));
  float step_size = *lr * Sleef_sqrtf(beta2_t) / beta1_t;

#pragma omp parallel for schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    float g = grad[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

    param[i] -= step_size * m[i] / (std::sqrt(v[i]) + eps);
  }
}

void CPUBackend::AdamWStep(float* param, const float* grad, float* m, float* v, size_t n,
                           const int* t, const float* lr, float weightDecay, float beta1,
                           float beta2, float eps) noexcept
{
  float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(*t));
  float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(*t));
  float step_size = *lr * Sleef_sqrtf(beta2_t) / beta1_t;

#pragma omp parallel for schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    float g = grad[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

    param[i] -= *lr * weightDecay * param[i];
    param[i] -= step_size * m[i] / (Sleef_sqrtf(v[i]) + eps);
  }
}

void CPUBackend::Im2Col(const float* input, int batchSize, int channels, int height, int width,
                        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                        float* columns) noexcept
{
  size_t imageSize = (size_t)channels * height * width;
  size_t colSize = (size_t)channels * kernelH * kernelW *
                   Deep::ConvOutDim(height, kernelH, strideH, padH) *
                   Deep::ConvOutDim(width, kernelW, strideW, padW);

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
  for (int batch = 0; batch < batchSize; ++batch)
  {
    Deep::Im2Col(input + (size_t)batch * imageSize,
                 channels,
                 height,
                 width,
                 kernelH,
                 kernelW,
                 strideH,
                 strideW,
                 padH,
                 padW,
                 columns + (size_t)batch * colSize);
  }
}

void CPUBackend::Col2Im(const float* columns, int batchSize, int channels, int height, int width,
                        int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                        float* outputImage) noexcept
{
  size_t imageSize = (size_t)channels * height * width;
  size_t colSize = (size_t)channels * kernelH * kernelW *
                   Deep::ConvOutDim(height, kernelH, strideH, padH) *
                   Deep::ConvOutDim(width, kernelW, strideW, padW);

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
  for (int batch = 0; batch < batchSize; ++batch)
  {
    Deep::Col2Im(columns + (size_t)batch * colSize,
                 channels,
                 height,
                 width,
                 kernelH,
                 kernelW,
                 strideH,
                 strideW,
                 padH,
                 padW,
                 outputImage + (size_t)batch * imageSize);
  }
}

void CPUBackend::MaxPool2DForward(const float* input, int batchSize, int channels, int height,
                                  int width, int poolH, int poolW, int strideH, int strideW,
                                  float* output, int* argmax) noexcept
{
  size_t imageSize = (size_t)channels * height * width;
  size_t outSize = (size_t)channels * Deep::PoolOutDim(height, poolH, strideH) *
                   Deep::PoolOutDim(width, poolW, strideW);

  for (int batch = 0; batch < batchSize; ++batch)
  {
    Deep::MaxPool2DForward(input + (size_t)batch * imageSize,
                           channels,
                           height,
                           width,
                           poolH,
                           poolW,
                           strideH,
                           strideW,
                           output + (size_t)batch * outSize,
                           argmax + (size_t)batch * outSize);
  }
}

void CPUBackend::MaxPool2DBackward(const float* outputGrad, const int* argmax, int batchSize,
                                   int channels, int height, int width, int outH, int outW,
                                   float* inputGrad) noexcept
{
  size_t imageSize = (size_t)channels * height * width;
  size_t outSize = (size_t)channels * outH * outW;

  for (int batch = 0; batch < batchSize; ++batch)
  {
    Deep::MaxPool2DBackward(outputGrad + (size_t)batch * outSize,
                            argmax + (size_t)batch * outSize,
                            channels,
                            height,
                            width,
                            outH,
                            outW,
                            inputGrad + (size_t)batch * imageSize);
  }
}

void CPUBackend::RepackForBatchedGemm(float* dst, const float* src, size_t batchSize, size_t rows,
                                      size_t cols) noexcept
{
  // dst[row][batch][:] = src[batch][row][:], matches
  // SimpleConvPCLayer::UpdateWeights()'s colsRepacked/lgRepacked loops
  // exactly (same index arithmetic, same collapse(2) parallelization).
  const int maxRows = static_cast<int>(rows);
  const int maxBatch = static_cast<int>(batchSize);

#pragma omp parallel for schedule(static)                                                          \
    collapse(2) if (rows * batchSize * cols > 65536 && !omp_in_parallel())
  for (int row = 0; row < maxRows; ++row)
  {
    for (int batch = 0; batch < maxBatch; ++batch)
    {
      size_t u_row = static_cast<size_t>(row);
      size_t u_batch = static_cast<size_t>(batch);

      const float* s = src + u_batch * rows * cols + u_row * cols;
      float* d = dst + u_row * batchSize * cols + u_batch * cols;
      std::memcpy(d, s, cols * sizeof(float));
    }
  }
}

void CPUBackend::MultiplyInto(float* dst, const float* a, const float* b, size_t n) noexcept
{
  MultiplyIntoDispatch(dst, a, b, n);
}

void CPUBackend::FusedActivationDerivativeMultiply(float* dst, const float* a,
                                                   float* activatedInOut, ActivationType dType,
                                                   size_t n) noexcept
{
  // dGELU needs Sleef_tanhf_u10 (see ActivationDerivativeFromActivatedScalar's
  // own documented GELU caveat) -- not in scope for the Highway pass yet,
  // stays on the original scalar path. Every other type is closed-form in
  // the activated value alone and is Highway-dispatched (SIMD, runtime
  // ISA-selected) instead, see CPUBackendHighway.cpp.
  if (dType == ActivationType::dGELU)
  {
    const ptrdiff_t maxN = static_cast<ptrdiff_t>(n);
#pragma omp parallel for schedule(static) if (n > 65536 && !omp_in_parallel())
    for (ptrdiff_t i = 0; i < maxN; ++i)
    {
      float deriv = ActivationDerivativeFromActivatedScalar(dType, activatedInOut[i]);
      dst[i] = a[i] * deriv;
      activatedInOut[i] = deriv;
    }
    return;
  }

  FusedActivationDerivativeMultiplyDispatch(dst, a, activatedInOut, dType, n);
}

void CPUBackend::FusedActivationDerivativeMultiplyFromPreact(float* dst, const float* a,
                                                              const float* preact,
                                                              ActivationType dType, size_t n) noexcept
{
  // Unlike FusedActivationDerivativeMultiply above, no scalar carve-out:
  // the Highway kernel handles every ActivationType, including
  // dGELU/dSIGMOID/dTANH, via hwy/contrib/math's vectorized Exp/Tanh --
  // see CPUBackendHighway.cpp's file comment.
  FusedActivationDerivativeMultiplyFromPreactDispatch(dst, a, preact, dType, n);
}

void CPUBackend::Fill(float* buf, size_t n, float value) noexcept
{
  FillDispatch(buf, n, value);
}

void CPUBackend::AddBiasPerChannel(float* buf, const float* bias, size_t channels,
                                   size_t spatialSize) noexcept
{
  AddBiasPerChannelDispatch(buf, bias, channels, spatialSize);
}

void CPUBackend::AddBiasPerChannelNCHW(float* buf, const float* bias, size_t batchSize,
                                       size_t channels, size_t spatialSize) noexcept
{
  // Reuses the existing, already-verified per-item kernel once per batch
  // slice -- same "call once per item with buf offset to that item's
  // slice" convention FullConvPCLayer::DirectFeedbackUpdate() already
  // established for this exact situation (a buffer this primitive can't
  // address in one call because channel isn't its outermost dimension).
  // Cheap on CPU (plain host loop); CUDABackend's own override instead
  // uses a single kernel launch, where per-item loop overhead is the
  // problem TryFusedConvForward() exists to eliminate elsewhere.
  for (size_t b = 0; b < batchSize; ++b)
    AddBiasPerChannelDispatch(buf + b * channels * spatialSize, bias, channels, spatialSize);
}

float CPUBackend::ComputePrecisionWeightedErrorAndEnergy(float* e, const float* z, const float* mu,
                                                         const float* p, size_t batchSize,
                                                         size_t width) noexcept
{
  float energy = 0.0f;
  const ptrdiff_t maxBatch = static_cast<ptrdiff_t>(batchSize);
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static) reduction(+ : energy)                                    \
    collapse(2) if (batchSize * width > 256 && !omp_in_parallel())
  for (ptrdiff_t b = 0; b < maxBatch; ++b)
  {
    for (ptrdiff_t i = 0; i < maxWidth; ++i)
    {
      size_t idx = (size_t)b * width + (size_t)i;
      float err = z[idx] - mu[idx];
      e[idx] = err;
      float precision = std::max(p[i], 1e-8f);
      energy += 0.5f * precision * err * err - 0.5f * std::log(precision);
    }
  }
  return energy;
}

void CPUBackend::ComputePrecisionWeightedError(float* e, const float* z, const float* mu,
                                               const float* /*p*/, size_t batchSize,
                                               size_t width) noexcept
{
  const size_t n = batchSize * width;
#pragma omp parallel for schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t idx = 0; idx < n; ++idx)
    e[idx] = z[idx] - mu[idx];
}

void CPUBackend::AxpyBroadcastInto(float* y, const float* x, const float* factor, size_t batchSize,
                                   size_t width, float alpha) noexcept
{
  const ptrdiff_t maxBatch = static_cast<ptrdiff_t>(batchSize);
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static)                                                          \
    collapse(2) if (batchSize * width > 65536 && !omp_in_parallel())
  for (ptrdiff_t b = 0; b < maxBatch; ++b)
  {
    for (ptrdiff_t i = 0; i < maxWidth; ++i)
    {
      size_t idx = (size_t)b * width + (size_t)i;
      y[idx] += alpha * x[idx] * factor[i];
    }
  }
}

void CPUBackend::MultiplyBroadcastInto(float* dst, const float* a, const float* factor,
                                       const float* b, size_t batchSize, size_t width) noexcept
{
  const ptrdiff_t maxBatch = static_cast<ptrdiff_t>(batchSize);
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static)                                                          \
    collapse(2) if (batchSize * width > 65536 && !omp_in_parallel())
  for (ptrdiff_t batch = 0; batch < maxBatch; ++batch)
  {
    for (ptrdiff_t i = 0; i < maxWidth; ++i)
    {
      size_t idx = (size_t)batch * width + (size_t)i;
      dst[idx] = a[idx] * factor[i] * b[idx];
    }
  }
}

void CPUBackend::UpdatePrecisionFromError(float* p, float* log_p, const float* e, size_t batchSize,
                                          size_t width, float pr) noexcept
{
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static) if (width > 4096 && !omp_in_parallel())
  for (ptrdiff_t i = 0; i < maxWidth; ++i)
  {
    float grad = 0.0f;
    for (size_t b = 0; b < batchSize; ++b)
    {
      float err = e[b * width + (size_t)i];
      grad += 0.5f * (p[i] * err * err - 1.0f);
    }
    grad /= (float)batchSize;
    log_p[i] -= pr * grad;
    log_p[i] = std::max(-5.0f, std::min(log_p[i], 5.0f));
    p[i] = std::exp(log_p[i]);
  }
}
} // namespace Deep
