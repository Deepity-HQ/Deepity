#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <deepity/backend/CPUBackendT.h>
#include <deepity/utils/MemoryArena.h> // Deep::detail::Aligned{Alloc,Free}Portable
#include <deepity/utils/QuantFormat.h>
#include <hwy/base.h> // hwy::ConvertScalarTo, see this file's generic decode/encode casts below
#include <random>

#include "CPUBackendHighway.h"

#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif

#include <sleef.h>

namespace Deep
{

namespace
{
// Only dRELU/dSIGMOID/dTANH/dGELU/d_eSIGMOID/dHARD_TANH/dLINEAR/NONE
// exist; FusedStateUpdate{,Momentum} themselves are out of scope for
// CPUBackendT (see its header's own doc comment on why), so there's no
// NeedsTranscendentalDerivative counterpart needed here.

// Shared by the MatMul family's hwy::float16_t branches below: no
// native half-GEMM in OpenBLAS/MKL, so every f16 GEMM operand is
// decoded into a float scratch, run through the existing sgemm path,
// and (only where the destination is T itself, not AccumT) encoded
// back.
void DecodeF16Rows(const hwy::float16_t* src, float* dst, size_t count) noexcept
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = hwy::ConvertScalarTo<float>(src[i]);
}

void EncodeF16Rows(const float* src, hwy::float16_t* dst, size_t count) noexcept
{
  for (size_t i = 0; i < count; ++i)
    dst[i] = hwy::ConvertScalarTo<hwy::float16_t>(src[i]);
}
} // namespace

template <typename T, typename AccumT>
T* CPUBackendT<T, AccumT>::Allocate(size_t numElements)
{
  return static_cast<T*>(detail::AlignedAllocPortable(16, numElements * sizeof(T)));
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Free(T* ptr) noexcept
{
  detail::AlignedFreePortable(ptr);
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Zero(T* ptr, size_t numElements) noexcept
{
  memset(ptr, 0, numElements * sizeof(T));
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Copy(T* dst, const T* src, size_t numElements) noexcept
{
  memcpy(dst, src, numElements * sizeof(T));
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::CopyFromHost(T* deviceDst, const T* hostSrc,
                                          size_t numElements) noexcept
{
  memcpy(deviceDst, hostSrc, numElements * sizeof(T));
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::CopyToHost(T* hostDst, const T* deviceSrc,
                                        size_t numElements) noexcept
{
  memcpy(hostDst, deviceSrc, numElements * sizeof(T));
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::RandomizeNormal(T* buf, size_t n, AccumT mean, AccumT stddev,
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
    std::normal_distribution<AccumT> dist(mean, stddev);

#pragma omp for
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i)
      buf[i] = hwy::ConvertScalarTo<T>(dist(rng));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::RandomizeUniform(T* buf, size_t n, AccumT min, AccumT max,
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
    std::uniform_real_distribution<AccumT> dist(min, max);

#pragma omp for
    for (ptrdiff_t i = 0; i < (ptrdiff_t)n; ++i)
      buf[i] = hwy::ConvertScalarTo<T>(dist(rng));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MatMul(bool transA, bool transB, int M, int N, int K, AccumT alpha,
                                    const T* A, int lda, const T* B, int ldb, AccumT beta, T* C,
                                    int ldc) noexcept
{
  if constexpr (std::is_same_v<T, float>)
  {
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, (float)alpha, A, lda, B, ldb,
                (float)beta, C, ldc);
  }
  else if constexpr (std::is_same_v<T, double>)
  {
    cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, (double)alpha, A, lda, B, ldb,
                (double)beta, C, ldc);
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    // No native half-GEMM in OpenBLAS/MKL: decode A, B, and (for the
    // beta term) C into float scratch, reuse cblas_sgemm, encode the
    // result back.
    size_t rowsA = static_cast<size_t>(transA ? K : M);
    size_t rowsB = static_cast<size_t>(transB ? N : K);
    std::vector<float> Af(rowsA * lda), Bf(rowsB * ldb), Cf(static_cast<size_t>(M) * ldc);
    DecodeF16Rows(A, Af.data(), Af.size());
    DecodeF16Rows(B, Bf.data(), Bf.size());
    DecodeF16Rows(C, Cf.data(), Cf.size());
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, Af.data(), lda, Bf.data(),
                ldb, beta, Cf.data(), ldc);
    EncodeF16Rows(Cf.data(), C, Cf.size());
  }
  else
  {
    // No native BLAS for this T (sub-8-bit storage formats): a plain,
    // correct, non-vectorized triple loop. Decode/encode-wrap through a
    // float scratch GEMM is the eventual real path for these formats
    // (see the quantization plan's "Bucket C" design), not attempted
    // here since no sub-8-bit T is instantiated yet.
    for (int i = 0; i < M; ++i)
    {
      for (int j = 0; j < N; ++j)
      {
        AccumT acc(0);
        for (int k = 0; k < K; ++k)
        {
          AccumT a = transA ? hwy::ConvertScalarTo<AccumT>(A[k * lda + i]) : hwy::ConvertScalarTo<AccumT>(A[i * lda + k]);
          AccumT b = transB ? hwy::ConvertScalarTo<AccumT>(B[j * ldb + k]) : hwy::ConvertScalarTo<AccumT>(B[k * ldb + j]);
          acc += a * b;
        }
        T& c = C[i * ldc + j];
        c = hwy::ConvertScalarTo<T>(alpha * acc + beta * hwy::ConvertScalarTo<AccumT>(c));
      }
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MatMulAccum(bool transA, bool transB, int M, int N, int K,
                                         AccumT alpha, const T* A, int lda, const T* B, int ldb,
                                         AccumT beta, AccumT* C, int ldc) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    // C is already AccumT=float, so unlike MatMul() above there's no
    // encode-back step: only A and B need decoding.
    size_t rowsA = static_cast<size_t>(transA ? K : M);
    size_t rowsB = static_cast<size_t>(transB ? N : K);
    std::vector<float> Af(rowsA * lda), Bf(rowsB * ldb);
    DecodeF16Rows(A, Af.data(), Af.size());
    DecodeF16Rows(B, Bf.data(), Bf.size());
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, Af.data(), lda, Bf.data(),
                ldb, beta, C, ldc);
  }
  else
  {
    // No decode/encode round trip here (unlike MatMul() above): C is
    // already AccumT*, so the product accumulates directly at full
    // AccumT precision with no intermediate rounding to T at all --
    // exactly the point of this method existing separately from
    // MatMul(), see this method's own declaration doc comment.
    for (int i = 0; i < M; ++i)
    {
      for (int j = 0; j < N; ++j)
      {
        AccumT acc(0);
        for (int k = 0; k < K; ++k)
        {
          AccumT a = transA ? hwy::ConvertScalarTo<AccumT>(A[k * lda + i]) : hwy::ConvertScalarTo<AccumT>(A[i * lda + k]);
          AccumT b = transB ? hwy::ConvertScalarTo<AccumT>(B[j * ldb + k]) : hwy::ConvertScalarTo<AccumT>(B[k * ldb + j]);
          acc += a * b;
        }
        AccumT& c = C[i * ldc + j];
        c = alpha * acc + beta * c;
      }
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MatMulAccumMixed(bool transA, bool transB, int M, int N, int K,
                                              AccumT alpha, const AccumT* A, int lda, const T* B,
                                              int ldb, AccumT beta, AccumT* C, int ldc) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    // Bit-exact with MatMul()'s own float/float path: A, B, and C are
    // all float* here too (AccumT==T==float).
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    cblas_dgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
    return;
  }

  if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    // A and C are already AccumT=float; only B (the stored T-typed
    // parameter W) needs the decode-to-scratch/sgemm round trip.
    size_t rowsB = static_cast<size_t>(transB ? N : K);
    std::vector<float> Bf(rowsB * ldb);
    DecodeF16Rows(B, Bf.data(), Bf.size());
    cblas_sgemm(CblasRowMajor, transA ? CblasTrans : CblasNoTrans,
                transB ? CblasTrans : CblasNoTrans, M, N, K, alpha, A, lda, Bf.data(), ldb, beta,
                C, ldc);
    return;
  }

  // A is already AccumT (a settling/gradient scratch, see this method's
  // own doc comment) -- only B, the real stored T-typed parameter (W),
  // needs decoding, for every other (T, AccumT) pair.
  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      AccumT acc(0);
      for (int k = 0; k < K; ++k)
      {
        AccumT a = transA ? A[k * lda + i] : A[i * lda + k];
        AccumT b = transB ? hwy::ConvertScalarTo<AccumT>(B[j * ldb + k]) : hwy::ConvertScalarTo<AccumT>(B[k * ldb + j]);
        acc += a * b;
      }
      AccumT& c = C[i * ldc + j];
      c = alpha * acc + beta * c;
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::SumRowsAccum(AccumT* dst, const T* src, size_t batchSize,
                                          size_t width) noexcept
{
  memset(dst, 0, width * sizeof(AccumT));

  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    for (size_t b = 0; b < batchSize; ++b)
      cblas_saxpy((int)width, 1.0f, src + b * width, 1, dst, 1);
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    for (size_t b = 0; b < batchSize; ++b)
      cblas_daxpy((int)width, 1.0, src + b * width, 1, dst, 1);
  }
  else
  {
    for (size_t b = 0; b < batchSize; ++b)
      for (size_t i = 0; i < width; ++i)
        dst[i] += hwy::ConvertScalarTo<AccumT>(src[b * width + i]);
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::SumRows(T* dst, const T* src, size_t batchSize,
                                     size_t width) noexcept
{
  memset(dst, 0, width * sizeof(T));

  // Bit-exactness against the non-templated CPUBackend for T=float
  // requires reusing cblas_saxpy itself, not a mathematically-equivalent
  // hand-rolled accumulation loop: BLAS's internal summation order
  // (e.g. SIMD-lane-parallel partial sums, then a horizontal reduce)
  // rounds differently than a straight sequential scalar accumulation,
  // even though both compute "the same sum" in exact arithmetic.
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    for (size_t b = 0; b < batchSize; ++b)
      cblas_saxpy((int)width, 1.0f, src + b * width, 1, dst, 1);
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    for (size_t b = 0; b < batchSize; ++b)
      cblas_daxpy((int)width, 1.0, src + b * width, 1, dst, 1);
  }
  else
  {
    for (size_t b = 0; b < batchSize; ++b)
      for (size_t i = 0; i < width; ++i)
        dst[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(dst[i]) + hwy::ConvertScalarTo<AccumT>(src[b * width + i]));
  }
}

template <typename T, typename AccumT>
AccumT CPUBackendT<T, AccumT>::Sum(const T* buf, size_t n) noexcept
{
  AccumT total(0);
#pragma omp parallel for schedule(static) reduction(+ : total) if (n > 4096 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
    total += hwy::ConvertScalarTo<AccumT>(buf[i]);
  return total;
}

// Scale/AxpyInto/AddBiasBroadcast dispatch to cblas_s/dscal/saxpy for
// T=float/double, matching the non-templated CPUBackend exactly, rather
// than a mathematically-equivalent hand-rolled loop -- see SumRows'
// own comment above on why that distinction matters for bit-exactness.
template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Scale(T* buf, size_t n, AccumT alpha) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    cblas_sscal((int)n, alpha, buf, 1);
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    cblas_dscal((int)n, alpha, buf, 1);
  }
  else
  {
    for (size_t i = 0; i < n; ++i)
      buf[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(buf[i]) * alpha);
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AxpyInto(T* y, const T* x, size_t n, AccumT alpha) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    cblas_saxpy((int)n, alpha, x, 1, y, 1);
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    cblas_daxpy((int)n, alpha, x, 1, y, 1);
  }
  else
  {
    for (size_t i = 0; i < n; ++i)
      y[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(y[i]) + alpha * hwy::ConvertScalarTo<AccumT>(x[i]));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AxpyIntoAccum(AccumT* y, const T* x, size_t n, AccumT alpha) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    cblas_saxpy((int)n, alpha, x, 1, y, 1);
    return;
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    cblas_daxpy((int)n, alpha, x, 1, y, 1);
    return;
  }

  for (size_t i = 0; i < n; ++i)
    y[i] += alpha * hwy::ConvertScalarTo<AccumT>(x[i]);
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AxpyIntoFromAccum(T* y, const AccumT* x, size_t n,
                                               AccumT alpha) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    cblas_saxpy((int)n, alpha, x, 1, y, 1);
    return;
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
    cblas_daxpy((int)n, alpha, x, 1, y, 1);
    return;
  }

  for (size_t i = 0; i < n; ++i)
    y[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(y[i]) + alpha * x[i]);
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AddBiasBroadcast(T* buf, const T* bias, size_t batchSize,
                                              size_t width) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
    for (size_t b = 0; b < batchSize; ++b)
      cblas_saxpy((int)width, 1.0f, bias, 1, buf + b * width, 1);
  }
  else if constexpr (std::is_same_v<T, double> && std::is_same_v<AccumT, double>)
  {
#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
    for (size_t b = 0; b < batchSize; ++b)
      cblas_daxpy((int)width, 1.0, bias, 1, buf + b * width, 1);
  }
  else
  {
#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
    for (size_t b = 0; b < batchSize; ++b)
      for (size_t i = 0; i < width; ++i)
        buf[b * width + i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(buf[b * width + i]) + hwy::ConvertScalarTo<AccumT>(bias[i]));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Activation(ActivationType type, T* buf, size_t n) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    To_Fn(type)(buf, n);
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    // Genuine Highway lane-level compute (issue #50), not the generic
    // scalar wrap below -- see CPUBackendHighway.h's own top comment.
    ActivationF16Dispatch(type, buf, n);
  }
  else
  {
    // Decode each T to AccumT, run the existing generic scalar kernel
    // there, re-encode back to T -- the "Bucket C" decode/compute/
    // encode wrap the quantization plan calls for, so a storage T with
    // no arithmetic of its own (e.g. a sub-8-bit grid type) never needs
    // operators of its own: it only has to convert to/from AccumT (see
    // QuantFormat.h). Bit-identical to the old T-only path when T==AccumT
    // (e.g. double), since hwy::ConvertScalarTo<AccumT>(x) and hwy::ConvertScalarTo<T>(x) are then both the identity.
    for (size_t i = 0; i < n; ++i)
      buf[i] = hwy::ConvertScalarTo<T>(ActivationScalarT<AccumT>(type, hwy::ConvertScalarTo<AccumT>(buf[i])));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::ActivationInto(ActivationType type, T* dst, const T* src,
                                            size_t n) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    cblas_scopy((int)n, src, 1, dst, 1);
    To_Fn(type)(dst, n);
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    ActivationIntoF16Dispatch(type, dst, src, n);
  }
  else
  {
    for (size_t i = 0; i < n; ++i)
      dst[i] = hwy::ConvertScalarTo<T>(ActivationScalarT<AccumT>(type, hwy::ConvertScalarTo<AccumT>(src[i])));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::ActivationDerivative(ActivationType type, T* buf, size_t n,
                                                  bool activated) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    To_dFn(type)(buf, n, activated);
  }
  else
  {
    // Matches To_dFn's own contract: `activated` selects the
    // closed-form-from-activated-value path where one exists. Same
    // decode/compute/encode-via-AccumT wrap as Activation() above.
    for (size_t i = 0; i < n; ++i)
      buf[i] = activated
                   ? hwy::ConvertScalarTo<T>(ActivationDerivativeFromActivatedScalar(type, hwy::ConvertScalarTo<AccumT>(buf[i])))
                   : hwy::ConvertScalarTo<T>(ActivationDerivativeScalarT<AccumT>(type, hwy::ConvertScalarTo<AccumT>(buf[i])));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::ActivationDerivativeInto(ActivationType type, T* dst, const T* src,
                                                      size_t n) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    To_dFn2(type)(dst, src, n);
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    ActivationDerivativeIntoF16Dispatch(type, dst, src, n);
  }
  else
  {
    for (size_t i = 0; i < n; ++i)
      dst[i] = hwy::ConvertScalarTo<T>(ActivationDerivativeScalarT<AccumT>(type, hwy::ConvertScalarTo<AccumT>(src[i])));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::ActivationDerivativeIntoAccum(ActivationType type, AccumT* dst,
                                                           const T* src, size_t n) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    // Bit-exact with ActivationDerivativeInto()'s own float/float path:
    // dst and src are both float* here too (AccumT==T==float), so
    // there's no reason to fall back to the generic scalar formula and
    // diverge from the SIMD kernel by a ULP or two.
    To_dFn2(type)(dst, src, n);
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    ActivationDerivativeIntoAccumF16Dispatch(type, dst, src, n);
  }
  else
  {
    for (size_t i = 0; i < n; ++i)
      dst[i] = ActivationDerivativeScalarT<AccumT>(type, hwy::ConvertScalarTo<AccumT>(src[i]));
  }
}

template <typename T, typename AccumT>
AccumT CPUBackendT<T, AccumT>::ComputeErrorAndEnergy(T* RESTRICT e, const T* RESTRICT z,
                                                     const T* RESTRICT mu, size_t n) noexcept
{
  AccumT energy(0);
#pragma omp parallel for reduction(+ : energy) schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    AccumT err = hwy::ConvertScalarTo<AccumT>(z[i]) - hwy::ConvertScalarTo<AccumT>(mu[i]);
    e[i] = hwy::ConvertScalarTo<T>(err);
    energy += err * err;
  }
  return hwy::ConvertScalarTo<AccumT>(0.5) * energy;
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::ComputeError(T* RESTRICT e, const T* RESTRICT z, const T* RESTRICT mu,
                                          size_t n) noexcept
{
#pragma omp parallel for schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
    e[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(z[i]) - hwy::ConvertScalarTo<AccumT>(mu[i]));
}

template <typename T, typename AccumT>
AccumT CPUBackendT<T, AccumT>::ComputeSoftmaxCrossEntropyErrorAndEnergy(
    T* e, const T* z, const T* mu, size_t batchSize, size_t nextSize,
    AccumT* /*rowEnergies, unused on CPU*/) noexcept
{
  AccumT energy(0);
  const AccumT eps(1e-8);

#pragma omp parallel for reduction(+ : energy)                                                     \
    schedule(static) if (batchSize > 8 && !omp_in_parallel())
  for (size_t b = 0; b < batchSize; ++b)
  {
    size_t base = b * nextSize;

    AccumT max_val = hwy::ConvertScalarTo<AccumT>(mu[base]);
    for (size_t j = 1; j < nextSize; ++j)
      max_val = std::max(max_val, hwy::ConvertScalarTo<AccumT>(mu[base + j]));

    AccumT sum_exp(0);
    for (size_t j = 0; j < nextSize; ++j)
    {
      AccumT ex = std::exp(hwy::ConvertScalarTo<AccumT>(mu[base + j]) - max_val);
      e[base + j] = hwy::ConvertScalarTo<T>(ex); // e temporarily holds unnormalized exp
      sum_exp += ex;
    }

    AccumT row_energy(0);
    for (size_t j = 0; j < nextSize; ++j)
    {
      AccumT prob = hwy::ConvertScalarTo<AccumT>(e[base + j]) / sum_exp;
      e[base + j] = hwy::ConvertScalarTo<T>(prob); // e now holds probs
      row_energy -= hwy::ConvertScalarTo<AccumT>(z[base + j]) * std::log(prob + eps);
    }
    energy += row_energy;

    for (size_t j = 0; j < nextSize; ++j)
      e[base + j] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(z[base + j]) - hwy::ConvertScalarTo<AccumT>(e[base + j])); // e now holds the final error
  }

  return energy;
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::ComputeSoftmaxCrossEntropyError(T* e, const T* z, const T* mu,
                                                             size_t batchSize,
                                                             size_t nextSize) noexcept
{
  if constexpr (std::is_same_v<T, float>)
  {
#pragma omp parallel for schedule(static) if (batchSize > 8 && !omp_in_parallel())
    for (size_t b = 0; b < batchSize; ++b)
    {
      size_t base = b * nextSize;
      ComputeSoftmaxCrossEntropyErrorRowDispatch(e + base, z + base, mu + base, nextSize);
    }
  }
  else
  {
#pragma omp parallel for schedule(static) if (batchSize > 8 && !omp_in_parallel())
    for (size_t b = 0; b < batchSize; ++b)
    {
      size_t base = b * nextSize;

      AccumT max_val = hwy::ConvertScalarTo<AccumT>(mu[base]);
      for (size_t j = 1; j < nextSize; ++j)
        max_val = std::max(max_val, hwy::ConvertScalarTo<AccumT>(mu[base + j]));

      AccumT sum_exp(0);
      for (size_t j = 0; j < nextSize; ++j)
      {
        AccumT ex = std::exp(hwy::ConvertScalarTo<AccumT>(mu[base + j]) - max_val);
        e[base + j] = hwy::ConvertScalarTo<T>(ex);
        sum_exp += ex;
      }
      for (size_t j = 0; j < nextSize; ++j)
        e[base + j] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(z[base + j]) - hwy::ConvertScalarTo<AccumT>(e[base + j]) / sum_exp);
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::IncrementCounter(int* counter) noexcept
{
  ++(*counter);
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AdamStep(T* param, const AccumT* grad, AccumT* m, AccumT* v, size_t n,
                                      const int* t, const AccumT* lr, AccumT beta1, AccumT beta2,
                                      AccumT eps) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    // Bit-exact reuse of the existing Highway-vectorized kernel, same
    // OMP chunking as the non-templated CPUBackend::AdamStep().
    float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(*t));
    float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(*t));
    float step_size = *lr * Sleef_sqrtf(beta2_t) / beta1_t;

    if (n > 100000 && !omp_in_parallel())
    {
#pragma omp parallel
      {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        size_t start = n * (size_t)tid / nt, end = n * (size_t)(tid + 1) / nt;
        AdamStepChunkDispatch(param, grad, m, v, start, end, step_size, beta1, beta2, eps);
      }
    }
    else
    {
      AdamStepChunkDispatch(param, grad, m, v, 0, n, step_size, beta1, beta2, eps);
    }
    return;
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    float beta1_t = 1.0f - std::pow(beta1, static_cast<float>(*t));
    float beta2_t = 1.0f - std::pow(beta2, static_cast<float>(*t));
    float step_size = *lr * std::sqrt(beta2_t) / beta1_t;

    if (n > 100000 && !omp_in_parallel())
    {
#pragma omp parallel
      {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        size_t start = n * (size_t)tid / nt, end = n * (size_t)(tid + 1) / nt;
        AdamStepF16Dispatch(param, grad, m, v, start, end, step_size, beta1, beta2, eps);
      }
    }
    else
    {
      AdamStepF16Dispatch(param, grad, m, v, 0, n, step_size, beta1, beta2, eps);
    }
    return;
  }

  AccumT beta1_t = hwy::ConvertScalarTo<AccumT>(1) - std::pow(beta1, hwy::ConvertScalarTo<AccumT>(*t));
  AccumT beta2_t = hwy::ConvertScalarTo<AccumT>(1) - std::pow(beta2, hwy::ConvertScalarTo<AccumT>(*t));
  AccumT step_size = *lr * std::sqrt(beta2_t) / beta1_t;

  for (size_t i = 0; i < n; ++i)
  {
    m[i] = beta1 * m[i] + (hwy::ConvertScalarTo<AccumT>(1) - beta1) * grad[i];
    v[i] = beta2 * v[i] + (hwy::ConvertScalarTo<AccumT>(1) - beta2) * grad[i] * grad[i];
    param[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(param[i]) - step_size * m[i] / (std::sqrt(v[i]) + eps));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AdamWStep(T* param, const AccumT* grad, AccumT* m, AccumT* v,
                                       size_t n, const int* t, const AccumT* lr,
                                       AccumT weightDecay, AccumT beta1, AccumT beta2,
                                       AccumT eps) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(*t));
    float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(*t));
    float step_size = *lr * Sleef_sqrtf(beta2_t) / beta1_t;
    float lrWeightDecay = *lr * weightDecay;

    if (n > 100000 && !omp_in_parallel())
    {
#pragma omp parallel
      {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        size_t start = n * (size_t)tid / nt, end = n * (size_t)(tid + 1) / nt;
        AdamWStepChunkDispatch(param, grad, m, v, start, end, step_size, beta1, beta2, eps,
                               lrWeightDecay);
      }
    }
    else
    {
      AdamWStepChunkDispatch(param, grad, m, v, 0, n, step_size, beta1, beta2, eps, lrWeightDecay);
    }
    return;
  }
  else if constexpr (std::is_same_v<T, hwy::float16_t> && std::is_same_v<AccumT, float>)
  {
    float beta1_t = 1.0f - std::pow(beta1, static_cast<float>(*t));
    float beta2_t = 1.0f - std::pow(beta2, static_cast<float>(*t));
    float step_size = *lr * std::sqrt(beta2_t) / beta1_t;
    float lrWeightDecay = *lr * weightDecay;

    if (n > 100000 && !omp_in_parallel())
    {
#pragma omp parallel
      {
        int nt = omp_get_num_threads(), tid = omp_get_thread_num();
        size_t start = n * (size_t)tid / nt, end = n * (size_t)(tid + 1) / nt;
        AdamWStepF16Dispatch(param, grad, m, v, start, end, step_size, beta1, beta2, eps,
                             lrWeightDecay);
      }
    }
    else
    {
      AdamWStepF16Dispatch(param, grad, m, v, 0, n, step_size, beta1, beta2, eps, lrWeightDecay);
    }
    return;
  }

  AccumT beta1_t = hwy::ConvertScalarTo<AccumT>(1) - std::pow(beta1, hwy::ConvertScalarTo<AccumT>(*t));
  AccumT beta2_t = hwy::ConvertScalarTo<AccumT>(1) - std::pow(beta2, hwy::ConvertScalarTo<AccumT>(*t));
  AccumT step_size = *lr * std::sqrt(beta2_t) / beta1_t;
  AccumT lrWeightDecay = *lr * weightDecay;

  for (size_t i = 0; i < n; ++i)
  {
    AccumT p = hwy::ConvertScalarTo<AccumT>(param[i]) - lrWeightDecay * hwy::ConvertScalarTo<AccumT>(param[i]);
    m[i] = beta1 * m[i] + (hwy::ConvertScalarTo<AccumT>(1) - beta1) * grad[i];
    v[i] = beta2 * v[i] + (hwy::ConvertScalarTo<AccumT>(1) - beta2) * grad[i] * grad[i];
    param[i] = hwy::ConvertScalarTo<T>(p - step_size * m[i] / (std::sqrt(v[i]) + eps));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MultiplyInto(T* dst, const T* a, const T* b, size_t n) noexcept
{
  if constexpr (std::is_same_v<T, float>)
  {
    MultiplyIntoDispatch(dst, a, b, n);
  }
  else
  {
    for (size_t i = 0; i < n; ++i)
      dst[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(a[i]) * hwy::ConvertScalarTo<AccumT>(b[i]));
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MultiplyIntoAccum(AccumT* dst, const T* a, const AccumT* b,
                                               size_t n) noexcept
{
  if constexpr (std::is_same_v<T, float> && std::is_same_v<AccumT, float>)
  {
    // Bit-exact with MultiplyInto()'s own float path: dst, a, and b are
    // all float* here too (AccumT==T==float).
    MultiplyIntoDispatch(dst, a, b, n);
    return;
  }

  for (size_t i = 0; i < n; ++i)
    dst[i] = hwy::ConvertScalarTo<AccumT>(a[i]) * b[i];
}

// --- Conv-family primitives: Im2Col/Col2Im/RepackForBatchedGemm are
// pure data movement (a copy, or a scatter-accumulate of copies), with
// no existing BLAS/Highway-dispatched equivalent to call into for any
// T -- so unlike every method above, there's no float/float fast path
// vs. generic-T split here, just one scalar loop whose per-element
// op order exactly matches the original float/AVX-intrinsic version's
// own order, which is what makes it bit-exact for T=float too (an
// elementwise copy or add can't be reordered into a different result,
// vectorized or not). Fill/AddBiasPerChannel(NCHW) are genuine
// arithmetic (an assign, or an add), done through the usual decode-to-
// AccumT/encode-to-T idiom. ---

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Im2Col(const T* input, int batchSize, int channels, int height,
                                    int width, int kernelH, int kernelW, int strideH, int strideW,
                                    int padH, int padW, T* columns) noexcept
{
  const int outH = (height + 2 * padH - kernelH) / strideH + 1;
  const int outW = (width + 2 * padW - kernelW) / strideW + 1;
  const size_t imageSize = (size_t)channels * height * width;
  const size_t colSize = (size_t)channels * kernelH * kernelW * outH * outW;

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
  for (int batch = 0; batch < batchSize; ++batch)
  {
    const T* in = input + (size_t)batch * imageSize;
    T* col = columns + (size_t)batch * colSize;

    for (int c = 0; c < channels; ++c)
    {
      for (int kh = 0; kh < kernelH; ++kh)
      {
        for (int kw = 0; kw < kernelW; ++kw)
        {
          const int rowIdx = (c * kernelH + kh) * kernelW + kw;
          T* destRow = col + (size_t)rowIdx * outH * outW;

          for (int oh = 0; oh < outH; ++oh)
          {
            const int inRow = oh * strideH - padH + kh;
            T* dst = destRow + (size_t)oh * outW;

            if (inRow < 0 || inRow >= height)
            {
              for (int ow = 0; ow < outW; ++ow)
                dst[ow] = hwy::ConvertScalarTo<T>(AccumT(0));
              continue;
            }

            const T* src = in + ((size_t)c * height + inRow) * width;

            for (int ow = 0; ow < outW; ++ow)
            {
              const int inCol = ow * strideW - padW + kw;
              dst[ow] = (inCol < 0 || inCol >= width) ? hwy::ConvertScalarTo<T>(AccumT(0))
                                                      : src[inCol];
            }
          }
        }
      }
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Col2Im(const T* columns, int batchSize, int channels, int height,
                                    int width, int kernelH, int kernelW, int strideH, int strideW,
                                    int padH, int padW, T* outputImage) noexcept
{
  const int outH = (height + 2 * padH - kernelH) / strideH + 1;
  const int outW = (width + 2 * padW - kernelW) / strideW + 1;
  const size_t imageSize = (size_t)channels * height * width;
  const size_t colSize = (size_t)channels * kernelH * kernelW * outH * outW;

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
  for (int batch = 0; batch < batchSize; ++batch)
  {
    const T* col = columns + (size_t)batch * colSize;
    T* out = outputImage + (size_t)batch * imageSize;

    for (int c = 0; c < channels; ++c)
    {
      for (int kh = 0; kh < kernelH; ++kh)
      {
        for (int kw = 0; kw < kernelW; ++kw)
        {
          const int rowIdx = (c * kernelH + kh) * kernelW + kw;
          const T* srcRow = col + (size_t)rowIdx * outH * outW;

          for (int oh = 0; oh < outH; ++oh)
          {
            const int inRow = oh * strideH - padH + kh;
            if (inRow < 0 || inRow >= height)
              continue;

            T* dst = out + ((size_t)c * height + inRow) * width;
            const T* src = srcRow + (size_t)oh * outW;

            for (int ow = 0; ow < outW; ++ow)
            {
              const int inCol = ow * strideW - padW + kw;
              if (inCol < 0 || inCol >= width)
                continue;
              dst[inCol] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(dst[inCol]) +
                                                   hwy::ConvertScalarTo<AccumT>(src[ow]));
            }
          }
        }
      }
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MaxPool2DForward(const T* input, int batchSize, int channels,
                                              int height, int width, int poolH, int poolW,
                                              int strideH, int strideW, T* output,
                                              int* argmax) noexcept
{
  const int outH = (height - poolH) / strideH + 1;
  const int outW = (width - poolW) / strideW + 1;
  const size_t imageSize = (size_t)channels * height * width;
  const size_t outSize = (size_t)channels * outH * outW;

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
  for (int batch = 0; batch < batchSize; ++batch)
  {
    const T* in = input + (size_t)batch * imageSize;
    T* out = output + (size_t)batch * outSize;
    int* arg = argmax + (size_t)batch * outSize;

    for (int c = 0; c < channels; ++c)
    {
      const T* inPlane = in + (size_t)c * height * width;
      T* outPlane = out + (size_t)c * outH * outW;
      int* argPlane = arg + (size_t)c * outH * outW;

      for (int oh = 0; oh < outH; ++oh)
      {
        for (int ow = 0; ow < outW; ++ow)
        {
          AccumT bestVal = -std::numeric_limits<AccumT>::infinity();
          int bestIdx = 0;

          for (int ph = 0; ph < poolH; ++ph)
          {
            const int ih = oh * strideH + ph;
            for (int pw = 0; pw < poolW; ++pw)
            {
              const int iw = ow * strideW + pw;
              const int idx = ih * width + iw;
              AccumT v = hwy::ConvertScalarTo<AccumT>(inPlane[idx]);
              if (v > bestVal)
              {
                bestVal = v;
                bestIdx = idx;
              }
            }
          }

          outPlane[oh * outW + ow] = hwy::ConvertScalarTo<T>(bestVal);
          argPlane[oh * outW + ow] = bestIdx;
        }
      }
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MaxPool2DBackward(const T* outputGrad, const int* argmax,
                                               int batchSize, int channels, int height, int width,
                                               int outH, int outW, T* inputGrad) noexcept
{
  const size_t imageSize = (size_t)channels * height * width;
  const size_t outSize = (size_t)channels * outH * outW;

#pragma omp parallel for schedule(static) if (batchSize > 4 && !omp_in_parallel())
  for (int batch = 0; batch < batchSize; ++batch)
  {
    const T* grad = outputGrad + (size_t)batch * outSize;
    const int* arg = argmax + (size_t)batch * outSize;
    T* inGrad = inputGrad + (size_t)batch * imageSize;

    for (int c = 0; c < channels; ++c)
    {
      const T* gradPlane = grad + (size_t)c * outH * outW;
      const int* argPlane = arg + (size_t)c * outH * outW;
      T* inPlane = inGrad + (size_t)c * height * width;

      for (int i = 0; i < outH * outW; ++i)
      {
        int idx = argPlane[i];
        inPlane[idx] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(inPlane[idx]) +
                                               hwy::ConvertScalarTo<AccumT>(gradPlane[i]));
      }
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::RepackForBatchedGemm(T* dst, const T* src, size_t batchSize,
                                                  size_t rows, size_t cols) noexcept
{
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

      const T* s = src + u_batch * rows * cols + u_row * cols;
      T* d = dst + u_row * batchSize * cols + u_batch * cols;
      std::memcpy(d, s, cols * sizeof(T));
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::Fill(T* buf, size_t n, AccumT value) noexcept
{
  T encoded = hwy::ConvertScalarTo<T>(value);
  for (size_t i = 0; i < n; ++i)
    buf[i] = encoded;
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::CenterRows(T* buf, size_t rows, size_t cols) noexcept
{
  if (!buf || rows == 0 || cols == 0)
    return;
#pragma omp parallel for schedule(static) if (rows * cols > 262144 && !omp_in_parallel())
  for (size_t r = 0; r < rows; ++r)
  {
    T* row = buf + r * cols;
    AccumT sum = AccumT(0);
    for (size_t c = 0; c < cols; ++c)
      sum += hwy::ConvertScalarTo<AccumT>(row[c]);
    const AccumT mean = sum / static_cast<AccumT>(cols);
    for (size_t c = 0; c < cols; ++c)
      row[c] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(row[c]) - mean);
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AddBiasPerChannel(T* buf, const T* bias, size_t channels,
                                               size_t spatialSize) noexcept
{
#pragma omp parallel for schedule(static) if (channels * spatialSize > 65536 && !omp_in_parallel())
  for (size_t c = 0; c < channels; ++c)
  {
    AccumT biasVal = hwy::ConvertScalarTo<AccumT>(bias[c]);
    T* row = buf + c * spatialSize;
    for (size_t i = 0; i < spatialSize; ++i)
      row[i] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(row[i]) + biasVal);
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AddBiasPerChannelNCHW(T* buf, const T* bias, size_t batchSize,
                                                   size_t channels, size_t spatialSize) noexcept
{
  for (size_t b = 0; b < batchSize; ++b)
    AddBiasPerChannel(buf + b * channels * spatialSize, bias, channels, spatialSize);
}

template <typename T, typename AccumT>
AccumT CPUBackendT<T, AccumT>::ComputePrecisionWeightedErrorAndEnergy(T* e, const T* z, const T* mu,
                                                                      const T* p, size_t batchSize,
                                                                      size_t width) noexcept
{
  AccumT energy(0);
  const ptrdiff_t maxBatch = static_cast<ptrdiff_t>(batchSize);
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static) reduction(+ : energy)                                    \
    collapse(2) if (batchSize * width > 256 && !omp_in_parallel())
  for (ptrdiff_t b = 0; b < maxBatch; ++b)
  {
    for (ptrdiff_t i = 0; i < maxWidth; ++i)
    {
      size_t idx = (size_t)b * width + (size_t)i;
      AccumT err = hwy::ConvertScalarTo<AccumT>(z[idx]) - hwy::ConvertScalarTo<AccumT>(mu[idx]);
      e[idx] = hwy::ConvertScalarTo<T>(err);
      AccumT precision = (std::max)(hwy::ConvertScalarTo<AccumT>(p[i]), AccumT(1e-8));
      energy += AccumT(0.5) * precision * err * err - AccumT(0.5) * std::log(precision);
    }
  }
  return energy;
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::ComputePrecisionWeightedError(T* e, const T* z, const T* mu,
                                                           const T* /*p*/, size_t batchSize,
                                                           size_t width) noexcept
{
  const size_t n = batchSize * width;
#pragma omp parallel for schedule(static) if (n > 100000 && !omp_in_parallel())
  for (size_t idx = 0; idx < n; ++idx)
    e[idx] = hwy::ConvertScalarTo<T>(hwy::ConvertScalarTo<AccumT>(z[idx]) -
                                     hwy::ConvertScalarTo<AccumT>(mu[idx]));
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::AxpyBroadcastInto(T* y, const T* x, const T* factor, size_t batchSize,
                                               size_t width, AccumT alpha) noexcept
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
      AccumT v = hwy::ConvertScalarTo<AccumT>(y[idx]) +
                alpha * hwy::ConvertScalarTo<AccumT>(x[idx]) * hwy::ConvertScalarTo<AccumT>(factor[i]);
      y[idx] = hwy::ConvertScalarTo<T>(v);
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::MultiplyBroadcastInto(T* dst, const T* a, const T* factor, const T* b,
                                                    size_t batchSize, size_t width) noexcept
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
      AccumT v = hwy::ConvertScalarTo<AccumT>(a[idx]) * hwy::ConvertScalarTo<AccumT>(factor[i]) *
                hwy::ConvertScalarTo<AccumT>(b[idx]);
      dst[idx] = hwy::ConvertScalarTo<T>(v);
    }
  }
}

template <typename T, typename AccumT>
void CPUBackendT<T, AccumT>::UpdatePrecisionFromError(T* p, T* log_p, const T* e, size_t batchSize,
                                                      size_t width, AccumT pr) noexcept
{
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static) if (width > 4096 && !omp_in_parallel())
  for (ptrdiff_t i = 0; i < maxWidth; ++i)
  {
    AccumT grad(0);
    AccumT pi = hwy::ConvertScalarTo<AccumT>(p[i]);
    for (size_t b = 0; b < batchSize; ++b)
    {
      AccumT err = hwy::ConvertScalarTo<AccumT>(e[b * width + (size_t)i]);
      grad += AccumT(0.5) * (pi * err * err - AccumT(1));
    }
    grad /= AccumT(batchSize);

    AccumT logPi = hwy::ConvertScalarTo<AccumT>(log_p[i]) - pr * grad;
    logPi = (std::max)(AccumT(-5), (std::min)(logPi, AccumT(5)));
    log_p[i] = hwy::ConvertScalarTo<T>(logPi);
    p[i] = hwy::ConvertScalarTo<T>(std::exp(logPi));
  }
}

// --- Explicit instantiation: the (T, AccumT) pairs this templating
// pass proves the machinery with so far (see the plan's "Implementation
// order" section) -- every future T added there needs a matching line
// here. Fp4E2M1Value pairs with AccumT=float (NOT the default AccumT=T
// a FullPCLayerT<Fp4E2M1Value> caller would get by leaving AccumT
// unspecified): the whole point of the AccumT split is that the
// optimizer accumulator stays wide even when storage is 4-bit, so
// Fp4E2M1Value must always be spelled with an explicit, wider AccumT.
template class CPUBackendT<float, float>;
template class CPUBackendT<double, double>;
template class CPUBackendT<Fp4E2M1Value, float>;
template class CPUBackendT<hwy::float16_t, float>;
template class CPUBackendT<Posit4_0Value, float>;
template class CPUBackendT<Posit4_1Value, float>;
template class CPUBackendT<Posit4_2Value, float>;

} // namespace Deep
