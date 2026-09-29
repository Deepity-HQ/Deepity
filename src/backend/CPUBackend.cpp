#include <cmath>
#include <cstdlib>
#include <cstring>
#include <deepity/backend/CPUBackend.h>
#include <deepity/utils/Im2Col.h>
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

namespace Deep
{
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

void CPUBackend::MatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
                        int lda, const float* B, int ldb, float beta, float* C, int ldc) noexcept
{ // Multithreading is the caller's job.
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
}

void CPUBackend::SumRows(float* dst, const float* src, size_t batchSize, size_t width) noexcept
{
  memset(dst, 0, width * sizeof(float));
  for (size_t b = 0; b < batchSize; ++b)
    cblas_saxpy(width, 1.0f, src + b * width, 1, dst, 1);
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

void CPUBackend::FusedStateUpdateMomentum(float* z, float* v, const float* feedback,
                                          ActivationType dType, const float* e, size_t n, float ir,
                                          float beta) noexcept
{
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

#pragma omp parallel for reduction(+ : energy) schedule(static) if (n > 256 && !omp_in_parallel())
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
#pragma omp parallel for schedule(static) if (n > 256 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
    e[i] = z[i] - mu[i];
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

#pragma omp parallel for schedule(static) if (n > 256 && !omp_in_parallel())
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

#pragma omp parallel for schedule(static) if (n > 256 && !omp_in_parallel())
  for (size_t i = 0; i < n; ++i)
  {
    float g = grad[i];
    m[i] = beta1 * m[i] + (1.0f - beta1) * g;
    v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

    param[i] -= *lr * weightDecay * param[i];
    param[i] -= step_size * m[i] / (Sleef_sqrtf(v[i]) + eps);
  }
}

void CPUBackend::Im2Col(const float* input, int channels, int height, int width, int kernelH,
                        int kernelW, int strideH, int strideW, int padH, int padW,
                        float* columns) noexcept
{
  Deep::Im2Col(
      input, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, columns);
}

void CPUBackend::Col2Im(const float* columns, int channels, int height, int width, int kernelH,
                        int kernelW, int strideH, int strideW, int padH, int padW,
                        float* outputImage) noexcept
{
  Deep::Col2Im(columns,
               channels,
               height,
               width,
               kernelH,
               kernelW,
               strideH,
               strideW,
               padH,
               padW,
               outputImage);
}

void CPUBackend::RepackForBatchedGemm(float* dst, const float* src, size_t batchSize, size_t rows,
                                      size_t cols) noexcept
{
  // dst[row][batch][:] = src[batch][row][:], matches
  // SimpleConvPCLayer::UpdateWeights()'s colsRepacked/lgRepacked loops
  // exactly (same index arithmetic, same collapse(2) parallelization).
  const int maxRows = static_cast<int>(rows);
  const int maxBatch = static_cast<int>(batchSize);

#pragma omp parallel for schedule(static) collapse(2)
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

void CPUBackend::FusedActivationDerivativeMultiply(float* dst, const float* a, float* activatedInOut,
                                                   ActivationType dType, size_t n) noexcept
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

void CPUBackend::Fill(float* buf, size_t n, float value) noexcept
{
  FillDispatch(buf, n, value);
}

void CPUBackend::AddBiasPerChannel(float* buf, const float* bias, size_t channels,
                                   size_t spatialSize) noexcept
{
  AddBiasPerChannelDispatch(buf, bias, channels, spatialSize);
}

float CPUBackend::ComputePrecisionWeightedErrorAndEnergy(float* e, const float* z, const float* mu,
                                                         const float* p, size_t batchSize,
                                                         size_t width) noexcept
{
  float energy = 0.0f;
  const ptrdiff_t maxBatch = static_cast<ptrdiff_t>(batchSize);
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static) reduction(+ : energy) collapse(2) \
    if (batchSize * width > 256 && !omp_in_parallel())
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

void CPUBackend::AxpyBroadcastInto(float* y, const float* x, const float* factor, size_t batchSize,
                                   size_t width, float alpha) noexcept
{
  const ptrdiff_t maxBatch = static_cast<ptrdiff_t>(batchSize);
  const ptrdiff_t maxWidth = static_cast<ptrdiff_t>(width);

#pragma omp parallel for schedule(static) collapse(2) if (batchSize * width > 65536 && !omp_in_parallel())
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

#pragma omp parallel for schedule(static) collapse(2) if (batchSize * width > 65536 && !omp_in_parallel())
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
