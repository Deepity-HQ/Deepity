/**
 * @file tFloat16HighwayVerify.cpp
 * @brief TEMP (issue #50): verifies CPUBackendT<hwy::float16_t, float>'s
 * new genuine-Highway-SIMD Activation()/ActivationInto()/
 * ActivationDerivativeInto()/ActivationDerivativeIntoAccum() paths
 * (CPUBackendHighway.cpp's ActivationF16Hwy family) against the
 * generic scalar ConvertScalarTo<AccumT> fallback every other
 * quantized T still uses, across every ActivationType, then times
 * both paths on a large buffer to confirm the SIMD path is actually
 * faster (the whole point of issue #50 -- f16 is meant to get real
 * vectorized compute, not the decode/compute/encode wrapper every
 * other sub-8-bit format uses).
 */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <deepity/backend/CPUBackendT.h>
#include <deepity/utils/Activations.h>
#include <hwy/base.h>
#include <random>
#include <string>
#include <vector>

using namespace Deep;

namespace
{
const ActivationType kForwardTypes[] = {
    ActivationType::RELU,   ActivationType::GELU,      ActivationType::SIGMOID,
    ActivationType::eSIGMOID, ActivationType::TANH,     ActivationType::HARD_TANH,
    ActivationType::LINEAR,
};
const ActivationType kDerivTypes[] = {
    ActivationType::dRELU,   ActivationType::dGELU,      ActivationType::dSIGMOID,
    ActivationType::d_eSIGMOID, ActivationType::dTANH,     ActivationType::dHARD_TANH,
    ActivationType::dLINEAR,
};

const char* TypeName(ActivationType t)
{
  switch (t)
  {
  case ActivationType::RELU: return "RELU";
  case ActivationType::GELU: return "GELU";
  case ActivationType::SIGMOID: return "SIGMOID";
  case ActivationType::eSIGMOID: return "eSIGMOID";
  case ActivationType::TANH: return "TANH";
  case ActivationType::HARD_TANH: return "HARD_TANH";
  case ActivationType::LINEAR: return "LINEAR";
  case ActivationType::dRELU: return "dRELU";
  case ActivationType::dGELU: return "dGELU";
  case ActivationType::dSIGMOID: return "dSIGMOID";
  case ActivationType::d_eSIGMOID: return "d_eSIGMOID";
  case ActivationType::dTANH: return "dTANH";
  case ActivationType::dHARD_TANH: return "dHARD_TANH";
  case ActivationType::dLINEAR: return "dLINEAR";
  default: return "?";
  }
}

/// @brief Reference "generic scalar wrap" path, exactly what every
/// OTHER quantized T still falls back to (see CPUBackendT.cpp's own
/// comment on that Bucket-C wrap) -- NOT calling into the new f16
/// dispatch at all, so this is a genuine independent reference.
std::vector<hwy::float16_t> ScalarForwardRef(ActivationType type,
                                             const std::vector<hwy::float16_t>& src)
{
  std::vector<hwy::float16_t> out(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    out[i] = hwy::ConvertScalarTo<hwy::float16_t>(
        ActivationScalarT<float>(type, hwy::ConvertScalarTo<float>(src[i])));
  return out;
}

std::vector<hwy::float16_t> ScalarDerivRefT(ActivationType dType,
                                            const std::vector<hwy::float16_t>& src)
{
  std::vector<hwy::float16_t> out(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    out[i] = hwy::ConvertScalarTo<hwy::float16_t>(
        ActivationDerivativeScalarT<float>(dType, hwy::ConvertScalarTo<float>(src[i])));
  return out;
}

std::vector<float> ScalarDerivRefAccum(ActivationType dType, const std::vector<hwy::float16_t>& src)
{
  std::vector<float> out(src.size());
  for (size_t i = 0; i < src.size(); ++i)
    out[i] = ActivationDerivativeScalarT<float>(dType, hwy::ConvertScalarTo<float>(src[i]));
  return out;
}

/// @brief f16 has ~10 bits of mantissa; a few ULPs of float32
/// difference between hn::Tanh's polynomial and std::tanh (see
/// CPUBackendHighway.cpp's own file comment on that tradeoff) almost
/// always rounds to the identical f16 value, but not guaranteed to --
/// this allows up to 1 ULP of final f16 slack rather than requiring
/// exact (==).
bool CloseF16(hwy::float16_t a, hwy::float16_t b, const char* label, size_t i)
{
  float fa = hwy::ConvertScalarTo<float>(a);
  float fb = hwy::ConvertScalarTo<float>(b);
  float diff = std::fabs(fa - fb);
  float scale = std::max(1.0f, std::max(std::fabs(fa), std::fabs(fb)));
  if (diff > scale * 2e-3f) // ~1 f16 ULP relative tolerance
  {
    printf("  MISMATCH in %s at index %zu: ref=%.6g hwy=%.6g\n", label, i, (double)fa, (double)fb);
    return false;
  }
  return true;
}

bool CloseF32(float a, float b, const char* label, size_t i)
{
  float diff = std::fabs(a - b);
  float scale = std::max(1.0f, std::max(std::fabs(a), std::fabs(b)));
  if (diff > scale * 2e-3f)
  {
    printf("  MISMATCH in %s at index %zu: ref=%.6g hwy=%.6g\n", label, i, (double)a, (double)b);
    return false;
  }
  return true;
}

/// @brief Prevents the compiler from proving a buffer's writes are
/// dead and deleting the whole loop that produced them -- the
/// Highway path below is always "live" (it crosses an opaque,
/// separately-compiled-library function call the optimizer can't see
/// through), but a hand-written scalar loop living in the SAME
/// translation unit as main() has no such protection: without this,
/// a first attempt at this benchmark had GCC silently eliminating the
/// entire "scalar fallback" loop (nothing ever read bigDst
/// afterward), making it look many times faster than the real
/// Highway kernel it was supposed to be compared against.
void ForceLive(const void* p)
{
  asm volatile("" : : "r"(p) : "memory");
}
} // namespace

int main()
{
  CPUBackendT<hwy::float16_t, float> backend;

  const size_t n = 4001; // deliberately not a multiple of any lane width, exercises the scalar tail
  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<hwy::float16_t> src(n);
  for (auto& v : src)
    v = hwy::ConvertScalarTo<hwy::float16_t>(dist(rng));

  bool ok = true;

  // --- Part 1: Activation() (in-place forward), every ActivationType. ---
  for (ActivationType type : kForwardTypes)
  {
    std::vector<hwy::float16_t> buf = src;
    backend.Activation(type, buf.data(), n);
    std::vector<hwy::float16_t> ref = ScalarForwardRef(type, src);
    for (size_t i = 0; i < n; ++i)
      ok &= CloseF16(ref[i], buf[i], (std::string("Activation/") + TypeName(type)).c_str(), i);
  }

  // --- Part 2: ActivationInto() (two-buffer forward). ---
  for (ActivationType type : kForwardTypes)
  {
    std::vector<hwy::float16_t> dst(n);
    backend.ActivationInto(type, dst.data(), src.data(), n);
    std::vector<hwy::float16_t> ref = ScalarForwardRef(type, src);
    for (size_t i = 0; i < n; ++i)
      ok &= CloseF16(ref[i], dst[i], (std::string("ActivationInto/") + TypeName(type)).c_str(), i);
  }

  // --- Part 3: ActivationDerivativeInto() (two-buffer, derivative from preact). ---
  for (ActivationType dType : kDerivTypes)
  {
    std::vector<hwy::float16_t> dst(n);
    backend.ActivationDerivativeInto(dType, dst.data(), src.data(), n);
    std::vector<hwy::float16_t> ref = ScalarDerivRefT(dType, src);
    for (size_t i = 0; i < n; ++i)
      ok &= CloseF16(ref[i], dst[i],
                     (std::string("ActivationDerivativeInto/") + TypeName(dType)).c_str(), i);
  }

  // --- Part 4: ActivationDerivativeIntoAccum() (AccumT=float output, no demote). ---
  for (ActivationType dType : kDerivTypes)
  {
    std::vector<float> dst(n);
    backend.ActivationDerivativeIntoAccum(dType, dst.data(), src.data(), n);
    std::vector<float> ref = ScalarDerivRefAccum(dType, src);
    for (size_t i = 0; i < n; ++i)
      ok &= CloseF32(ref[i], dst[i],
                     (std::string("ActivationDerivativeIntoAccum/") + TypeName(dType)).c_str(), i);
  }

  printf(ok ? "PASS: CPUBackendT<hwy::float16_t,float>'s Activation family matches the scalar "
              "reference within 1 f16 ULP\n"
            : "FAIL: see mismatches above\n");

  // --- Part 5: timing comparison on a large buffer -- confirms the
  // new Highway SIMD path is actually faster than the generic scalar
  // fallback every other quantized T still uses (the whole point of
  // issue #50). ---
  const size_t bigN = 8'000'000;
  std::vector<hwy::float16_t> bigSrc(bigN);
  for (auto& v : bigSrc)
    v = hwy::ConvertScalarTo<hwy::float16_t>(dist(rng));
  std::vector<hwy::float16_t> bigDst(bigN);

  const int reps = 20;

  // Warm up both buffers (first-touch page faults) before timing either
  // path, so neither one's measurement is skewed by the other running
  // second on already-resident memory.
  backend.ActivationInto(ActivationType::LINEAR, bigDst.data(), bigSrc.data(), bigN);

  auto t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
  {
    backend.ActivationInto(ActivationType::TANH, bigDst.data(), bigSrc.data(), bigN);
    ForceLive(bigDst.data());
  }
  auto t1 = std::chrono::steady_clock::now();
  double simdMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
  {
    for (size_t i = 0; i < bigN; ++i)
      bigDst[i] = hwy::ConvertScalarTo<hwy::float16_t>(
          ActivationScalarT<float>(ActivationType::TANH, hwy::ConvertScalarTo<float>(bigSrc[i])));
    ForceLive(bigDst.data());
  }
  t1 = std::chrono::steady_clock::now();
  double scalarMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  printf("TANH over %zu elements x%d: Highway SIMD=%.2fms, scalar fallback=%.2fms (%.2fx)\n", bigN,
        reps, simdMs, scalarMs, scalarMs / simdMs);

  bool ok5 = simdMs < scalarMs;
  printf(ok5 ? "PASS: Highway SIMD path is faster than the scalar fallback\n"
             : "FAIL: Highway SIMD path was not faster (see timings above)\n");

  // --- Diagnostic: isolate PromoteTo/DemoteTo conversion overhead
  // from hn::Tanh overhead by timing RELU (no transcendental) too. ---
  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
  {
    backend.ActivationInto(ActivationType::RELU, bigDst.data(), bigSrc.data(), bigN);
    ForceLive(bigDst.data());
  }
  t1 = std::chrono::steady_clock::now();
  double simdReluMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

  t0 = std::chrono::steady_clock::now();
  for (int r = 0; r < reps; ++r)
  {
    for (size_t i = 0; i < bigN; ++i)
      bigDst[i] = hwy::ConvertScalarTo<hwy::float16_t>(
          ActivationScalarT<float>(ActivationType::RELU, hwy::ConvertScalarTo<float>(bigSrc[i])));
    ForceLive(bigDst.data());
  }
  t1 = std::chrono::steady_clock::now();
  double scalarReluMs = std::chrono::duration<double, std::milli>(t1 - t0).count();
  printf("RELU over %zu elements x%d: Highway SIMD=%.2fms, scalar fallback=%.2fms (%.2fx)\n", bigN,
        reps, simdReluMs, scalarReluMs, scalarReluMs / simdReluMs);

  return (ok && ok5) ? 0 : 1;
}
