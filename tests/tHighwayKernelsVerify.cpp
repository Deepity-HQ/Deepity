/**
 * @file tHighwayKernelsVerify.cpp
 * @brief Correctness check for the two Highway-dispatched CPUBackend
 * kernels (AddBiasPerChannel, FusedActivationDerivativeMultiply) against
 * an independent scalar reference -- deliberately NOT reusing any
 * backend code, matching tCrossEntropyVerify.cpp's own convention.
 *
 * Sizes are swept across a range DELIBERATELY including non-SIMD-aligned
 * ("odd") totals (e.g. spatialSize=1, 3, 7, 17, 31, 200000+1) -- tail
 * handling is the single most common real SIMD bug, and a happy-path-
 * only power-of-two test would never catch one.
 */
#include <cmath>
#include <cstdio>
#include <deepity/backend/CPUBackend.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
bool NearlyEqual(float a, float b, float tol = 1e-5f)
{
  return std::fabs(a - b) <= tol * (std::fabs(a) + std::fabs(b) + 1e-6f);
}

// Independent reference, plain scalar, matching AddBiasPerChannel's own
// documented contract exactly.
void ReferenceAddBiasPerChannel(std::vector<float>& buf, const std::vector<float>& bias,
                                size_t channels, size_t spatialSize)
{
  for (size_t c = 0; c < channels; ++c)
    for (size_t s = 0; s < spatialSize; ++s)
      buf[c * spatialSize + s] += bias[c];
}

// Independent reference for FusedActivationDerivativeMultiply, matching
// ActivationDerivativeFromActivatedScalar's own formulas.
void ReferenceFusedADM(std::vector<float>& dst, const std::vector<float>& a,
                       std::vector<float>& activatedInOut, ActivationType dType, size_t n)
{
  for (size_t i = 0; i < n; ++i)
  {
    float act = activatedInOut[i];
    float deriv;
    switch (dType)
    {
    case ActivationType::dRELU:
      deriv = (act > 0.0f) ? 1.0f : 0.0f;
      break;
    case ActivationType::dSIGMOID:
      deriv = act * (1.0f - act);
      break;
    case ActivationType::d_eSIGMOID:
      deriv = 2.0f * act * (1.0f - act);
      break;
    case ActivationType::dTANH:
      deriv = 1.0f - act * act;
      break;
    default:
      deriv = 1.0f;
      break;
    }
    dst[i] = a[i] * deriv;
    activatedInOut[i] = deriv;
  }
}

bool TestAddBiasPerChannel(CPUBackend& backend, size_t channels, size_t spatialSize, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> bufRef(channels * spatialSize), bufHwy(channels * spatialSize);
  std::vector<float> bias(channels);
  for (auto& v : bufRef)
    v = dist(rng);
  bufHwy = bufRef;
  for (auto& v : bias)
    v = dist(rng);

  ReferenceAddBiasPerChannel(bufRef, bias, channels, spatialSize);
  backend.AddBiasPerChannel(bufHwy.data(), bias.data(), channels, spatialSize);

  for (size_t i = 0; i < bufRef.size(); ++i)
  {
    if (!NearlyEqual(bufRef[i], bufHwy[i]))
    {
      printf("  FAIL AddBiasPerChannel(channels=%zu, spatialSize=%zu): mismatch at %zu: "
             "ref=%.6f hwy=%.6f\n",
             channels, spatialSize, i, bufRef[i], bufHwy[i]);
      return false;
    }
  }
  return true;
}

bool TestFill(CPUBackend& backend, size_t n, float value, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> bufRef(n), bufHwy(n);
  for (auto& v : bufRef)
    v = dist(rng); // pre-fill with garbage so a no-op Fill can't accidentally pass
  bufHwy = bufRef;

  for (auto& v : bufRef)
    v = value;
  backend.Fill(bufHwy.data(), n, value);

  for (size_t i = 0; i < n; ++i)
  {
    if (!NearlyEqual(bufRef[i], bufHwy[i]))
    {
      printf("  FAIL Fill(n=%zu): mismatch at %zu: ref=%.6f hwy=%.6f\n", n, i, bufRef[i], bufHwy[i]);
      return false;
    }
  }
  return true;
}

bool TestMultiplyInto(CPUBackend& backend, size_t n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);

  std::vector<float> a(n), b(n), dstRef(n), dstHwy(n);
  for (auto& v : a)
    v = dist(rng);
  for (auto& v : b)
    v = dist(rng);

  for (size_t i = 0; i < n; ++i)
    dstRef[i] = a[i] * b[i];
  backend.MultiplyInto(dstHwy.data(), a.data(), b.data(), n);

  for (size_t i = 0; i < n; ++i)
  {
    if (!NearlyEqual(dstRef[i], dstHwy[i]))
    {
      printf("  FAIL MultiplyInto(n=%zu): mismatch at %zu: ref=%.6f hwy=%.6f\n", n, i, dstRef[i],
             dstHwy[i]);
      return false;
    }
  }
  return true;
}

bool TestFusedADM(CPUBackend& backend, ActivationType dType, size_t n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> distA(-1.0f, 1.0f);
  std::uniform_real_distribution<float> distAct(-0.99f, 0.99f); // valid "activated" range

  std::vector<float> a(n), actRef(n), actHwy(n), dstRef(n), dstHwy(n);
  for (auto& v : a)
    v = distA(rng);
  for (auto& v : actRef)
    v = distAct(rng);
  actHwy = actRef;

  ReferenceFusedADM(dstRef, a, actRef, dType, n);
  backend.FusedActivationDerivativeMultiply(dstHwy.data(), a.data(), actHwy.data(), dType, n);

  for (size_t i = 0; i < n; ++i)
  {
    if (!NearlyEqual(dstRef[i], dstHwy[i]) || !NearlyEqual(actRef[i], actHwy[i]))
    {
      printf("  FAIL FusedADM(dType=%d, n=%zu): mismatch at %zu: dst ref=%.6f hwy=%.6f, "
             "act ref=%.6f hwy=%.6f\n",
             (int)dType, n, i, dstRef[i], dstHwy[i], actRef[i], actHwy[i]);
      return false;
    }
  }
  return true;
}
} // namespace

int main()
{
  CPUBackend backend;
  bool allPass = true;

  // Deliberately includes tiny, odd, and large-with-odd-remainder sizes.
  std::vector<size_t> sizes = {0, 1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 63, 64, 100, 4095, 4096, 4097,
                               200003};

  printf("=== AddBiasPerChannel ===\n");
  for (size_t channels : {1, 3, 8, 16})
    for (size_t spatial : sizes)
    {
      bool ok = TestAddBiasPerChannel(backend, channels, spatial, (unsigned)(channels * 97 + spatial));
      allPass = allPass && ok;
    }
  printf("%s\n\n", allPass ? "PASS" : "FAIL");

  printf("=== FusedActivationDerivativeMultiply ===\n");
  bool admPass = true;
  for (ActivationType t : {ActivationType::dRELU, ActivationType::dSIGMOID,
                           ActivationType::d_eSIGMOID, ActivationType::dTANH,
                           ActivationType::dLINEAR, ActivationType::NONE})
    for (size_t n : sizes)
    {
      bool ok = TestFusedADM(backend, t, n, (unsigned)((int)t * 97 + n));
      admPass = admPass && ok;
    }
  printf("%s\n\n", admPass ? "PASS" : "FAIL");

  allPass = allPass && admPass;

  printf("=== Fill ===\n");
  bool fillPass = true;
  for (size_t n : sizes)
    for (float value : {0.0f, 1.0f, -3.5f})
      fillPass = fillPass && TestFill(backend, n, value, (unsigned)(n * 7 + (int)(value * 100)));
  printf("%s\n\n", fillPass ? "PASS" : "FAIL");

  printf("=== MultiplyInto ===\n");
  bool mulPass = true;
  for (size_t n : sizes)
    mulPass = mulPass && TestMultiplyInto(backend, n, (unsigned)(n * 13 + 1));
  printf("%s\n\n", mulPass ? "PASS" : "FAIL");

  allPass = allPass && fillPass && mulPass;
  printf("%s\n", allPass ? "PASS" : "FAIL");
  return allPass ? 0 : 1;
}
