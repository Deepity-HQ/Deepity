#pragma once

#include <cstdio>
#include <cstring>
#include <deepity/backend/IComputeBackend.h>

/**
 * @file RandomizeDistribution.h
 * @brief Shared "name(a,b)" distribution-string parser for every PC
 * layer's RandomizeWeights(rng, distribution) overload.
 */

#ifdef _MSC_VER
#define DEEPITY_SCANF_DISTRIBUTION(str, name, a, b)                                                \
  sscanf_s((str), "%15[^(](%f,%f)", (name), (unsigned)sizeof(name), &(a), &(b))
#else
#define DEEPITY_SCANF_DISTRIBUTION(str, name, a, b)                                                \
  sscanf((str), "%15[^(](%f,%f)", (name), &(a), &(b))
#endif

namespace Deep
{
/// @brief Randomizes `buf` from a "name(a,b)" distribution string, e.g.
/// "normal(0,1)" or "uniform(-0.3,0.3)". A `distribution` that doesn't
/// parse as one of those two names (including the bare word "normal"
/// with no "(a,b)") falls back to normal(fallbackMean, fallbackStddev).
inline void RandomizeWithDistribution(IComputeBackend* backend, float* buf, size_t n,
                                      const char* distribution, uint32_t seed, float fallbackMean,
                                      float fallbackStddev) noexcept
{
  char name[16] = {0};
  float a = 0.0f, b = 1.0f;

  if (!distribution || DEEPITY_SCANF_DISTRIBUTION(distribution, name, a, b) != 3)
  {
    std::memcpy(name, "normal", sizeof("normal"));
    a = fallbackMean;
    b = fallbackStddev;
  }

  if (std::strcmp(name, "normal") == 0)
    backend->RandomizeNormal(buf, n, a, b, seed);
  else if (std::strcmp(name, "uniform") == 0)
    backend->RandomizeUniform(buf, n, a, b, seed);
  else
    backend->RandomizeNormal(buf, n, fallbackMean, fallbackStddev, seed);
}

/// @brief Templated counterpart to RandomizeWithDistribution(). The
/// "name(a,b)" literals are parsed as plain floats, then widened to
/// AccumT for the RandomizeNormal/RandomizeUniform call.
template <typename T, typename AccumT>
inline void RandomizeWithDistributionT(IComputeBackendT<T, AccumT>* backend, T* buf, size_t n,
                                       const char* distribution, uint32_t seed,
                                       AccumT fallbackMean, AccumT fallbackStddev) noexcept
{
  char name[16] = {0};
  float a = 0.0f, b = 1.0f;

  if (!distribution || DEEPITY_SCANF_DISTRIBUTION(distribution, name, a, b) != 3)
  {
    std::memcpy(name, "normal", sizeof("normal"));
    backend->RandomizeNormal(buf, n, fallbackMean, fallbackStddev, seed);
    return;
  }

  if (std::strcmp(name, "normal") == 0)
    backend->RandomizeNormal(buf, n, AccumT(a), AccumT(b), seed);
  else if (std::strcmp(name, "uniform") == 0)
    backend->RandomizeUniform(buf, n, AccumT(a), AccumT(b), seed);
  else
    backend->RandomizeNormal(buf, n, fallbackMean, fallbackStddev, seed);
}
} // namespace Deep
