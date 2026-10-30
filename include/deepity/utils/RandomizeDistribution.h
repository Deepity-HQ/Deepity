#pragma once

#include <cstdio>
#include <cstring>
#include <deepity/backend/IComputeBackend.h>

/**
 * @file RandomizeDistribution.h
 * @brief Shared "name(a,b)" distribution-string parser for every PC layer's
 * RandomizeWeights(rng, distribution) overload, generalizing the parsing
 * SimplePCLayer::RandomizeWeights() originally did inline. One copy
 * instead of one per layer file, so e.g. "normal"/"uniform" spelling and
 * the malformed-string fallback behavior can't drift between layers.
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
/// "normal(0,1)" or "uniform(-0.3,0.3)". `distribution` not parsing as
/// one of those two names (including every layer's own default argument,
/// the bare word "normal" with no "(a,b)" -- deliberately unparseable)
/// falls back to normal(fallbackMean, fallbackStddev): each layer's own
/// pre-existing, shape-aware init formula, unchanged for every caller
/// that doesn't pass an explicit distribution.
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

/// @brief Templated counterpart to RandomizeWithDistribution(), for
/// FullPCLayer<T, AccumT>: the "name(a,b)" literals are parsed as plain
/// floats (parameter strings aren't expected to carry double-precision
/// detail), then widened to AccumT for the RandomizeNormal/
/// RandomizeUniform call, matching IComputeBackendT<T, AccumT>'s own
/// AccumT-typed distribution parameters.
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
