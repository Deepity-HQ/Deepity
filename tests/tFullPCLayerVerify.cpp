/**
 * @file tFullPCLayerVerify.cpp
 * @brief One test, two parts.
 *
 * Part A: FullPCLayer with defaults (a=1.0, useResidual=false) vs
 * DirectKPPCLayer, identically seeded, same input/target, one full
 * CalculateState+UpdateState+UpdateWeights cycle. Should match exactly
 * -- confirms the refactor didn't change default behavior at all.
 *
 * Part B: FullPCLayer with a=0.5, useResidual=true (size==nextSize),
 * a few real training steps. Sanity-level only (no NaN/Inf, energy
 * stays finite) -- not a full hand-derived-value check, matching
 * "just one test, enough that it runs."
 */
#include <cmath>
#include <cstdio>
#include <deepity/layers/DirectKPPCLayer.h>
#include <deepity/layers/FullPCLayer.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
bool AllFinite(const float* buf, size_t n)
{
  for (size_t i = 0; i < n; ++i)
    if (!std::isfinite(buf[i]))
      return false;
  return true;
}

float MaxAbsDiff(const float* a, const float* b, size_t n)
{
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
} // namespace

int main()
{
  const size_t size = 8, nextSize = 8, terminalSize = 4, batchSize = 4;
  const float lr = 0.01f, ir = 0.1f, fl = 0.001f, lmbda = 0.0f;

  std::vector<float> input(batchSize * size), target(batchSize * terminalSize);
  std::mt19937 dataRng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dist(dataRng);
  for (auto& v : target)
    v = dist(dataRng);

  // ================= PART A: default-behavior match =================
  printf("=== Part A: FullPCLayer defaults vs DirectKPPCLayer ===\n");

  DirectKPPCLayer dLayer0(size,
                          nextSize,
                          terminalSize,
                          batchSize,
                          lr,
                          ir,
                          fl,
                          lmbda,
                          ActivationType::TANH,
                          ActivationType::dTANH);
  DirectKPPCLayer dLayer1(nextSize,
                          0,
                          terminalSize,
                          batchSize,
                          lr,
                          ir,
                          fl,
                          lmbda,
                          ActivationType::LINEAR,
                          ActivationType::dLINEAR);
  dLayer0.SetLayerAbove(&dLayer1);
  dLayer1.SetLayerBelow(&dLayer0);
  dLayer0.SetTerminalLayer(&dLayer1);
  dLayer1.SetTerminalLayer(&dLayer1);
  dLayer0.SetOptimizer(OptimizerType::SGD);
  dLayer0.SetPsiOptimizer(OptimizerType::SGD);

  FullPCLayer fLayer0(size,
                      nextSize,
                      terminalSize,
                      batchSize,
                      lr,
                      ir,
                      fl,
                      lmbda,
                      ActivationType::TANH,
                      ActivationType::dTANH);
  FullPCLayer fLayer1(nextSize,
                      0,
                      terminalSize,
                      batchSize,
                      lr,
                      ir,
                      fl,
                      lmbda,
                      ActivationType::LINEAR,
                      ActivationType::dLINEAR);
  fLayer0.SetLayerAbove(&fLayer1);
  fLayer1.SetLayerBelow(&fLayer0);
  fLayer0.SetTerminalLayer(&fLayer1);
  fLayer1.SetTerminalLayer(&fLayer1);
  fLayer0.SetOptimizer(OptimizerType::SGD);
  fLayer0.SetPsiOptimizer(OptimizerType::SGD);
  // Defaults: a=1.0, useResidual=false -- deliberately NOT set here,
  // to confirm the class's own defaults reproduce DirectKPPCLayer.

  // Identically-seeded RNGs, not a weight copy -- RandomizeWeights()
  // is deterministic given the same seed and identical underlying
  // logic (confirmed: both classes' RandomizeWeights() bodies are
  // byte-identical), so this produces bit-identical W/Psi on both
  // layers without needing a non-const weight setter.
  std::mt19937 rngD(42), rngF(42);
  dLayer0.RandomizeWeights(rngD);
  fLayer0.RandomizeWeights(rngF);

  dLayer0.ClampState(input);
  dLayer1.ClampState(target);
  fLayer0.ClampState(input);
  fLayer1.ClampState(target);

  dLayer0.CalculateState(false);
  dLayer1.CalculateState(false);
  dLayer0.DirectFeedbackUpdate();
  dLayer0.UpdateState();
  dLayer0.UpdateWeights();

  fLayer0.CalculateState(false);
  fLayer1.CalculateState(false);
  fLayer0.DirectFeedbackUpdate();
  fLayer0.UpdateState();
  fLayer0.UpdateWeights();

  float muDiff = MaxAbsDiff(dLayer0.GetMu(), fLayer0.GetMu(), batchSize * nextSize);
  float wDiff = MaxAbsDiff(dLayer0.GetWeights(), fLayer0.GetWeights(), size * nextSize);
  float zDiff = MaxAbsDiff(dLayer0.GetBeliefs(), fLayer0.GetBeliefs(), batchSize * size);

  printf("mu max abs diff: %g\n", muDiff);
  printf("W  max abs diff: %g\n", wDiff);
  printf("z  max abs diff: %g\n", zDiff);

  bool partAPass = (muDiff < 1e-5f) && (wDiff < 1e-5f) && (zDiff < 1e-5f);
  printf("Part A: %s\n\n", partAPass ? "PASS" : "FAIL");

  // ================= PART B: a/residual sanity check =================
  printf("=== Part B: a=0.5, useResidual=true, sanity only ===\n");

  FullPCLayer gLayer0(size,
                      nextSize,
                      terminalSize,
                      batchSize,
                      lr,
                      ir,
                      fl,
                      lmbda,
                      ActivationType::TANH,
                      ActivationType::dTANH);
  FullPCLayer gLayer1(nextSize,
                      0,
                      terminalSize,
                      batchSize,
                      lr,
                      ir,
                      fl,
                      lmbda,
                      ActivationType::LINEAR,
                      ActivationType::dLINEAR);
  gLayer0.SetLayerAbove(&gLayer1);
  gLayer1.SetLayerBelow(&gLayer0);
  gLayer0.SetTerminalLayer(&gLayer1);
  gLayer1.SetTerminalLayer(&gLayer1);
  gLayer0.SetOptimizer(OptimizerType::SGD);
  gLayer0.SetPsiOptimizer(OptimizerType::SGD);
  gLayer0.SetMuPCScale(0.5f);
  gLayer0.SetResidual(true); // valid: size == nextSize == 8

  std::mt19937 rngG(7);
  gLayer0.RandomizeWeights(rngG);

  gLayer0.ClampState(input);
  gLayer1.ClampState(target);

  bool allFinite = true;
  float firstEnergy = -1.0f, lastEnergy = -1.0f;

  for (int step = 0; step < 20; ++step)
  {
    gLayer0.ClampState(input);
    gLayer0.CalculateState(false);
    float e = gLayer1.CalculateState(true);
    if (step == 0)
      firstEnergy = e;
    lastEnergy = e;

    gLayer0.DirectFeedbackUpdate();
    gLayer0.UpdateState();
    gLayer0.UpdateWeights();

    if (!AllFinite(gLayer0.GetMu(), batchSize * nextSize) ||
        !AllFinite(gLayer0.GetBeliefs(), batchSize * size) ||
        !AllFinite(gLayer0.GetWeights(), size * nextSize) || !std::isfinite(e))
    {
      allFinite = false;
      printf("  step %d: NON-FINITE VALUE DETECTED\n", step);
      break;
    }
  }

  printf("First energy: %g, Last energy: %g\n", firstEnergy, lastEnergy);
  printf("All finite throughout: %s\n", allFinite ? "YES" : "NO");
  bool partBPass = allFinite && (lastEnergy < firstEnergy * 2.0f); // loose: not exploding
  printf("Part B: %s\n\n", partBPass ? "PASS" : "FAIL");

  bool allPass = partAPass && partBPass;
  printf("%s\n", allPass ? "PASS" : "FAIL");
  return allPass ? 0 : 1;
}
