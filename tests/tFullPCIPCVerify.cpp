/**
 * @file tFullPCIPCVerify.cpp
 * @brief Correctness check for FullPCNetwork's iPC mode (SetUseIPC), never
 * exercised anywhere in the codebase before this file: every existing test
 * and every example script only ever calls it with iPC off (or doesn't
 * call it at all).
 *
 * Part A: black-box confirmation that useIPC actually changes TrainStep()'s
 * behavior (not a dead flag) -- two identically-seeded networks, one with
 * iPC on and one off, given the same data, must diverge once
 * inferenceSteps > 1, since iPC applies multiple weight updates per
 * TrainStep() instead of one.
 *
 * Part B: white-box confirmation of the specific hazard FullPCLayer.h's
 * InvalidateMuCache() docs call out by name: "a clamped layer's cached mu
 * ... goes stale the moment UpdateWeights() runs mid-loop". This
 * reproduces TrainStep()'s own settleStep loop by hand (Step(false) +
 * UpdateWeights() + InvalidateMuCache(), exactly matching
 * FullPCNetwork.cpp) so it can snapshot the INPUT layer's cached mu
 * immediately after a mid-loop UpdateWeights() call but BEFORE
 * InvalidateMuCache() runs, proving that snapshot is stale relative to an
 * independent from-scratch recompute using the just-updated W -- i.e. that
 * InvalidateMuCache() is load-bearing, not a no-op.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
float MaxAbsDiff(const std::vector<float>& a, const std::vector<float>& b)
{
  float m = 0.0f;
  for (size_t i = 0; i < a.size(); ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}

// input(4) -> hidden(4) -> terminal(3), the smallest shape with a real
// (unclamped) hidden layer whose error depends on the input layer's mu.
void BuildNet(FullPCNetwork& net)
{
  net.AddLayer(4, 4, 3, 0.3f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 3, 3, 0.3f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(3, 0, 3, 0.3f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetOptimizer(OptimizerType::SGD);
}
} // namespace

int main()
{
  const std::vector<float> x = {0.4f, -0.7f, 0.2f, 0.9f};
  const std::vector<float> y = {0.5f, -0.3f, 0.1f};

  // ================= Part A: iPC vs non-iPC diverge =================
  printf("=== Part A: useIPC actually changes TrainStep() ===\n");

  FullPCNetwork netIPC(1), netNoIPC(1);
  BuildNet(netIPC);
  BuildNet(netNoIPC);
  netIPC.SetUseIPC(true);
  netNoIPC.SetUseIPC(false);

  std::mt19937 rngA(21), rngB(21);
  netIPC.RandomizeWeights(rngA);
  netNoIPC.RandomizeWeights(rngB);

  netIPC.TrainStep(x, y, /*inferenceSteps=*/5);
  netNoIPC.TrainStep(x, y, /*inferenceSteps=*/5);

  const float* wIPC = netIPC.GetLayers()[0]->GetWeights();
  const float* wNoIPC = netNoIPC.GetLayers()[0]->GetWeights();
  size_t wSize = netIPC.GetLayers()[0]->GetInputSize() * netIPC.GetLayers()[0]->GetOutputSize();

  float wDiff = 0.0f;
  for (size_t i = 0; i < wSize; ++i)
    wDiff = std::max(wDiff, std::fabs(wIPC[i] - wNoIPC[i]));

  printf("Layer0 W max abs diff (iPC vs non-iPC), same seed/data: %g\n", wDiff);
  bool partAPass = wDiff > 1e-4f; // must meaningfully diverge, not just float noise
  printf("Part A: %s\n\n", partAPass ? "PASS" : "FAIL");

  // ================= Part B: InvalidateMuCache() is load-bearing =====
  printf("=== Part B: mid-loop InvalidateMuCache() correctness ===\n");

  FullPCNetwork net(1);
  BuildNet(net);
  net.SetUseIPC(true);
  std::mt19937 rngC(21);
  net.RandomizeWeights(rngC);

  net.ResetState();
  net.Clamp(x);
  net.GetTerminalLayer()->ClampState(y);
  net.ProjectForward();
  net.CalculateTerminalError();
  net.DirectFeedbackUpdate();

  FullPCLayer* inputLayer = net.GetLayers()[0].get();
  size_t size = inputLayer->GetInputSize();
  size_t nextSize = inputLayer->GetOutputSize();

  // One warm-up settleStep first: right after ProjectForward(), the
  // hidden layer's z is seeded EQUAL to the input layer's mu, so its
  // bottom-up error (and therefore inputLayer's weight gradient) is
  // exactly zero on the very first step -- nothing would go stale yet.
  // UpdateState() needs one pass to pull hidden's z away from that
  // equilibrium before the staleness this test is checking for exists.
  net.Step(false);
  net.UpdateWeights();
  for (auto& l : net.GetLayers())
    l->InvalidateMuCache();

  bool partBPass = true;
  const int steps = 4;
  for (int t = 0; t < steps && partBPass; ++t)
  {
    net.Step(false);        // CalculateState + UpdateState on every layer
    net.UpdateWeights();    // iPC: applied every settling step

    // Snapshot the STALE cached mu: the cache is still marked valid from
    // before this step's UpdateWeights() call, so this replays the old
    // value rather than recomputing from the just-updated W.
    inputLayer->ComputeMuOnly();
    std::vector<float> staleMu(nextSize);
    for (size_t i = 0; i < nextSize; ++i)
      staleMu[i] = inputLayer->GetMu()[i];

    // What TrainStep() actually does at this point in its loop.
    for (auto& l : net.GetLayers())
      l->InvalidateMuCache();

    inputLayer->ComputeMuOnly();
    std::vector<float> freshMu(nextSize);
    for (size_t i = 0; i < nextSize; ++i)
      freshMu[i] = inputLayer->GetMu()[i];

    // Independent from-scratch reference: mu = tanh(z) @ W^T + b, using
    // the CURRENT (just-updated) W/b directly, matching FullPCLayer's
    // defaults (a=1.0, no residual) -- deliberately not reusing
    // ComputeMuOnly() at all.
    const float* z = inputLayer->GetBeliefs();
    const float* W = inputLayer->GetWeights(); // [nextSize, size]
    const float* b = inputLayer->GetBiases();  // [nextSize]
    std::vector<float> manualMu(nextSize);
    for (size_t j = 0; j < nextSize; ++j)
    {
      float acc = b[j];
      for (size_t i = 0; i < size; ++i)
        acc += std::tanh(z[i]) * W[j * size + i];
      manualMu[j] = acc;
    }

    float freshDiff = MaxAbsDiff(freshMu, manualMu);
    float staleVsFreshDiff = MaxAbsDiff(staleMu, freshMu);

    printf("  step %d: fresh-vs-manual diff %g, stale-vs-fresh diff %g\n", t, freshDiff,
           staleVsFreshDiff);

    // Invalidating must produce the mathematically correct mu ...
    bool freshCorrect = freshDiff < 1e-4f;
    // ... and that value must actually differ from what the (forgotten)
    // stale cache would have returned, proving the guard is necessary,
    // not just harmless.
    bool cacheWasStale = staleVsFreshDiff > 1e-4f;

    if (!freshCorrect || !cacheWasStale)
      partBPass = false;
  }

  printf("Part B: %s\n\n", partBPass ? "PASS" : "FAIL");

  bool allPass = partAPass && partBPass;
  printf("%s\n", allPass ? "PASS" : "FAIL");
  return allPass ? 0 : 1;
}
