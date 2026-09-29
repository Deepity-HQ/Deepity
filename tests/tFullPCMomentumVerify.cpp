/**
 * @file tFullPCMomentumVerify.cpp
 * @brief Exact, multi-step recurrence check for FullPCLayer's momentum
 * settling (SetMomentum), never exercised anywhere in the codebase
 * before this file.
 *
 * Unlike the finite-difference checks elsewhere in this test suite, the
 * documented update is a closed-form recurrence (FullPCLayer.h /
 * IComputeBackend.h's FusedStateUpdateMomentum docs):
 *   v = beta*v + (1-beta)*((feedback*deriv) - e)
 *   z += ir*v
 * so it's checked exactly (tight tolerance, not O(eps^2)), by
 * independently re-deriving `v`/`z` step by step from the layer's own
 * W/z/e/e_above buffers and comparing against the real UpdateState().
 *
 * Uses input(4) -> hidden(4) -> terminal(3): the hidden layer is the one
 * under test, since it's the only one with BOTH a nonzero bottom-up error
 * (against the input layer's mu) and top-down feedback (from the
 * terminal), matching how momentum settling is actually used in
 * training. `v` itself has no public getter, so the check instead starts
 * from ResetState()'s documented v=0 and tracks the same recurrence in
 * lockstep across several steps, rather than reading it back directly.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

int main()
{
  const float ir = 0.12f;
  const float beta = 0.75f;
  const float tolerance = 1e-4f; // exact recurrence, not finite-difference

  FullPCNetwork net(1);
  net.AddLayer(4, 4, 3, 0.3f, ir, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 3, 3, 0.3f, ir, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(3, 0, 3, 0.3f, ir, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetOptimizer(OptimizerType::SGD);
  net.SetUseMomentum(true, beta); // the thing under test (only the hidden
                                   // layer's UpdateState() actually reaches
                                   // the momentum branch; input/terminal
                                   // stay clamped throughout, so their
                                   // UpdateState() calls are no-ops
                                   // regardless of this flag)

  std::mt19937 rng(31);
  net.RandomizeWeights(rng);

  std::vector<float> x = {0.4f, -0.7f, 0.2f, 0.9f};
  std::vector<float> y = {0.5f, -0.3f, 0.1f};

  net.ResetState(); // v := 0, matching the recurrence's own starting point
  net.Clamp(x);
  net.GetTerminalLayer()->ClampState(y);
  net.ProjectForward();
  net.CalculateTerminalError();
  net.DirectFeedbackUpdate();

  FullPCLayer* hidden = net.GetLayers()[1].get();
  FullPCLayer* terminal = net.GetLayers()[2].get();
  size_t size = hidden->GetInputSize();     // 4
  size_t nextSize = hidden->GetOutputSize(); // 3
  const float* W = hidden->GetWeights();     // [nextSize, size]

  std::vector<float> vManual(size, 0.0f);

  const int steps = 5;
  float worstDiff = 0.0f;

  for (int t = 0; t < steps; ++t)
  {
    for (auto& l : net.GetLayers())
      l->CalculateState(false);

    std::vector<float> zBefore(size), eOwn(size), eAbove(nextSize);
    for (size_t i = 0; i < size; ++i)
    {
      zBefore[i] = hidden->GetBeliefs()[i];
      eOwn[i] = hidden->GetErrors()[i];
    }
    for (size_t j = 0; j < nextSize; ++j)
      eAbove[j] = terminal->GetErrors()[j];

    std::vector<float> zManual(size);
    for (size_t i = 0; i < size; ++i)
    {
      float t_ = std::tanh(zBefore[i]);
      float deriv = 1.0f - t_ * t_;

      float feedback = 0.0f;
      for (size_t j = 0; j < nextSize; ++j)
        feedback += eAbove[j] * W[j * size + i];

      float update = feedback * deriv - eOwn[i];
      vManual[i] = beta * vManual[i] + (1.0f - beta) * update;
      zManual[i] = zBefore[i] + ir * vManual[i];
    }

    for (auto& l : net.GetLayers())
      l->UpdateState();

    float stepDiff = 0.0f;
    for (size_t i = 0; i < size; ++i)
      stepDiff = std::max(stepDiff, std::fabs(zManual[i] - hidden->GetBeliefs()[i]));
    worstDiff = std::max(worstDiff, stepDiff);

    printf("  step %d: max abs diff (manual vs real z) %g\n", t, stepDiff);
  }

  printf("Worst step diff: %g\n", worstDiff);

  bool pass = worstDiff < tolerance;
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
