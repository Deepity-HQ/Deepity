/**
 * @file tFullPCMuPCInitVerify.cpp
 * @brief Statistical check that RandomizeWeights() actually implements
 * muPC's OTHER Table 1 change (unit-variance W init, b_l = 1), the half
 * of the parameterization that was missing before SetMuPCInit() existed
 * -- only the forward-scaling factor `a` was previously wired up.
 *
 * Uses wide layers (200/150/80) so the empirical variance of ~1e4-3e4 iid
 * Gaussian samples is a tight estimate of the true variance (relative
 * tolerance 15% is generous, not fragile, for that many samples), and
 * checks three things:
 *   1. With muPC scaling ON, W's sample variance is ~1.0 on every layer,
 *      not the fan-scaled 2/(fan_in+fan_out) plain-PC default.
 *   2. With muPC scaling OFF (the default), W's variance is unchanged
 *      from the plain-PC formula, i.e. this change is opt-in only.
 *   3. Psi (Deepity's own DFA addition, outside the muPC paper's scope)
 *      keeps the fan-scaled default EVEN WITH muPC scaling on.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
float SampleVariance(const float* buf, size_t n)
{
  double mean = 0.0;
  for (size_t i = 0; i < n; ++i)
    mean += buf[i];
  mean /= (double)n;

  double var = 0.0;
  for (size_t i = 0; i < n; ++i)
  {
    double d = buf[i] - mean;
    var += d * d;
  }
  return (float)(var / (double)n);
}

bool Close(float actual, float expected, float relTol)
{
  return std::fabs(actual - expected) <= relTol * expected;
}
} // namespace

int main()
{
  bool pass = true;

  // ============= Part A: muPC ON -> unit-variance W =============
  printf("=== Part A: muPC scaling ON -> W variance ~= 1.0 ===\n");
  {
    FullPCNetwork net(1);
    net.AddLayer(200, 150, 80, 0.1f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(150, 80, 80, 0.1f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(80, 0, 80, 0.1f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.SetUseMuPCScaling(true);
    net.Compile();

    std::mt19937 rng(3);
    net.RandomizeWeights(rng);

    for (size_t i = 0; i + 1 < net.GetLayers().size(); ++i)
    {
      FullPCLayer* l = net.GetLayers()[i].get();
      size_t n = l->GetInputSize() * l->GetOutputSize();
      float var = SampleVariance(l->GetWeights(), n);
      bool ok = Close(var, 1.0f, 0.15f);
      printf("  layer %zu: W variance %.4f (expected ~1.0) %s\n", i, var, ok ? "OK" : "FAIL");
      pass = pass && ok;

      // Psi must stay fan-scaled even though muPC init is on for W.
      size_t terminalSize = l->GetTerminalSize();
      float expectedPsiVar = 2.0f / (float)(l->GetInputSize() + terminalSize);
      float psiVar = SampleVariance(l->GetDirectFeedbackWeights(), l->GetInputSize() * terminalSize);
      bool psiOk = Close(psiVar, expectedPsiVar, 0.20f);
      printf("    Psi variance %.6f (expected ~%.6f) %s\n", psiVar, expectedPsiVar,
             psiOk ? "OK" : "FAIL");
      pass = pass && psiOk;
    }
  }

  // ============= Part B: muPC OFF -> plain fan-scaled W (unchanged) ==
  printf("\n=== Part B: muPC scaling OFF -> plain fan-scaled W (regression) ===\n");
  {
    FullPCNetwork net(1);
    net.AddLayer(200, 80, 80, 0.1f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
    net.AddLayer(80, 0, 80, 0.1f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
    net.Compile(); // useMuPCScaling stays false (default)

    std::mt19937 rng(3);
    net.RandomizeWeights(rng);

    FullPCLayer* l = net.GetLayers()[0].get();
    size_t n = l->GetInputSize() * l->GetOutputSize();
    float expected = 2.0f / (float)(l->GetInputSize() + l->GetOutputSize());
    float var = SampleVariance(l->GetWeights(), n);
    bool ok = Close(var, expected, 0.15f);
    printf("  layer 0: W variance %.6f (expected ~%.6f) %s\n", var, expected, ok ? "OK" : "FAIL");
    pass = pass && ok;
  }

  printf("\n%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
