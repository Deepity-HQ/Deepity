/**
 * @file tFullConvPCOtherTogglesSanity.cpp
 * @brief Smoke test (not a gradient check) for FullConvPCLayer's
 * remaining toggles -- iPC, momentum, muPC scaling, residual, DKP/Psi --
 * confirming each runs without crashing or producing non-finite values
 * when combined. ePC and cross-entropy are the two toggles this codebase
 * actually gradient-checks (tFullConvPCEPCGradientVerify.cpp); the rest
 * were ported by direct analogy to FullPCLayer's already-verified dense
 * formulas and are checked here only for basic sanity, matching how
 * much scrutiny each toggle's own risk warrants.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullConvPCNetwork.h>
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
} // namespace

int main()
{
  FullConvPCNetwork net(1);
  // input(3,6,6) -> hidden(3,6,6, same-pad, residual-eligible: H=2 so
  // this is layers[1], the only index the l=2..H residual loop covers
  // for this depth) -> collapsing hidden(4,1,1) -> terminal(4,1,1).
  net.AddLayer(3, 3, 6, 6, 3, 3, 1, 1, 1, 1, 4, 0.01f, 0.1f, 1e-3f, 1e-4f, ActivationType::TANH,
              ActivationType::dTANH);
  net.AddLayer(3, 3, 6, 6, 3, 3, 1, 1, 1, 1, 4, 0.01f, 0.1f, 1e-3f, 1e-4f, ActivationType::TANH,
              ActivationType::dTANH);
  net.AddLayer(3, 4, 6, 6, 6, 6, 1, 1, 0, 0, 4, 0.01f, 0.1f, 1e-3f, 1e-4f, ActivationType::TANH,
              ActivationType::dTANH);
  net.AddLayer(4, 0, 1, 1, 1, 1, 1, 1, 0, 0, 4, 0.01f, 0.1f, 1e-3f, 1e-4f, ActivationType::LINEAR,
              ActivationType::dLINEAR);

  net.SetUseMuPCScaling(true);
  net.SetUseResidualConnections(true);
  net.Compile();

  net.SetUseIPC(true);
  net.SetUseMomentum(true, 0.85f);
  net.SetUseCrossEntropy(true);
  net.SetOptimizer(OptimizerType::ADAMW);
  net.SetPsiOptimizer(OptimizerType::ADAMW);

  std::mt19937 rng(13);
  net.RandomizeWeights(rng);

  std::mt19937 dataRng(4);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x(3 * 6 * 6);
  std::vector<float> y = {0.001f, 0.001f, 0.997f, 0.001f};
  for (auto& v : x)
    v = dist(dataRng);

  bool allFinite = true;
  float lastEnergy = -1.0f;
  for (int step = 0; step < 25 && allFinite; ++step)
  {
    float e = net.TrainStep(x, y, 4);
    lastEnergy = e;

    allFinite = std::isfinite(e);
    for (auto& l : net.GetLayers())
    {
      if (l->GetOutChannels() > 0)
      {
        size_t wSize = (size_t)l->GetOutChannels() * l->GetInChannels() * l->GetKernelH() *
                       l->GetKernelW();
        allFinite = allFinite && AllFinite(l->GetWeights(), wSize);
      }
      allFinite = allFinite && AllFinite(l->GetBeliefs(), l->GetBatchSize() * l->GetInputSize());
    }

    net.DirectFeedbackUpdate(); // exercise Psi's own update path too
  }

  printf("Last energy: %g\n", lastEnergy);
  printf("All finite throughout: %s\n", allFinite ? "YES" : "NO");
  printf("%s\n", allFinite ? "PASS" : "FAIL");
  return allFinite ? 0 : 1;
}
