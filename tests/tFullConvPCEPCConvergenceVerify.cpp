/**
 * @file tFullConvPCEPCConvergenceVerify.cpp
 * @brief Conv analog of tFullPCEPCConvergenceVerify.cpp: a deep conv PC
 * stack settled with ePC should reach far lower energy in the same small
 * number of settling steps than plain one-hop local settling, same
 * signal-decay argument, now through Im2Col/Col2Im instead of dense
 * GEMMs.
 *
 * Settled by hand (not via TrainStep(), which calls ProjectForward()
 * first and would give plain settling an unfair warm start -- see the
 * dense version's identical reasoning).
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullConvPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
void BuildNet(FullConvPCNetwork& net)
{
  // input(4,8,8) -> 7 same-res hidden conv layers -> 1 collapsing hidden
  // layer -> terminal(4,1,1). 8 genuinely hidden layers total.
  net.AddLayer(4, 4, 8, 8, 3, 3, 1, 1, 1, 1, 4, 0.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH,
              ActivationType::dTANH);
  for (int i = 0; i < 6; ++i)
    net.AddLayer(4, 4, 8, 8, 3, 3, 1, 1, 1, 1, 4, 0.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH,
                ActivationType::dTANH);
  net.AddLayer(4, 4, 8, 8, 8, 8, 1, 1, 0, 0, 4, 0.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH,
              ActivationType::dTANH);
  net.AddLayer(4, 0, 1, 1, 1, 1, 1, 1, 0, 0, 4, 0.0f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();
}
} // namespace

int main()
{
  std::mt19937 dataRng(11);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float> x(4 * 8 * 8), y(4);
  for (auto& v : x)
    v = dist(dataRng);
  for (auto& v : y)
    v = dist(dataRng);

  const int inferenceSteps = 5;

  FullConvPCNetwork netPlain(1), netEPC(1);
  BuildNet(netPlain);
  BuildNet(netEPC);
  netEPC.SetUseEPC(true);

  std::mt19937 rngA(29), rngB(29);
  netPlain.RandomizeWeights(rngA);
  netEPC.RandomizeWeights(rngB);

  netPlain.ResetState();
  netPlain.Clamp(x);
  netPlain.GetTerminalLayer()->ClampState(y);
  netPlain.CalculateTerminalError();
  netPlain.DirectFeedbackUpdate();
  for (int t = 0; t < inferenceSteps; ++t)
    netPlain.Step(false);
  float plainEnergy = 0.0f;
  for (auto& l : netPlain.GetLayers())
    plainEnergy += l->CalculateState(true);

  netEPC.ResetState();
  netEPC.Clamp(x);
  netEPC.GetTerminalLayer()->ClampState(y);
  netEPC.CalculateTerminalError();
  netEPC.DirectFeedbackUpdate();
  for (int t = 0; t < inferenceSteps; ++t)
    netEPC.EPCStep();
  float epcEnergy = 0.0f;
  for (auto& l : netEPC.GetLayers())
    epcEnergy += l->CalculateState(true);

  printf("Plain one-hop settling, %d steps: final energy %.4f\n", inferenceSteps, plainEnergy);
  printf("ePC settling,           %d steps: final energy %.4f\n", inferenceSteps, epcEnergy);
  printf("Ratio (plain/ePC): %.2fx\n", plainEnergy / epcEnergy);

  bool pass = std::isfinite(epcEnergy) && epcEnergy > 0.0f && epcEnergy < plainEnergy / 3.0f;
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
