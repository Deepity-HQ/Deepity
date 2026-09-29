/**
 * @file tFullPCEPCConvergenceVerify.cpp
 * @brief Demonstrates ePC's actual claim (arXiv:2505.20137): a deep
 * network settled with ePC reaches far lower energy in the SAME small
 * number of settling steps than plain one-hop local settling, because
 * ePC's backward sweep chains the loss gradient through every layer in
 * one pass instead of needing one settling step per hop of depth.
 *
 * Two identically-seeded 4-hidden-weight-layer (3 genuinely hidden)
 * networks, DKP's Psi contribution disabled (fl=0) on both so the
 * comparison isolates ePC's own effect, settled by hand (deliberately
 * NOT via TrainStep(), which calls ProjectForward() first -- that warm
 * -starts every hidden layer from a real forward pass through current
 * weights, already doing a lot of signal-decay's own job for a one-hop
 * network, unlike the paper's own zero-init experimental setup) for the
 * same few settling steps -- deliberately too few for a one-hop message
 * to have reached the bottom layer yet.
 */
#include <cmath>
#include <cstdio>
#include <deepity/networks/FullPCNetwork.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
void BuildNet(FullPCNetwork& net)
{
  // 6 genuinely hidden layers (7 weight-bearing total): deep enough that
  // a one-hop message clearly can't reach the bottom in the handful of
  // settling steps used below.
  for (int i = 0; i < 6; ++i)
    net.AddLayer(8, 8, 4, 0.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(8, 4, 4, 0.0f, 0.1f, 0.0f, 0.0f, ActivationType::TANH, ActivationType::dTANH);
  net.AddLayer(4, 0, 4, 0.0f, 0.1f, 0.0f, 0.0f, ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile(); // lr=0: this test only cares about settling, not weights
}
} // namespace

int main()
{
  const std::vector<float> x = {0.3f, -0.5f, 0.2f, 0.8f, -0.1f, 0.4f, -0.6f, 0.7f};
  const std::vector<float> y = {0.5f, -0.4f, 0.3f, 0.1f};
  const int inferenceSteps = 5; // deliberately small: a one-hop message
                                 // needs ~1 step per layer of depth to
                                 // reach the bottom of this 4-deep stack,
                                 // starting from zero-init (see the file
                                 // header comment on why ProjectForward()
                                 // is deliberately skipped below)

  FullPCNetwork netPlain(1), netEPC(1);
  BuildNet(netPlain);
  BuildNet(netEPC);
  netEPC.SetUseEPC(true);

  std::mt19937 rngA(17), rngB(17);
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

  // Loose bar (>=3x lower), not a tight bound: the point is demonstrating
  // the real, order-of-magnitude effect the paper describes, not pinning
  // an exact number to this specific architecture/seed.
  bool pass = std::isfinite(epcEnergy) && epcEnergy > 0.0f && epcEnergy < plainEnergy / 3.0f;
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
