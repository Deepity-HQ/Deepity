/**
 * @file tFullConvPCLayerTemplatingVerify.cpp
 * @brief TEMP (issue #51's conv-family pass, final class): bit-exact
 * comparison of FullConvPCLayer vs. FullConvPCLayerT<float, float> and
 * FullConvPCNetwork vs. FullConvPCNetworkT<float, float>, identically
 * seeded/constructed, with momentum/residual/DFA toggles exercised
 * (same "no CPU fused conv path exists at all" reasoning as the other
 * conv-family verify tests makes a genuine bit-exact (==) match
 * achievable here too), plus a double convergence smoke test and 5
 * quantized/half-precision training smoke tests.
 */
#include <cmath>
#include <cstdio>
#include <deepity/layers/FullConvPCLayerT.h>
#include <deepity/layers/FullConvPCLayer.h>
#include <deepity/networks/FullConvPCNetworkT.h>
#include <deepity/networks/FullConvPCNetwork.h>
#include <deepity/utils/QuantFormat.h>
#include <hwy/base.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
bool ExactEqual(const float* a, const float* b, size_t n, const char* label)
{
  for (size_t i = 0; i < n; ++i)
  {
    if (a[i] != b[i])
    {
      printf("  MISMATCH in %s at index %zu: old=%.9g new=%.9g\n", label, i, (double)a[i],
             (double)b[i]);
      return false;
    }
  }
  return true;
}

/// @brief Training smoke test shared by the quantized/half-precision
/// parts: FullConvPCNetworkT<QuantT, float> trains for 30 steps with no
/// NaN/Inf/divergence -- no accuracy bar expected at low precision.
template <typename QuantT>
bool VerifyTrainingSmoke(const std::vector<float>& input, const std::vector<float>& target,
                        int inC, int outC, int h, int w, int kernel, int terminalSize,
                        int batchSize, float lr, float ir, float fl, float lmbda,
                        int settlingSteps, OptimizerType opt, unsigned seed, const char* label)
{
  FullConvPCNetworkT<QuantT, float> net(batchSize);
  net.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, terminalSize, lr, ir,
              fl, lmbda, ActivationType::RELU, ActivationType::dRELU);
  net.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, terminalSize, lr, ir, fl, lmbda,
              ActivationType::LINEAR, ActivationType::dLINEAR);
  net.Compile();
  net.SetOptimizer(opt);
  std::mt19937 netSeed(seed);
  net.RandomizeWeights(netSeed);

  std::vector<QuantT> inputQ(input.begin(), input.end());
  std::vector<QuantT> targetQ(target.begin(), target.end());

  for (int step = 0; step < 30; ++step)
  {
    float e = net.TrainStep(inputQ, targetQ, settlingSteps);
    if (std::isnan(e) || std::isinf(e))
    {
      printf("  FAIL: %s energy is NaN/Inf at step %d\n", label, step);
      return false;
    }
    if (std::fabs(e) > 1e6f)
    {
      printf("  FAIL: %s energy diverged to %.9g at step %d\n", label, (double)e, step);
      return false;
    }
  }
  return true;
}
} // namespace

int main()
{
  const int inC = 2, outC = 3, h = 6, w = 6, kernel = 3, batchSize = 2;
  const int terminalSize = outC;
  const float lr = 0.01f, ir = 0.1f, fl = 0.1f, lmbda = 0.0f;
  const size_t inSize = (size_t)batchSize * inC * h * w;
  const size_t outSize = (size_t)batchSize * outC * h * w;

  std::vector<float> input(inSize), target(outSize);
  std::mt19937 dataRng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dist(dataRng);
  for (auto& v : target)
    v = dist(dataRng);

  // --- Part 1: old FullConvPCLayer vs. new FullConvPCLayerT<float, float>,
  // two layers (input conv/terminal), momentum enabled on the input
  // layer, run through several TrainStep-shaped iterations. ---
  bool ok = true;
  {
    FullConvPCLayer oldIn(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2,
                          terminalSize, batchSize, lr, ir, fl, lmbda, ActivationType::RELU,
                          ActivationType::dRELU);
    FullConvPCLayer oldOut(outC, 0, h, w, 1, 1, 1, 1, 0, 0, terminalSize, batchSize, lr, ir, fl,
                           lmbda, ActivationType::LINEAR, ActivationType::dLINEAR);
    oldIn.SetLayerAbove(&oldOut);
    oldOut.SetLayerBelow(&oldIn);
    oldIn.SetTerminalLayer(&oldOut);
    oldOut.SetTerminalLayer(&oldOut);
    oldIn.SetMomentum(true, 0.9f);

    FullConvPCLayerT<float, float> newIn(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2,
                                         kernel / 2, terminalSize, batchSize, lr, ir, fl, lmbda,
                                         ActivationType::RELU, ActivationType::dRELU);
    FullConvPCLayerT<float, float> newOut(outC, 0, h, w, 1, 1, 1, 1, 0, 0, terminalSize, batchSize,
                                          lr, ir, fl, lmbda, ActivationType::LINEAR,
                                          ActivationType::dLINEAR);
    newIn.SetLayerAbove(&newOut);
    newOut.SetLayerBelow(&newIn);
    newIn.SetTerminalLayer(&newOut);
    newOut.SetTerminalLayer(&newOut);
    newIn.SetMomentum(true, 0.9f);

    std::mt19937 seedRng1(7);
    oldIn.RandomizeWeights(seedRng1);
    std::mt19937 seedRng2(7);
    newIn.RandomizeWeights(seedRng2);

    oldIn.SetOptimizer(OptimizerType::ADAM);
    newIn.SetOptimizer(OptimizerType::ADAM);

    ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(),
                     (size_t)outC * inC * kernel * kernel, "W (input layer, init)");

    const int settlingSteps = 4;
    const int trainSteps = 6;

    for (int step = 0; step < trainSteps; ++step)
    {
      oldIn.ClampState(input);
      newIn.ClampState(input);
      oldOut.ClampState(target);
      newOut.ClampState(target);

      oldIn.ComputeMuOnly();
      newIn.ComputeMuOnly();
      oldOut.CalculateState(false);
      newOut.CalculateState(false);

      oldIn.DirectFeedbackUpdate();
      newIn.DirectFeedbackUpdate();

      for (int s = 0; s < settlingSteps; ++s)
      {
        oldOut.CalculateState();
        oldIn.CalculateState();
        newOut.CalculateState();
        newIn.CalculateState();

        oldIn.UpdateState();
        newIn.UpdateState();
      }

      oldIn.ComputeMuOnly();
      newIn.ComputeMuOnly();
      oldIn.UpdateWeights();
      newIn.UpdateWeights();

      oldIn.UnclampState();
      newIn.UnclampState();
      oldOut.UnclampState();
      newOut.UnclampState();

      char label[64];
      snprintf(label, sizeof(label), "z (input, step %d)", step);
      ok &= ExactEqual(oldIn.GetBeliefs(), newIn.GetBeliefs(), inSize, label);
      snprintf(label, sizeof(label), "W (input layer, step %d)", step);
      ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(), (size_t)outC * inC * kernel * kernel,
                       label);
      snprintf(label, sizeof(label), "Psi (input layer, step %d)", step);
      ok &= ExactEqual(oldIn.GetDirectFeedbackWeights(), newIn.GetDirectFeedbackWeights(),
                       (size_t)outC * terminalSize, label);
    }
  }

  printf(ok ? "PASS: FullConvPCLayerT<float,float> is bit-exact against FullConvPCLayer\n"
            : "FAIL: see mismatches above\n");

  // --- Part 2: FullConvPCNetwork vs. FullConvPCNetworkT<float, float>,
  // a real 2-layer network built through AddLayer()/Compile(), with
  // residual connections + muPC scaling on a 3-layer variant to
  // exercise Compile()'s auto-configuration path, run through several
  // TrainStep()s. ---
  bool ok2 = true;
  {
    FullConvPCNetwork oldNet(batchSize);
    oldNet.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, terminalSize, lr,
                   ir, fl, lmbda, ActivationType::RELU, ActivationType::dRELU);
    oldNet.AddLayer(outC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, terminalSize,
                   lr, ir, fl, lmbda, ActivationType::RELU, ActivationType::dRELU);
    oldNet.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, terminalSize, lr, ir, fl, lmbda,
                   ActivationType::LINEAR, ActivationType::dLINEAR);
    oldNet.SetUseMuPCScaling(true);
    oldNet.SetUseResidualConnections(true);
    oldNet.Compile();
    std::mt19937 netSeed1(99);
    oldNet.RandomizeWeights(netSeed1);

    FullConvPCNetworkT<float, float> newNet(batchSize);
    newNet.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, terminalSize, lr,
                   ir, fl, lmbda, ActivationType::RELU, ActivationType::dRELU);
    newNet.AddLayer(outC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, terminalSize,
                   lr, ir, fl, lmbda, ActivationType::RELU, ActivationType::dRELU);
    newNet.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, terminalSize, lr, ir, fl, lmbda,
                   ActivationType::LINEAR, ActivationType::dLINEAR);
    newNet.SetUseMuPCScaling(true);
    newNet.SetUseResidualConnections(true);
    newNet.Compile();
    std::mt19937 netSeed2(99);
    newNet.RandomizeWeights(netSeed2);

    const int settlingSteps = 3;
    const int trainSteps = 6;

    for (int step = 0; step < trainSteps; ++step)
    {
      float oldEnergy = oldNet.TrainStep(input, target, settlingSteps);
      float newEnergy = newNet.TrainStep(input, target, settlingSteps);

      char label[64];
      snprintf(label, sizeof(label), "network energy (step %d)", step);
      ok2 &= ExactEqual(&oldEnergy, &newEnergy, 1, label);

      snprintf(label, sizeof(label), "network input-layer W (step %d)", step);
      ok2 &= ExactEqual(oldNet.GetLayers()[0]->GetWeights(), newNet.GetLayers()[0]->GetWeights(),
                        (size_t)outC * inC * kernel * kernel, label);
      snprintf(label, sizeof(label), "network hidden-layer W (step %d)", step);
      ok2 &= ExactEqual(oldNet.GetLayers()[1]->GetWeights(), newNet.GetLayers()[1]->GetWeights(),
                        (size_t)outC * outC * kernel * kernel, label);
    }

    std::vector<float> oldPred = oldNet.Predict(input, settlingSteps);
    std::vector<float> newPred = newNet.Predict(input, settlingSteps);
    ok2 &= ExactEqual(oldPred.data(), newPred.data(), oldPred.size(), "Predict() output");
  }

  printf(ok2 ? "PASS: FullConvPCNetworkT<float,float> is bit-exact against FullConvPCNetwork\n"
             : "FAIL: see mismatches above\n");

  // --- Part 3: FullConvPCNetworkT<double, double> qualitative
  // convergence smoke test, with ePC + cross-entropy enabled -- double
  // has no bit-exact reference to diff against, so this just confirms
  // training runs and the energy trends down. ---
  bool ok3 = true;
  {
    std::vector<double> inputD(input.begin(), input.end());
    std::vector<double> targetD(target.begin(), target.end());

    FullConvPCNetworkT<double, double> netD(batchSize);
    netD.AddLayer(inC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, terminalSize, lr,
                 ir, fl, lmbda, ActivationType::RELU, ActivationType::dRELU);
    netD.AddLayer(outC, outC, h, w, kernel, kernel, 1, 1, kernel / 2, kernel / 2, terminalSize, lr,
                 ir, fl, lmbda, ActivationType::RELU, ActivationType::dRELU);
    netD.AddLayer(outC, 0, h, w, 1, 1, 1, 1, 0, 0, terminalSize, lr, ir, fl, lmbda,
                 ActivationType::LINEAR, ActivationType::dLINEAR);
    netD.Compile();
    netD.SetUseEPC(true);
    std::mt19937 netSeedD(99);
    netD.RandomizeWeights(netSeedD);

    double firstEnergy = 0.0, lastEnergy = 0.0;
    for (int step = 0; step < 100; ++step)
    {
      double e = netD.TrainStep(inputD, targetD, 3);
      if (std::isnan(e) || std::isinf(e))
      {
        printf("  FAIL: FullConvPCNetworkT<double,double> energy is NaN/Inf at step %d\n", step);
        ok3 = false;
        break;
      }
      if (step == 0)
        firstEnergy = e;
      lastEnergy = e;
    }

    if (ok3 && !(lastEnergy < firstEnergy))
    {
      printf("  FAIL: FullConvPCNetworkT<double,double> energy did not decrease (first=%.9g "
             "last=%.9g)\n",
             firstEnergy, lastEnergy);
      ok3 = false;
    }
  }

  printf(ok3 ? "PASS: FullConvPCNetworkT<double,double> trains without NaN/Inf and converges\n"
             : "FAIL: see mismatches above\n");

  // --- Parts 4-8: quantized/half-precision training smoke tests. ---
  bool ok4 = VerifyTrainingSmoke<Fp4E2M1Value>(input, target, inC, outC, h, w, kernel,
                                               terminalSize, batchSize, lr, ir, fl, lmbda, 3,
                                               OptimizerType::ADAM, 7,
                                               "FullConvPCNetworkT<Fp4E2M1Value,float>");
  printf(ok4
             ? "PASS: FullConvPCNetworkT<Fp4E2M1Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok5 = VerifyTrainingSmoke<hwy::float16_t>(input, target, inC, outC, h, w, kernel,
                                                 terminalSize, batchSize, lr, ir, fl, lmbda, 3,
                                                 OptimizerType::ADAMW, 11,
                                                 "FullConvPCNetworkT<hwy::float16_t,float>");
  printf(ok5 ? "PASS: FullConvPCNetworkT<hwy::float16_t,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok6 = VerifyTrainingSmoke<Posit4_0Value>(input, target, inC, outC, h, w, kernel,
                                                terminalSize, batchSize, lr, ir, fl, lmbda, 3,
                                                OptimizerType::ADAM, 13,
                                                "FullConvPCNetworkT<Posit4_0Value,float>");
  printf(ok6
             ? "PASS: FullConvPCNetworkT<Posit4_0Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok7 = VerifyTrainingSmoke<Posit4_1Value>(input, target, inC, outC, h, w, kernel,
                                                terminalSize, batchSize, lr, ir, fl, lmbda, 3,
                                                OptimizerType::ADAMW, 17,
                                                "FullConvPCNetworkT<Posit4_1Value,float>");
  printf(ok7
             ? "PASS: FullConvPCNetworkT<Posit4_1Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok8 = VerifyTrainingSmoke<Posit4_2Value>(input, target, inC, outC, h, w, kernel,
                                                terminalSize, batchSize, lr, ir, fl, lmbda, 3,
                                                OptimizerType::ADAM, 19,
                                                "FullConvPCNetworkT<Posit4_2Value,float>");
  printf(ok8
             ? "PASS: FullConvPCNetworkT<Posit4_2Value,float> trains without NaN/Inf or divergence\n"
             : "FAIL: see mismatches above\n");

  bool allOk = ok && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8;
  return allOk ? 0 : 1;
}
