/**
 * @file tDirectKPPCLayerTemplatingVerify.cpp
 * @brief TEMP (issue #51's DirectKPPCLayer/DirectKPPCNetwork pass):
 * bit-exact comparison of DirectKPPCLayer vs. DirectKPPCLayerT<float,
 * float>, identically seeded/constructed, run through several real
 * TrainStep-shaped iterations. DirectKPPCLayer never calls any of
 * CPUBackend's oneDNN-fused Try* paths (confirmed by reading
 * DirectKPPCLayer.cpp -- every op goes through plain MatMul/
 * AddBiasBroadcast/ActivationDerivativeInto/etc.), and this class is
 * structurally the Psi/DFA subset of FullPCLayer, whose own templated
 * port (tFullPCLayerTemplatingVerify.cpp's Part 1) already achieved a
 * genuine bit-exact (==) match using the identical restructuring --
 * so this test holds DirectKPPCLayerT to the same bar.
 */
#include <cmath>
#include <cstdio>
#include <deepity/layers/DirectKPPCLayerT.h>
#include <deepity/layers/DirectKPPCLayer.h>
#include <deepity/networks/DirectKPPCNetworkT.h>
#include <deepity/networks/DirectKPPCNetwork.h>
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
/// parts: DirectKPPCNetworkT<QuantT, float> trains for 100 steps with
/// no NaN/Inf/divergence -- no accuracy bar expected at low precision.
template <typename QuantT>
bool VerifyTrainingSmoke(const std::vector<float>& input, const std::vector<float>& target,
                        size_t inSize, size_t hidSize, size_t outSize, size_t batchSize, float lr,
                        float ir, float fl, float lmbda, int settlingSteps, OptimizerType opt,
                        unsigned seed, const char* label)
{
  DirectKPPCNetworkT<QuantT, float> net(batchSize);
  net.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
              ActivationType::dSIGMOID);
  net.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();
  net.SetOptimizer(opt);
  net.SetPsiOptimizer(opt);
  std::mt19937 netSeed(seed);
  net.RandomizeWeights(netSeed);

  std::vector<QuantT> inputQ(input.begin(), input.end());
  std::vector<QuantT> targetQ(target.begin(), target.end());

  for (int step = 0; step < 100; ++step)
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
  const size_t inSize = 8, hidSize = 8, outSize = 4, batchSize = 4;
  const float lr = 0.01f, ir = 0.1f, fl = 0.001f, lmbda = 0.0f;

  std::vector<float> input(batchSize * inSize), target(batchSize * outSize);
  std::mt19937 dataRng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dist(dataRng);
  for (auto& v : target)
    v = dist(dataRng);

  // --- Part 1: old DirectKPPCLayer vs. new DirectKPPCLayerT<float,
  // float>, three layers (input/hidden/terminal, needed so
  // UpdateWeights()'s Psi path has a real terminalLayer to read from),
  // run through several TrainStep-shaped iterations. DirectFeedbackUpdate()
  // itself is exercised at the network level in Part 2 below, matching
  // tFullPCLayerTemplatingVerify.cpp's own Part 1/Part 2 split. ---
  bool ok = true;
  {
    DirectKPPCLayer oldIn(inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda,
                          ActivationType::LINEAR, ActivationType::dLINEAR);
    DirectKPPCLayer oldHid(hidSize, outSize, outSize, batchSize, lr, ir, fl, lmbda,
                           ActivationType::SIGMOID, ActivationType::dSIGMOID);
    DirectKPPCLayer oldOut(outSize, 0, outSize, batchSize, lr, ir, fl, lmbda,
                           ActivationType::LINEAR, ActivationType::dLINEAR);
    oldIn.SetLayerAbove(&oldHid);
    oldHid.SetLayerBelow(&oldIn);
    oldHid.SetLayerAbove(&oldOut);
    oldOut.SetLayerBelow(&oldHid);
    oldIn.SetTerminalLayer(&oldOut);
    oldHid.SetTerminalLayer(&oldOut);
    oldOut.SetTerminalLayer(&oldOut);

    DirectKPPCLayerT<float, float> newIn(inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda,
                                         ActivationType::LINEAR, ActivationType::dLINEAR);
    DirectKPPCLayerT<float, float> newHid(hidSize, outSize, outSize, batchSize, lr, ir, fl, lmbda,
                                          ActivationType::SIGMOID, ActivationType::dSIGMOID);
    DirectKPPCLayerT<float, float> newOut(outSize, 0, outSize, batchSize, lr, ir, fl, lmbda,
                                          ActivationType::LINEAR, ActivationType::dLINEAR);
    newIn.SetLayerAbove(&newHid);
    newHid.SetLayerBelow(&newIn);
    newHid.SetLayerAbove(&newOut);
    newOut.SetLayerBelow(&newHid);
    newIn.SetTerminalLayer(&newOut);
    newHid.SetTerminalLayer(&newOut);
    newOut.SetTerminalLayer(&newOut);

    std::mt19937 seedRng1(7);
    oldIn.RandomizeWeights(seedRng1);
    std::mt19937 seedRng1b(7);
    oldHid.RandomizeWeights(seedRng1b);

    std::mt19937 seedRng2(7);
    newIn.RandomizeWeights(seedRng2);
    std::mt19937 seedRng2b(7);
    newHid.RandomizeWeights(seedRng2b);

    oldIn.SetOptimizer(OptimizerType::ADAM);
    oldHid.SetOptimizer(OptimizerType::ADAMW);
    newIn.SetOptimizer(OptimizerType::ADAM);
    newHid.SetOptimizer(OptimizerType::ADAMW);

    ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(), inSize * hidSize,
                     "W (input layer, init)");

    const int settlingSteps = 5;
    const int trainSteps = 10;

    for (int step = 0; step < trainSteps; ++step)
    {
      oldIn.ClampState(input);
      newIn.ClampState(input);
      oldOut.ClampState(target);
      newOut.ClampState(target);

      for (int s = 0; s < settlingSteps; ++s)
      {
        oldOut.CalculateState();
        oldHid.CalculateState();
        oldIn.CalculateState();
        newOut.CalculateState();
        newHid.CalculateState();
        newIn.CalculateState();

        oldHid.UpdateState();
        oldIn.UpdateState();
        newHid.UpdateState();
        newIn.UpdateState();
      }

      oldHid.UpdateWeights();
      oldIn.UpdateWeights();
      newHid.UpdateWeights();
      newIn.UpdateWeights();

      oldIn.UnclampState();
      newIn.UnclampState();
      oldOut.UnclampState();
      newOut.UnclampState();

      char label[64];
      snprintf(label, sizeof(label), "z (input, step %d)", step);
      ok &= ExactEqual(oldIn.GetBeliefs(), newIn.GetBeliefs(), batchSize * inSize, label);
      snprintf(label, sizeof(label), "W (input layer, step %d)", step);
      ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(), inSize * hidSize, label);
      snprintf(label, sizeof(label), "b (hidden layer, step %d)", step);
      ok &= ExactEqual(oldHid.GetBiases(), newHid.GetBiases(), outSize, label);
    }
  }

  printf(ok ? "PASS: DirectKPPCLayerT<float,float> is bit-exact against DirectKPPCLayer\n"
            : "FAIL: see mismatches above\n");

  // --- Part 2: DirectKPPCNetwork vs. DirectKPPCNetworkT<float, float>,
  // a real 3-layer network built through AddLayer()/Compile(), run
  // through several TrainStep()s -- exercises ProjectForward()/
  // CalculateTerminalError()/DirectFeedbackUpdate() too. ---
  bool ok2 = true;
  {
    DirectKPPCNetwork oldNet(batchSize);
    oldNet.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    oldNet.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
                   ActivationType::dSIGMOID);
    oldNet.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    oldNet.Compile();
    std::mt19937 netSeed1(99);
    oldNet.RandomizeWeights(netSeed1);

    DirectKPPCNetworkT<float, float> newNet(batchSize);
    newNet.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    newNet.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
                   ActivationType::dSIGMOID);
    newNet.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    newNet.Compile();
    std::mt19937 netSeed2(99);
    newNet.RandomizeWeights(netSeed2);

    const int settlingSteps = 3;
    const int trainSteps = 10;

    for (int step = 0; step < trainSteps; ++step)
    {
      float oldEnergy = oldNet.TrainStep(input, target, settlingSteps);
      float newEnergy = newNet.TrainStep(input, target, settlingSteps);

      char label[64];
      snprintf(label, sizeof(label), "network energy (step %d)", step);
      ok2 &= ExactEqual(&oldEnergy, &newEnergy, 1, label);

      snprintf(label, sizeof(label), "network input-layer W (step %d)", step);
      ok2 &= ExactEqual(oldNet.GetLayers()[0]->GetWeights(), newNet.GetLayers()[0]->GetWeights(),
                        inSize * hidSize, label);
      snprintf(label, sizeof(label), "network hidden-layer Psi (step %d)", step);
      ok2 &= ExactEqual(oldNet.GetLayers()[1]->GetDirectFeedbackWeights(),
                        newNet.GetLayers()[1]->GetDirectFeedbackWeights(), hidSize * outSize,
                        label);
    }

    std::vector<float> oldPred = oldNet.Predict(input, settlingSteps);
    std::vector<float> newPred = newNet.Predict(input, settlingSteps);
    ok2 &= ExactEqual(oldPred.data(), newPred.data(), oldPred.size(), "Predict() output");
  }

  printf(ok2 ? "PASS: DirectKPPCNetworkT<float,float> is bit-exact against DirectKPPCNetwork\n"
             : "FAIL: see mismatches above\n");

  // --- Part 3: DirectKPPCNetworkT<double, double> qualitative
  // convergence smoke test -- double has no bit-exact reference to
  // diff against (DirectKPPCNetwork is float-only), so this just
  // confirms training runs and the energy trends down. ---
  bool ok3 = true;
  {
    std::vector<double> inputD(input.begin(), input.end());
    std::vector<double> targetD(target.begin(), target.end());

    DirectKPPCNetworkT<double, double> netD(batchSize);
    netD.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                 ActivationType::dLINEAR);
    netD.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
                 ActivationType::dSIGMOID);
    netD.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                 ActivationType::dLINEAR);
    netD.Compile();
    std::mt19937 netSeedD(99);
    netD.RandomizeWeights(netSeedD);

    double firstEnergy = 0.0, lastEnergy = 0.0;
    for (int step = 0; step < 200; ++step)
    {
      double e = netD.TrainStep(inputD, targetD, 3);
      if (std::isnan(e) || std::isinf(e))
      {
        printf("  FAIL: DirectKPPCNetworkT<double,double> energy is NaN/Inf at step %d\n", step);
        ok3 = false;
        break;
      }
      if (step == 0)
        firstEnergy = e;
      lastEnergy = e;
    }

    if (ok3 && !(lastEnergy < firstEnergy))
    {
      printf("  FAIL: DirectKPPCNetworkT<double,double> energy did not decrease (first=%.9g "
             "last=%.9g)\n",
             firstEnergy, lastEnergy);
      ok3 = false;
    }
  }

  printf(ok3 ? "PASS: DirectKPPCNetworkT<double,double> trains without NaN/Inf and converges\n"
             : "FAIL: see mismatches above\n");

  // --- Parts 4-8: quantized/half-precision training smoke tests, same
  // scope as tFullPCLayerTemplatingVerify.cpp's own Parts 5b/6b/7b/8b/9b. ---
  bool ok4 = VerifyTrainingSmoke<Fp4E2M1Value>(
      input, target, inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda, 3,
      OptimizerType::ADAM, 7, "DirectKPPCNetworkT<Fp4E2M1Value,float>");
  printf(ok4
             ? "PASS: DirectKPPCNetworkT<Fp4E2M1Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok5 = VerifyTrainingSmoke<hwy::float16_t>(
      input, target, inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda, 3,
      OptimizerType::ADAMW, 11, "DirectKPPCNetworkT<hwy::float16_t,float>");
  printf(ok5
             ? "PASS: DirectKPPCNetworkT<hwy::float16_t,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok6 = VerifyTrainingSmoke<Posit4_0Value>(
      input, target, inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda, 3,
      OptimizerType::ADAM, 13, "DirectKPPCNetworkT<Posit4_0Value,float>");
  printf(ok6
             ? "PASS: DirectKPPCNetworkT<Posit4_0Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok7 = VerifyTrainingSmoke<Posit4_1Value>(
      input, target, inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda, 3,
      OptimizerType::ADAMW, 17, "DirectKPPCNetworkT<Posit4_1Value,float>");
  printf(ok7
             ? "PASS: DirectKPPCNetworkT<Posit4_1Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool ok8 = VerifyTrainingSmoke<Posit4_2Value>(
      input, target, inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda, 3,
      OptimizerType::ADAM, 19, "DirectKPPCNetworkT<Posit4_2Value,float>");
  printf(ok8
             ? "PASS: DirectKPPCNetworkT<Posit4_2Value,float> trains without NaN/Inf or "
               "divergence\n"
             : "FAIL: see mismatches above\n");

  bool allOk = ok && ok2 && ok3 && ok4 && ok5 && ok6 && ok7 && ok8;
  return allOk ? 0 : 1;
}
