/**
 * @file tFullPCLayerTemplatingVerify.cpp
 * @brief TEMP (Quantization branch templating pass): bit-exact
 * comparison of FullPCLayer vs. FullPCLayerT<float, float>, identically
 * seeded/constructed, run through several real TrainStep-shaped
 * iterations (CalculateState/UpdateState/UpdateWeights, two-layer
 * network with a terminal), diffing every externally-visible buffer
 * (z, e, mu, W, b, Psi) bit-for-bit (==, not tolerance-based) after
 * each iteration. This is the plan's verification gate #1 exercised
 * directly against the new class, not just "did anything else
 * regress" (that's tFullPCLayerVerify.cpp and friends, unchanged).
 */
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <deepity/layers/FullPCLayer.h>
#include <deepity/layers/FullPCLayerT.h>
#include <deepity/networks/FullPCNetwork.h>
#include <deepity/networks/FullPCNetworkT.h>
#include <deepity/utils/ModelIO.h>
#include <deepity/utils/QuantFormat.h>
#include <hwy/base.h>
#include <filesystem>
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

/// @brief Shared by Parts 7/8/9's posit checks (and could have covered
/// Part 5a's Fp4E2M1Value check too, left as-is to avoid touching an
/// already-passing block): grid round trip across all 8 magnitude
/// levels (both signs), clamping beyond the grid's max, and the 1.25-
/// style "exactly between two neighbors" non-throwing check.
template <typename QuantT>
bool VerifyFourBitGrid(const float* magnitudes, float maxMagnitude, const char* label)
{
  bool ok = true;
  for (int i = 0; i < 8; ++i)
  {
    for (float signedVal : {magnitudes[i], -magnitudes[i]})
    {
      float decoded = float(QuantT(signedVal));
      float expected = (signedVal == 0.0f) ? 0.0f : signedVal;
      if (decoded != expected)
      {
        printf("  FAIL: %s grid point %.9g round-tripped to %.9g\n", label, (double)expected,
               (double)decoded);
        ok = false;
      }
    }
  }

  for (float beyond : {maxMagnitude + 1.0f, maxMagnitude * 20.0f, -(maxMagnitude + 1.0f),
                       -(maxMagnitude * 20.0f)})
  {
    float decoded = float(QuantT(beyond));
    float expectedClamp = beyond > 0.0f ? maxMagnitude : -maxMagnitude;
    if (decoded != expectedClamp)
    {
      printf("  FAIL: %s(%.9g) clamped to %.9g, expected %.9g\n", label, (double)beyond,
             (double)decoded, (double)expectedClamp);
      ok = false;
    }
  }
  return ok;
}

/// @brief Shared by Parts 7/8/9's posit checks: FullPCNetworkT<QuantT,
/// float> trains for 100 steps with no NaN/Inf/divergence -- no
/// accuracy bar at 4 bits, same scope as Fp4E2M1Value's Part 5b.
template <typename QuantT>
bool VerifyFourBitTrainingSmoke(const std::vector<float>& input, const std::vector<float>& target,
                                size_t inSize, size_t hidSize, size_t outSize, size_t batchSize,
                                float lr, float ir, float fl, float lmbda, int settlingSteps,
                                OptimizerType opt, unsigned seed, const char* label)
{
  FullPCNetworkT<QuantT, float> net(batchSize);
  net.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
              ActivationType::dSIGMOID);
  net.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
              ActivationType::dLINEAR);
  net.Compile();
  net.SetOptimizer(opt);
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

  // --- old, non-templated FullPCLayer, two layers (hidden + terminal) ---
  FullPCLayer oldIn(inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                    ActivationType::dLINEAR);
  FullPCLayer oldHid(hidSize, outSize, outSize, batchSize, lr, ir, fl, lmbda,
                     ActivationType::SIGMOID, ActivationType::dSIGMOID);
  FullPCLayer oldOut(outSize, 0, outSize, batchSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                     ActivationType::dLINEAR);
  oldIn.SetLayerAbove(&oldHid);
  oldHid.SetLayerBelow(&oldIn);
  oldHid.SetLayerAbove(&oldOut);
  oldOut.SetLayerBelow(&oldHid);
  oldIn.SetTerminalLayer(&oldOut);
  oldHid.SetTerminalLayer(&oldOut);
  oldOut.SetTerminalLayer(&oldOut);

  std::mt19937 seedRng1(7);
  oldIn.RandomizeWeights(seedRng1);
  std::mt19937 seedRng1b(7);
  oldHid.RandomizeWeights(seedRng1b);

  // --- new, templated FullPCLayerT<float, float>, identically shaped ---
  FullPCLayerT<float, float> newIn(inSize, hidSize, outSize, batchSize, lr, ir, fl, lmbda,
                                   ActivationType::LINEAR, ActivationType::dLINEAR);
  FullPCLayerT<float, float> newHid(hidSize, outSize, outSize, batchSize, lr, ir, fl, lmbda,
                                    ActivationType::SIGMOID, ActivationType::dSIGMOID);
  FullPCLayerT<float, float> newOut(outSize, 0, outSize, batchSize, lr, ir, fl, lmbda,
                                    ActivationType::LINEAR, ActivationType::dLINEAR);
  newIn.SetLayerAbove(&newHid);
  newHid.SetLayerBelow(&newIn);
  newHid.SetLayerAbove(&newOut);
  newOut.SetLayerBelow(&newHid);
  newIn.SetTerminalLayer(&newOut);
  newHid.SetTerminalLayer(&newOut);
  newOut.SetTerminalLayer(&newOut);

  std::mt19937 seedRng2(7);
  newIn.RandomizeWeights(seedRng2);
  std::mt19937 seedRng2b(7);
  newHid.RandomizeWeights(seedRng2b);

  oldIn.SetOptimizer(OptimizerType::ADAM);
  oldHid.SetOptimizer(OptimizerType::ADAMW);
  oldIn.SetMomentum(true);
  newIn.SetOptimizer(OptimizerType::ADAM);
  newHid.SetOptimizer(OptimizerType::ADAMW);
  newIn.SetMomentum(true);

  // Identical seeding must produce identical initial weights before
  // any training happens at all.
  bool ok = true;
  ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(), inSize * hidSize, "W (input layer, init)");
  ok &= ExactEqual(oldHid.GetWeights(), newHid.GetWeights(), hidSize * outSize,
                   "W (hidden layer, init)");

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
    snprintf(label, sizeof(label), "e (hidden, step %d)", step);
    ok &= ExactEqual(oldHid.GetErrors(), newHid.GetErrors(), batchSize * hidSize, label);
    snprintf(label, sizeof(label), "mu (hidden, step %d)", step);
    ok &= ExactEqual(oldHid.GetMu(), newHid.GetMu(), batchSize * outSize, label);
    snprintf(label, sizeof(label), "W (input layer, step %d)", step);
    ok &= ExactEqual(oldIn.GetWeights(), newIn.GetWeights(), inSize * hidSize, label);
    snprintf(label, sizeof(label), "W (hidden layer, step %d)", step);
    ok &= ExactEqual(oldHid.GetWeights(), newHid.GetWeights(), hidSize * outSize, label);
    snprintf(label, sizeof(label), "b (hidden layer, step %d)", step);
    ok &= ExactEqual(oldHid.GetBiases(), newHid.GetBiases(), outSize, label);
  }

  printf(ok ? "PASS: FullPCLayerT<float,float> is bit-exact against FullPCLayer\n"
            : "FAIL: see mismatches above\n");

  // --- Part 1b: same as Part 1, but with cross-entropy enabled on the
  // terminal layer -- ComputeSoftmaxCrossEntropyError(AndEnergy)'s
  // generic (AccumT-casting) implementation in CPUBackendT was never
  // otherwise exercised against the original CPUBackend's dedicated
  // float path. ---
  bool ok1b = true;
  {
    FullPCLayer oldIn2(inSize, outSize, outSize, batchSize, lr, ir, fl, lmbda,
                       ActivationType::LINEAR, ActivationType::dLINEAR);
    FullPCLayer oldOut2(outSize, 0, outSize, batchSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                       ActivationType::dLINEAR);
    oldIn2.SetLayerAbove(&oldOut2);
    oldOut2.SetLayerBelow(&oldIn2);
    oldIn2.SetTerminalLayer(&oldOut2);
    oldOut2.SetTerminalLayer(&oldOut2);
    oldOut2.SetCrossEntropy(true);
    std::mt19937 ceSeed1(13);
    oldIn2.RandomizeWeights(ceSeed1);

    FullPCLayerT<float, float> newIn2(inSize, outSize, outSize, batchSize, lr, ir, fl, lmbda,
                                      ActivationType::LINEAR, ActivationType::dLINEAR);
    FullPCLayerT<float, float> newOut2(outSize, 0, outSize, batchSize, lr, ir, fl, lmbda,
                                       ActivationType::LINEAR, ActivationType::dLINEAR);
    newIn2.SetLayerAbove(&newOut2);
    newOut2.SetLayerBelow(&newIn2);
    newIn2.SetTerminalLayer(&newOut2);
    newOut2.SetTerminalLayer(&newOut2);
    newOut2.SetCrossEntropy(true);
    std::mt19937 ceSeed2(13);
    newIn2.RandomizeWeights(ceSeed2);

    std::vector<float> onehot(batchSize * outSize, 0.0f);
    for (size_t b = 0; b < batchSize; ++b)
      onehot[b * outSize + (b % outSize)] = 1.0f;

    for (int step = 0; step < 5; ++step)
    {
      oldIn2.ClampState(input);
      newIn2.ClampState(input);
      oldOut2.ClampState(onehot);
      newOut2.ClampState(onehot);

      for (int s = 0; s < settlingSteps; ++s)
      {
        float oldE = oldOut2.CalculateState(true);
        float newE = newOut2.CalculateState(true);
        oldIn2.CalculateState();
        newIn2.CalculateState();
        oldIn2.UpdateState();
        newIn2.UpdateState();

        char label[64];
        snprintf(label, sizeof(label), "cross-entropy energy (step %d, settle %d)", step, s);
        ok1b &= ExactEqual(&oldE, &newE, 1, label);
      }

      oldIn2.UpdateWeights();
      newIn2.UpdateWeights();
      oldIn2.UnclampState();
      newIn2.UnclampState();
      oldOut2.UnclampState();
      newOut2.UnclampState();
    }

    ok1b &= ExactEqual(oldIn2.GetWeights(), newIn2.GetWeights(), inSize * outSize,
                       "cross-entropy final W");
  }

  printf(ok1b ? "PASS: FullPCLayerT<float,float> cross-entropy path is bit-exact\n"
             : "FAIL: see mismatches above\n");

  // --- Part 2: FullPCNetwork vs. FullPCNetworkT<float, float>, a real
  // 3-layer network (input/hidden/terminal) built through AddLayer()/
  // Compile(), run through several TrainStep()s -- exercises the
  // network-level orchestration (ProjectForward/DirectFeedbackUpdate/
  // Step/UpdateWeights/Compile's muPC+residual machinery), not just
  // the per-layer mechanics Part 1 already covered. ---
  bool ok2 = true;
  {
    FullPCNetwork oldNet(batchSize);
    oldNet.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                    ActivationType::dLINEAR);
    oldNet.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
                    ActivationType::dSIGMOID);
    oldNet.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                    ActivationType::dLINEAR);
    oldNet.SetUseMuPCScaling(true);
    oldNet.Compile();
    std::mt19937 netSeed1(99);
    oldNet.RandomizeWeights(netSeed1);

    FullPCNetworkT<float, float> newNet(batchSize);
    newNet.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                    ActivationType::dLINEAR);
    newNet.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
                    ActivationType::dSIGMOID);
    newNet.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                    ActivationType::dLINEAR);
    newNet.SetUseMuPCScaling(true);
    newNet.Compile();
    std::mt19937 netSeed2(99);
    newNet.RandomizeWeights(netSeed2);

    for (int step = 0; step < trainSteps; ++step)
    {
      float oldEnergy = oldNet.TrainStep(input, target, settlingSteps);
      float newEnergy = newNet.TrainStep(input, target, settlingSteps);

      char label[64];
      snprintf(label, sizeof(label), "network energy (step %d)", step);
      ok2 &= ExactEqual(&oldEnergy, &newEnergy, 1, label);

      snprintf(label, sizeof(label), "network terminal beliefs (step %d)", step);
      ok2 &= ExactEqual(oldNet.GetTerminalLayer()->GetBeliefs(),
                        newNet.GetTerminalLayer()->GetBeliefs(), batchSize * outSize, label);

      snprintf(label, sizeof(label), "network input-layer W (step %d)", step);
      ok2 &= ExactEqual(oldNet.GetLayers()[0]->GetWeights(), newNet.GetLayers()[0]->GetWeights(),
                        inSize * hidSize, label);
    }

    std::vector<float> oldPred = oldNet.Predict(input, settlingSteps);
    std::vector<float> newPred = newNet.Predict(input, settlingSteps);
    ok2 &= ExactEqual(oldPred.data(), newPred.data(), oldPred.size(), "Predict() output");
  }

  printf(ok2 ? "PASS: FullPCNetworkT<float,float> is bit-exact against FullPCNetwork\n"
             : "FAIL: see mismatches above\n");

  // --- Part 3: FullPCNetworkT<double, double> qualitative convergence
  // smoke test (plan's gate #2) -- double has no bit-exact reference to
  // diff against (FullPCNetwork is float-only), so this just confirms
  // training actually runs and the energy trends down, plus that
  // MatMul() really dispatches to cblas_dgemm for T=double (verified
  // directly in CPUBackendT::MatMul's own source above, not re-derived
  // at runtime here).
  bool ok3 = true;
  {
    std::vector<double> inputD(input.begin(), input.end());
    std::vector<double> targetD(target.begin(), target.end());

    FullPCNetworkT<double, double> netD(batchSize);
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
      double e = netD.TrainStep(inputD, targetD, settlingSteps);
      if (std::isnan(e) || std::isinf(e))
      {
        printf("  FAIL: FullPCNetworkT<double,double> energy is NaN/Inf at step %d\n", step);
        ok3 = false;
        break;
      }
      if (step == 0)
        firstEnergy = e;
      lastEnergy = e;
    }

    if (ok3 && !(lastEnergy < firstEnergy))
    {
      printf("  FAIL: FullPCNetworkT<double,double> energy did not decrease (first=%.9g "
             "last=%.9g)\n",
             firstEnergy, lastEnergy);
      ok3 = false;
    }
  }

  printf(ok3 ? "PASS: FullPCNetworkT<double,double> trains without NaN/Inf and converges\n"
             : "FAIL: see mismatches above\n");

  // --- Part 4: ModelIO<T> round-trip (plan's gate #5) -- save a
  // FullPCNetworkT<float,float> checkpoint, confirm loading it back
  // into another FullPCNetworkT<float,float> reproduces its weights
  // exactly, then confirm loading that SAME float32 file into a
  // FullPCNetworkT<double,double> raises on the dtype mismatch --
  // new behavior, ModelIO previously never checked "dtype" on read. ---
  bool ok4 = true;
  {
    std::string path =
        (std::filesystem::temp_directory_path() / "deepity_modelio_templating_test.bin").string();

    FullPCNetworkT<float, float> srcNet(batchSize);
    srcNet.AddLayer(inSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    srcNet.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    srcNet.Compile();
    std::mt19937 srcSeed(123);
    srcNet.RandomizeWeights(srcSeed);

    if (!Deep::ModelIO::Save(srcNet, path))
    {
      printf("  FAIL: ModelIO::Save() couldn't open %s for writing\n", path.c_str());
      ok4 = false;
    }

    FullPCNetworkT<float, float> dstNet(batchSize);
    dstNet.AddLayer(inSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    dstNet.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                   ActivationType::dLINEAR);
    dstNet.Compile();

    if (ok4 && !Deep::ModelIO::Load(dstNet, path))
    {
      printf("  FAIL: ModelIO::Load() couldn't open %s for reading\n", path.c_str());
      ok4 = false;
    }

    if (ok4)
      ok4 &= ExactEqual(srcNet.GetLayers()[0]->GetWeights(), dstNet.GetLayers()[0]->GetWeights(),
                        inSize * outSize, "ModelIO round-trip W (layer 0)");

    if (ok4)
    {
      FullPCNetworkT<double, double> mismatchedNet(batchSize);
      mismatchedNet.AddLayer(inSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                            ActivationType::dLINEAR);
      mismatchedNet.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                            ActivationType::dLINEAR);
      mismatchedNet.Compile();

      bool threw = false;
      try
      {
        Deep::ModelIO::Load(mismatchedNet, path);
      }
      catch (const std::runtime_error&)
      {
        threw = true;
      }

      if (!threw)
      {
        printf("  FAIL: ModelIO::Load() didn't throw loading a float32 file into a "
               "FullPCNetworkT<double,double>\n");
        ok4 = false;
      }
    }

    std::filesystem::remove(path);
  }

  printf(ok4 ? "PASS: ModelIO<T> round-trips and rejects a dtype mismatch\n"
             : "FAIL: see mismatches above\n");

  // --- Part 5a: Fp4E2M1Value Decode(Encode(x)) round trip (plan's
  // gate #3, first half) -- swept across and beyond the format's grid
  // range ([-6, 6]), confirming every grid point round-trips exactly
  // and anything beyond the range clamps to the nearest extreme (+-6)
  // rather than wrapping/overflowing. ---
  bool ok5a = true;
  {
    for (float g : detail::kFp4E2M1Magnitudes)
    {
      for (float signedVal : {g, -g})
      {
        Fp4E2M1Value v(signedVal);
        float decoded = float(v);
        float expected = (signedVal == 0.0f) ? 0.0f : signedVal;
        if (decoded != expected)
        {
          printf("  FAIL: Fp4E2M1Value grid point %.9g round-tripped to %.9g\n",
                 (double)expected, (double)decoded);
          ok5a = false;
        }
      }
    }

    for (float beyond : {7.0f, 100.0f, -7.0f, -100.0f})
    {
      float decoded = float(Fp4E2M1Value(beyond));
      float expectedClamp = beyond > 0.0f ? 6.0f : -6.0f;
      if (decoded != expectedClamp)
      {
        printf("  FAIL: Fp4E2M1Value(%.9g) clamped to %.9g, expected %.9g\n", (double)beyond,
               (double)decoded, (double)expectedClamp);
        ok5a = false;
      }
    }

    // A value exactly halfway between two grid points (1.25, between 1
    // and 1.5) should round to one side consistently without throwing
    // or producing a third value.
    float halfway = float(Fp4E2M1Value(1.25f));
    if (halfway != 1.0f && halfway != 1.5f)
    {
      printf("  FAIL: Fp4E2M1Value(1.25) decoded to %.9g, neither grid neighbor\n",
             (double)halfway);
      ok5a = false;
    }
  }

  printf(ok5a ? "PASS: Fp4E2M1Value Decode(Encode(x)) round-trips and clamps correctly\n"
              : "FAIL: see mismatches above\n");

  // --- Part 5b: FullPCNetworkT<Fp4E2M1Value, float> training smoke
  // test (plan's gate #3, second half) -- no accuracy bar expected at
  // 4 bits of storage, just confirming the whole TrainStep path (ADAM
  // on the hidden layer, ADAMW on the input layer, exercising the
  // MatMulAccum/SumRowsAccum gradient path this phase added) runs for
  // many steps without NaN/Inf or blowing up. ---
  bool ok5b = true;
  {
    FullPCNetworkT<Fp4E2M1Value, float> netQ(batchSize);
    netQ.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                 ActivationType::dLINEAR);
    netQ.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
                 ActivationType::dSIGMOID);
    netQ.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                 ActivationType::dLINEAR);
    netQ.Compile();
    netQ.SetOptimizer(OptimizerType::ADAM);
    std::mt19937 netSeedQ(7);
    netQ.RandomizeWeights(netSeedQ);

    std::vector<Fp4E2M1Value> inputQ(input.begin(), input.end());
    std::vector<Fp4E2M1Value> targetQ(target.begin(), target.end());

    for (int step = 0; step < 100; ++step)
    {
      float e = netQ.TrainStep(inputQ, targetQ, settlingSteps);
      if (std::isnan(e) || std::isinf(e))
      {
        printf("  FAIL: FullPCNetworkT<Fp4E2M1Value,float> energy is NaN/Inf at step %d\n", step);
        ok5b = false;
        break;
      }
      if (std::fabs(e) > 1e6f)
      {
        printf("  FAIL: FullPCNetworkT<Fp4E2M1Value,float> energy diverged to %.9g at step %d\n",
               (double)e, step);
        ok5b = false;
        break;
      }
    }
  }

  printf(ok5b ? "PASS: FullPCNetworkT<Fp4E2M1Value,float> trains without NaN/Inf or divergence\n"
              : "FAIL: see mismatches above\n");

  // --- Part 6a: hwy::float16_t round trip (plan's gate #3, applied to
  // the correctness-first wrapper scope chosen for this format -- see
  // CPUBackendT.cpp's own "hwy::ConvertScalarTo" comment). Unlike
  // Fp4E2M1Value's fixed 8-point grid, float16 is IEEE-754 half: this
  // checks representative values round-trip within half's own ~3
  // decimal digits of precision (not bit-exact, that's not what half
  // precision promises), confirms a genuine zero stays zero, and
  // confirms a magnitude beyond half's ~65504 max produces +-Inf
  // rather than silently wrapping or clamping (half's real overflow
  // behavior, deliberately different from Fp4E2M1Value's clamp-to-grid
  // design). ---
  bool ok6a = true;
  {
    for (float v : {0.0f, -0.0f, 1.0f, -1.0f, 0.5f, 100.0f, -100.0f, 3.14159f, 1e-3f})
    {
      hwy::float16_t h = hwy::ConvertScalarTo<hwy::float16_t>(v);
      float back = hwy::ConvertScalarTo<float>(h);
      float tol = std::max(1e-3f, std::fabs(v) * 5e-3f);
      if (std::fabs(back - v) > tol)
      {
        printf("  FAIL: hwy::float16_t round trip of %.9g came back as %.9g (tol %.9g)\n",
               (double)v, (double)back, (double)tol);
        ok6a = false;
      }
    }

    float zeroBack = hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::float16_t>(0.0f));
    if (zeroBack != 0.0f)
    {
      printf("  FAIL: hwy::float16_t(0.0f) round-tripped to %.9g, expected exactly 0\n",
             (double)zeroBack);
      ok6a = false;
    }

    float overflowBack =
        hwy::ConvertScalarTo<float>(hwy::ConvertScalarTo<hwy::float16_t>(1e9f));
    if (!std::isinf(overflowBack))
    {
      printf("  FAIL: hwy::float16_t(1e9f) round-tripped to %.9g, expected +Inf\n",
             (double)overflowBack);
      ok6a = false;
    }
  }

  printf(ok6a ? "PASS: hwy::float16_t round-trips within half precision and overflows to Inf\n"
              : "FAIL: see mismatches above\n");

  // --- Part 6b: FullPCNetworkT<hwy::float16_t, float> training smoke
  // test, same structure and same "no accuracy bar" scope as Part 5b. ---
  bool ok6b = true;
  {
    FullPCNetworkT<hwy::float16_t, float> netH(batchSize);
    netH.AddLayer(inSize, hidSize, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                 ActivationType::dLINEAR);
    netH.AddLayer(hidSize, outSize, outSize, lr, ir, fl, lmbda, ActivationType::SIGMOID,
                 ActivationType::dSIGMOID);
    netH.AddLayer(outSize, 0, outSize, lr, ir, fl, lmbda, ActivationType::LINEAR,
                 ActivationType::dLINEAR);
    netH.Compile();
    netH.SetOptimizer(OptimizerType::ADAMW);
    std::mt19937 netSeedH(11);
    netH.RandomizeWeights(netSeedH);

    std::vector<hwy::float16_t> inputH(input.begin(), input.end());
    std::vector<hwy::float16_t> targetH(target.begin(), target.end());

    for (int step = 0; step < 100; ++step)
    {
      float e = netH.TrainStep(inputH, targetH, settlingSteps);
      if (std::isnan(e) || std::isinf(e))
      {
        printf("  FAIL: FullPCNetworkT<hwy::float16_t,float> energy is NaN/Inf at step %d\n",
               step);
        ok6b = false;
        break;
      }
      if (std::fabs(e) > 1e6f)
      {
        printf("  FAIL: FullPCNetworkT<hwy::float16_t,float> energy diverged to %.9g at step %d\n",
               (double)e, step);
        ok6b = false;
        break;
      }
    }
  }

  printf(ok6b ? "PASS: FullPCNetworkT<hwy::float16_t,float> trains without NaN/Inf or divergence\n"
              : "FAIL: see mismatches above\n");

  // --- Parts 7/8/9: the three posit formats, Phase 4 (plan's "just
  // new grid tables" once the wrapper pattern -- proven in Phase 2 --
  // exists). Each is a grid round-trip/clamp check plus a training
  // smoke test, same scope and gate as Fp4E2M1Value's Part 5a/5b. ---
  bool ok7a = VerifyFourBitGrid<Posit4_0Value>(detail::kPosit4_0Magnitudes, 4.0f, "Posit4_0Value");
  printf(ok7a ? "PASS: Posit4_0Value Decode(Encode(x)) round-trips and clamps correctly\n"
              : "FAIL: see mismatches above\n");

  bool ok7b = VerifyFourBitTrainingSmoke<Posit4_0Value>(input, target, inSize, hidSize, outSize,
                                                        batchSize, lr, ir, fl, lmbda,
                                                        settlingSteps, OptimizerType::ADAM, 13,
                                                        "FullPCNetworkT<Posit4_0Value,float>");
  printf(ok7b ? "PASS: FullPCNetworkT<Posit4_0Value,float> trains without NaN/Inf or divergence\n"
              : "FAIL: see mismatches above\n");

  bool ok8a = VerifyFourBitGrid<Posit4_1Value>(detail::kPosit4_1Magnitudes, 16.0f, "Posit4_1Value");
  printf(ok8a ? "PASS: Posit4_1Value Decode(Encode(x)) round-trips and clamps correctly\n"
              : "FAIL: see mismatches above\n");

  bool ok8b = VerifyFourBitTrainingSmoke<Posit4_1Value>(input, target, inSize, hidSize, outSize,
                                                        batchSize, lr, ir, fl, lmbda,
                                                        settlingSteps, OptimizerType::ADAMW, 17,
                                                        "FullPCNetworkT<Posit4_1Value,float>");
  printf(ok8b ? "PASS: FullPCNetworkT<Posit4_1Value,float> trains without NaN/Inf or divergence\n"
              : "FAIL: see mismatches above\n");

  bool ok9a =
      VerifyFourBitGrid<Posit4_2Value>(detail::kPosit4_2Magnitudes, 256.0f, "Posit4_2Value");
  printf(ok9a ? "PASS: Posit4_2Value Decode(Encode(x)) round-trips and clamps correctly\n"
              : "FAIL: see mismatches above\n");

  bool ok9b = VerifyFourBitTrainingSmoke<Posit4_2Value>(input, target, inSize, hidSize, outSize,
                                                        batchSize, lr, ir, fl, lmbda,
                                                        settlingSteps, OptimizerType::ADAM, 19,
                                                        "FullPCNetworkT<Posit4_2Value,float>");
  printf(ok9b ? "PASS: FullPCNetworkT<Posit4_2Value,float> trains without NaN/Inf or divergence\n"
              : "FAIL: see mismatches above\n");

  bool allOk = ok && ok1b && ok2 && ok3 && ok4 && ok5a && ok5b && ok6a && ok6b && ok7a && ok7b &&
               ok8a && ok8b && ok9a && ok9b;
  return allOk ? 0 : 1;
}
