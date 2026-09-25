#include <cmath>
#include <cstdio>
#include <deepity/layers/FullPCLayer.h>
#include <random>
#include <vector>

using namespace Deep;

namespace
{
// Independent reference: plain, obvious, unoptimized softmax + CE.
// Deliberately NOT reusing any backend code.
void ReferenceSoftmaxCE(const std::vector<float>& mu, const std::vector<float>& target,
                        size_t batchSize, size_t width, std::vector<float>& expected_e,
                        float& expected_energy)
{
  expected_e.resize(batchSize * width);
  expected_energy = 0.0f;
  const float eps = 1e-8f;

  for (size_t b = 0; b < batchSize; ++b)
  {
    size_t base = b * width;
    float max_val = mu[base];
    for (size_t j = 1; j < width; ++j)
      max_val = std::max(max_val, mu[base + j]);

    std::vector<float> probs(width);
    float sum_exp = 0.0f;
    for (size_t j = 0; j < width; ++j)
    {
      probs[j] = std::exp(mu[base + j] - max_val);
      sum_exp += probs[j];
    }
    for (size_t j = 0; j < width; ++j)
      probs[j] /= sum_exp;

    for (size_t j = 0; j < width; ++j)
    {
      expected_energy -= target[base + j] * std::log(probs[j] + eps);
      expected_e[base + j] = target[base + j] - probs[j];
    }
  }
}

float MaxAbsDiff(const std::vector<float>& a, const float* b, size_t n)
{
  float m = 0.0f;
  for (size_t i = 0; i < n; ++i)
    m = std::max(m, std::fabs(a[i] - b[i]));
  return m;
}
} // namespace

int main()
{
  const size_t size = 6, nextSize = 4, terminalSize = 4, batchSize = 5;
  const float lr = 0.01f, ir = 0.1f, fl = 0.001f, lmbda = 0.0f;

  FullPCLayer layer0(size,
                     nextSize,
                     terminalSize,
                     batchSize,
                     lr,
                     ir,
                     fl,
                     lmbda,
                     ActivationType::TANH,
                     ActivationType::dTANH);
  FullPCLayer layer1(nextSize,
                     0,
                     terminalSize,
                     batchSize,
                     lr,
                     ir,
                     fl,
                     lmbda,
                     ActivationType::LINEAR,
                     ActivationType::dLINEAR);
  layer0.SetLayerAbove(&layer1);
  layer1.SetLayerBelow(&layer0);
  layer0.SetTerminalLayer(&layer1);
  layer1.SetTerminalLayer(&layer1);
  layer1.SetCrossEntropy(true); // the thing under test

  std::mt19937 rng(99);
  layer0.RandomizeWeights(rng);

  std::vector<float> input(batchSize * size);
  std::mt19937 dataRng(7);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : input)
    v = dist(dataRng);

  // One-hot-ish target (a real classification-style target, not
  // arbitrary noise -- matches how this feature will actually be used)
  std::vector<float> target(batchSize * terminalSize, 0.001f);
  for (size_t b = 0; b < batchSize; ++b)
    target[b * terminalSize + (b % terminalSize)] = 0.999f;

  layer0.ClampState(input);
  layer1.ClampState(target);

  // Let the layer compute mu naturally -- no manual control over its
  // value, so the reference computation below is checked against a
  // real, network-produced mu, not a hand-picked convenient one.
  layer0.CalculateState(false);

  // Read back the real mu and independently verify it's finite
  // before trusting it as the reference computation's input.
  std::vector<float> mu(batchSize * nextSize);
  // GetMu() is const float*; copy out via the public accessor path
  // used elsewhere in this codebase.
  const float* muPtr = layer0.GetMu();
  for (size_t i = 0; i < mu.size(); ++i)
    mu[i] = muPtr[i];

  bool muFinite = true;
  for (float v : mu)
    if (!std::isfinite(v))
      muFinite = false;
  printf("mu finite: %s\n", muFinite ? "YES" : "NO");

  std::vector<float> expected_e;
  float expected_energy;
  ReferenceSoftmaxCE(mu, target, batchSize, terminalSize, expected_e, expected_energy);

  float actual_energy = layer1.CalculateState(true);
  const float* actual_e = layer1.GetErrors();

  float eDiff = MaxAbsDiff(expected_e, actual_e, batchSize * terminalSize);
  float energyDiff = std::fabs(expected_energy - actual_energy);

  printf("Expected energy: %g\n", expected_energy);
  printf("Actual energy:   %g\n", actual_energy);
  printf("Energy diff:     %g\n", energyDiff);
  printf("e max abs diff:  %g\n", eDiff);

  bool pass = muFinite && (eDiff < 1e-4f) && (energyDiff < 1e-3f);
  printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}