#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <numeric>
#include <RBLayer.h>
#include <iomanip>
#include <algorithm>

float MeanReconstructionError(const float *e_bu, size_t batchSize, size_t inSize)
{
    float total = 0.0f;
    for (size_t b = 0; b < batchSize; b++)
    {
        float norm = 0.0f;
        for (size_t i = 0; i < inSize; i++)
        {
            float v = e_bu[b * inSize + i];
            norm += v * v;
        }
        total += std::sqrt(norm / inSize);
    }
    return total / batchSize;
}

float EvalOnFixedSet(Deep::RBLayer &layer, const std::vector<float> &dataset,
                     size_t numSamples, size_t inputDim, size_t outputDim, size_t B)
{
    float totalErr = 0.0f;
    size_t count = 0;

    for (size_t i = 0; i + B <= numSamples; i += B)
    {
        const float *batch = dataset.data() + i * inputDim;

        // Zero r and converge
        std::fill(layer.GetBeliefs(), layer.GetBeliefs() + B * outputDim, 0.0f);
        for (int s = 0; s < 100; s++)
        {
            layer.CalcError(batch, nullptr, B);
            layer.UpdateBeliefs(batch, nullptr, B);
        }
        layer.CalcError(batch, nullptr, B);
        totalErr += MeanReconstructionError(layer.GetInferenceError(), B, inputDim);
        count++;
    }
    return totalErr / count;
}

int main(void)
{
    const size_t INPUT_DIM = 64;
    const size_t OUTPUT_DIM = 32;
    const size_t NUM_PATTERNS = 256;
    const size_t ITERS = 100000;
    const size_t LOG_EVERY = 5000;

    auto linear = [](float *, size_t) {};   // no-op: f(x) = x
    auto dLinear = [](float *x, size_t n) { // f'(x) = 1
        std::fill(x, x + n, 1.0f);
    };

    Deep::RBLayer layer(INPUT_DIM, OUTPUT_DIM,
                    /*var=*/1.0f, /*var_td=*/10.0f, /*alpha=*/0.01f,
                    /*k_1=*/0.001f, /*k_2=*/0.0001f, /*lmbda=*/0.0f,
                    /*batchSize=*/64, /*stepSize=*/30,
                    /*act=*/linear, /*dAct=*/dLinear);

    size_t B = layer.GetBatchSize();

    std::vector<float> fixedDataset(NUM_PATTERNS * INPUT_DIM);
    for (size_t p = 0; p < NUM_PATTERNS; p++)
    {
        float freq = 1.0f + (p % 16);
        float phase = (p / 16) * (M_PI / 8.0f);
        for (size_t i = 0; i < INPUT_DIM; i++)
        {
            fixedDataset[p * INPUT_DIM + i] = std::sin(freq * i * 2.0f * M_PI / INPUT_DIM + phase);
        }
    }

    std::mt19937 rng(42);
    std::vector<size_t> indices(NUM_PATTERNS);
    std::iota(indices.begin(), indices.end(), 0);

    std::vector<float> batch(B * INPUT_DIM);

    std::cout << "Fixed dataset: " << NUM_PATTERNS << " sinusoid patterns\n";
    std::cout << "Layer: " << INPUT_DIM << " -> " << OUTPUT_DIM << "\n";
    std::cout << "Task: layer should reconstruct all patterns from the fixed set\n\n";

    float baseline = EvalOnFixedSet(layer, fixedDataset, NUM_PATTERNS, INPUT_DIM, OUTPUT_DIM, B);
    std::cout << "Baseline error (untrained): " << baseline << "\n\n";

    std::cout << "Iter       | Train Error | vs Baseline | U norm\n";
    std::cout << "-----------+-------------+-------------+-------\n";

    float *U = layer.GetWeights();

    for (size_t iter = 0; iter < ITERS; iter++)
    {
        std::shuffle(indices.begin(), indices.end(), rng);
        for (size_t b = 0; b < B; b++)
        {
            size_t p = indices[b % NUM_PATTERNS];
            std::copy(fixedDataset.data() + p * INPUT_DIM,
                      fixedDataset.data() + p * INPUT_DIM + INPUT_DIM,
                      batch.data() + b * INPUT_DIM);
        }

        layer.RunInferenceStep(batch.data(), nullptr, B);

        if (iter % LOG_EVERY == 0)
        {
            float err = EvalOnFixedSet(layer, fixedDataset, NUM_PATTERNS, INPUT_DIM, OUTPUT_DIM, B);
            float pct = 100.0f * (baseline - err) / baseline;

            float uNorm = 0.0f;
            for (size_t i = 0; i < INPUT_DIM * OUTPUT_DIM; i++)
                uNorm += U[i] * U[i];

            std::cout << std::setw(10) << iter << " | "
                      << std::fixed << std::setprecision(6) << err << " | "
                      << std::setprecision(1) << pct << "% better"
                      << " | U norm: " << std::setprecision(4) << std::sqrt(uNorm) << "\n";
        }
    }

    std::cout << "\nDone.\n";
    return 0;
}