/**
 * @file tCUDAIm2ColVerify.cpp
 * @brief CPU-vs-GPU differential check for CUDABackend::Im2Col/Col2Im --
 * the two genuinely novel, hand-written CUDA kernels added tonight for
 * SimpleConvPCLayer's GPU port. Unlike the other four new backend
 * methods (MultiplyInto, Fill, RepackForBatchedGemm, AddBiasPerChannel,
 * all straightforward translations of existing, simple loops), these
 * two involve real per-thread index decomposition (row -> channel/kh/kw,
 * col -> oh/ow) with no prior verification at all -- "compiles" says
 * nothing about whether that arithmetic is actually correct on real
 * hardware. Same methodology as earlier tonight's CUDA differential
 * tests: identical input run through both backends, compared directly.
 *
 * Config: 3 channels, 6x6 input, 3x3 kernel, stride 1, pad 1 (output
 * stays 6x6) -- large enough to exercise real padding and multi-channel
 * behavior, small enough to run instantly.
 */
#include <deepity/backend/Backend.h>
#include <deepity/utils/Im2Col.h>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using namespace Deep;

int main()
{
    const int channels = 3, height = 6, width = 6;
    const int kH = 3, kW = 3, strideH = 1, strideW = 1, padH = 1, padW = 1;
    const int outH = ConvOutDim(height, kH, strideH, padH);
    const int outW = ConvOutDim(width, kW, strideW, padW);

    printf("outH=%d outW=%d (expected 6, 6)\n", outH, outW);

    const size_t inputSize = (size_t)channels * height * width;
    const size_t colRows = (size_t)channels * kH * kW;
    const size_t colCols = (size_t)outH * outW;
    const size_t colSize = colRows * colCols;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<float> hInput(inputSize);
    for (auto &v : hInput)
        v = dist(rng);

    // --- Im2Col: CPU ---
    auto cpuBackend = CreateBackend(DeviceType::DEVICE_CPU);
    std::vector<float> colsCpu(colSize);
    cpuBackend->Im2Col(hInput.data(), channels, height, width,
                       kH, kW, strideH, strideW, padH, padW, colsCpu.data());

    // --- Im2Col: GPU ---
    auto gpuBackend = CreateBackend(DeviceType::DEVICE_GPU);
    float *dInput = gpuBackend->Allocate(inputSize);
    float *dCols = gpuBackend->Allocate(colSize);
    gpuBackend->CopyFromHost(dInput, hInput.data(), inputSize);
    gpuBackend->Im2Col(dInput, channels, height, width,
                       kH, kW, strideH, strideW, padH, padW, dCols);

    std::vector<float> colsGpu(colSize);
    gpuBackend->CopyToHost(colsGpu.data(), dCols, colSize);

    float maxDiffIm2Col = 0.0f;
    for (size_t i = 0; i < colSize; ++i)
        maxDiffIm2Col = std::max(maxDiffIm2Col, std::fabs(colsCpu[i] - colsGpu[i]));

    printf("Im2Col max abs diff: %g\n", maxDiffIm2Col);
    bool im2colPass = maxDiffIm2Col < 1e-5f;
    printf("Im2Col: %s\n", im2colPass ? "PASS" : "FAIL");

    // --- Col2Im: CPU ---
    std::vector<float> hColumns(colSize);
    for (auto &v : hColumns)
        v = dist(rng);

    std::vector<float> outCpu(inputSize, 0.0f);
    cpuBackend->Col2Im(hColumns.data(), channels, height, width,
                       kH, kW, strideH, strideW, padH, padW, outCpu.data());

    // --- Col2Im: GPU ---
    float *dColumns = gpuBackend->Allocate(colSize);
    float *dOutput = gpuBackend->Allocate(inputSize);
    gpuBackend->CopyFromHost(dColumns, hColumns.data(), colSize);
    gpuBackend->Zero(dOutput, inputSize); // Col2Im ACCUMULATES -- must zero first, matching CPU's own contract

    gpuBackend->Col2Im(dColumns, channels, height, width,
                       kH, kW, strideH, strideW, padH, padW, dOutput);

    std::vector<float> outGpu(inputSize);
    gpuBackend->CopyToHost(outGpu.data(), dOutput, inputSize);

    float maxDiffCol2Im = 0.0f;
    for (size_t i = 0; i < inputSize; ++i)
        maxDiffCol2Im = std::max(maxDiffCol2Im, std::fabs(outCpu[i] - outGpu[i]));

    printf("Col2Im max abs diff: %g\n", maxDiffCol2Im);
    bool col2imPass = maxDiffCol2Im < 1e-4f; // slightly looser -- atomicAdd accumulation order can differ from CPU's SIMD order
    printf("Col2Im: %s\n", col2imPass ? "PASS" : "FAIL");

    gpuBackend->Free(dInput);
    gpuBackend->Free(dCols);
    gpuBackend->Free(dColumns);
    gpuBackend->Free(dOutput);

    bool allPassed = im2colPass && col2imPass;
    printf("\n%s\n", allPassed ? "PASS" : "FAIL");
    return allPassed ? 0 : 1;
}