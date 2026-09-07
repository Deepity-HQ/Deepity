/**
 * @file tMatMulLargeAsymmetricVerify.cpp
 * @brief Reproduces, in isolation, the exact GEMM shape that crashed
 * DirectFeedbackUpdate() on Tiny ImageNet: transA=true, transB=true,
 * M=1024 (nextSize), N=12288 (size), K=250 (batchSize) -- K much
 * smaller than M/N, a shape tMatMulVerify.cpp's tiny (M=N=K=2-3) cases
 * never exercised, and one likely to select a different cuBLAS internal
 * kernel (the crash was inside ampere_sgemm_128x64_tt specifically).
 *
 * Differential test: CPU's GEMM path is already trusted (verified
 * correct in tMatMulVerify.cpp and used throughout tonight's CPU runs),
 * so any GPU disagreement at this specific scale isolates a real,
 * scale-dependent bug rather than a general transpose-logic error.
 */
#include <deepity/backend/Backend.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <random>

using namespace Deep;

int main()
{
    // Exact real dimensions from DirectFeedbackUpdate()'s second GEMM
    // on the Tiny ImageNet run (layer 0: size=12288, nextSize=1024,
    // batchSize=250).
    const int M = 1024;  // nextSize
    const int N = 12288; // size
    const int K = 250;   // batchSize

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // proj stored [K, M] (batchSize, nextSize) -- transA=true means
    // op(A) = A^T = [M, K].
    std::vector<float> hProj(K * M);
    for (auto &v : hProj)
        v = dist(rng);

    // zF stored [K, N] (batchSize, size) -- transB=true means
    // op(B) = B^T = [N, K]... wait, matching the real call: zF is
    // passed as B with transB=true, stored [K, N] per the same
    // batchSize-major convention as every other buffer in this codebase.
    std::vector<float> hZF(K * N);
    for (auto &v : hZF)
        v = dist(rng);

    // W stored [M, N] (nextSize, size), started at zero (beta=1.0 in
    // the real call accumulates onto existing weights; zero here isolates
    // just this GEMM's own contribution for comparison).
    std::vector<float> hWZero(M * N, 0.0f);

    float alpha = 0.5f; // arbitrary, matches fl/batchSize's role
    float beta = 1.0f;

    auto RunOn = [&](DeviceType device, const char *name) -> std::vector<float>
    {
        auto backend = CreateBackend(device);

        Tensor proj(backend.get(), device, hProj);
        Tensor zF(backend.get(), device, hZF);
        Tensor W(backend.get(), device, hWZero);

        backend->MatMul(
            /*transA=*/true, /*transB=*/true,
            M, N, K,
            alpha, proj.Data(), M,
            zF.Data(), N,
            beta, W.Data(), N);

        std::vector<float> result(M * N);
        W.CopyToHost(result.data());
        printf("[%s] done\n", name);
        return result;
    };

    printf("Running CPU reference...\n");
    std::vector<float> cpuResult = RunOn(DeviceType::DEVICE_CPU, "CPU");

    printf("Running GPU (this is where the real crash happened)...\n");
    std::vector<float> gpuResult = RunOn(DeviceType::DEVICE_GPU, "GPU");

    printf("Comparing...\n");
    int mismatches = 0;
    float maxDiff = 0.0f;
    for (size_t i = 0; i < cpuResult.size(); ++i)
    {
        float diff = std::fabs(cpuResult[i] - gpuResult[i]);
        maxDiff = std::max(maxDiff, diff);
        if (diff > 1e-2f)
            mismatches++;
    }

    printf("Max diff: %f, mismatches (>1e-2): %d / %zu\n", maxDiff, mismatches, cpuResult.size());

    if (mismatches > 0)
    {
        printf("FAILED: GPU MatMul disagrees with CPU at this exact real-world scale.\n");
        return 1;
    }

    printf("PASSED (though note: if this test PASSES but the real training run still\n");
    printf("crashes, the bug is NOT in MatMul itself -- look at buffer sizing/lifetime\n");
    printf("in DirectKPPCLayer's actual DirectFeedbackUpdate() call site instead.\n");
    return 0;
}