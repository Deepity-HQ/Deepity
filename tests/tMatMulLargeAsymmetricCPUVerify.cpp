/**
 * @file tMatMulLargeAsymmetricCPUVerify.cpp
 * @brief CPU-only isolation of tMatMulLargeAsymmetricVerify.cpp's own
 * CPU reference call -- that test is gated behind DEEPITY_CUDA_ENABLED
 * (its whole point is a CPU-vs-GPU differential check), so it never
 * builds or runs on a CUDA-less machine, even though its CPU-side call
 * alone needs no GPU at all. Confirmed segfaulting on a plain CPU-only
 * build (reported during OSC GPU testing of unrelated work, then
 * reproduced here with no CUDA involved): backend->MatMul() with
 * transA=true, transB=true, M=1024, N=12288, K=250 crashes before
 * returning. Root cause not yet investigated.
 */
#include <deepity/backend/Backend.h>
#include <cstdio>
#include <vector>
#include <random>

using namespace Deep;

int main()
{
    const int M = 1024, N = 12288, K = 250;

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<float> hProj(K * M);
    for (auto &v : hProj) v = dist(rng);
    std::vector<float> hZF(K * N);
    for (auto &v : hZF) v = dist(rng);
    std::vector<float> hWZero(M * N, 0.0f);

    printf("Allocated host buffers OK.\n");

    auto backend = CreateBackend(DeviceType::DEVICE_CPU);
    Tensor proj(backend.get(), DeviceType::DEVICE_CPU, hProj);
    Tensor zF(backend.get(), DeviceType::DEVICE_CPU, hZF);
    Tensor W(backend.get(), DeviceType::DEVICE_CPU, hWZero);

    printf("Tensors constructed OK. Calling MatMul...\n");
    fflush(stdout);

    backend->MatMul(
        /*transA=*/true, /*transB=*/true,
        M, N, K,
        0.5f, proj.Data(), M,
        zF.Data(), N,
        1.0f, W.Data(), N);

    printf("MatMul returned OK.\n");

    std::vector<float> result(M * N);
    W.CopyToHost(result.data());
    printf("CopyToHost OK. result[0]=%f\n", result[0]);

    return 0;
}
