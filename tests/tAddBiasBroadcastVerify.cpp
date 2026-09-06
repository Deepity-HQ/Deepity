/**
 * @file tAddBiasBroadcastVerify.cpp
 * @brief Verifies IComputeBackend::AddBiasBroadcast against an
 * independently, hand-computed expected result -- written specifically
 * because a live MNIST run regressed from ~97% accuracy to ~10% (exactly
 * chance) immediately after this function replaced a per-batch-row loop
 * of individual AxpyInto calls. Something in the new implementation is
 * mathematically wrong; this isolates exactly what.
 */
#include <deepity/backend/Backend.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace Deep;

namespace
{
    bool CheckClose(const std::vector<float> &actual, const std::vector<float> &expected,
                    const char *backendName)
    {
        bool ok = true;
        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (std::fabs(actual[i] - expected[i]) > 1e-4f)
            {
                printf("  [%s] MISMATCH at index %zu: got %.4f, expected %.4f\n",
                       backendName, i, actual[i], expected[i]);
                ok = false;
            }
        }
        return ok;
    }

    bool RunTest(DeviceType device, const char *backendName)
    {
        auto backend = CreateBackend(device);

        const size_t batchSize = 3, width = 4;
        std::vector<float> hBuf = {1, 2, 3, 4, 10, 20, 30, 40, 100, 200, 300, 400};
        std::vector<float> hBias = {0.1f, 0.2f, 0.3f, 0.4f};
        std::vector<float> expected = {1.1f, 2.2f, 3.3f, 4.4f,
                                       10.1f, 20.2f, 30.3f, 40.4f,
                                       100.1f, 200.2f, 300.3f, 400.4f};

        Tensor buf(backend.get(), device, hBuf);
        Tensor bias(backend.get(), device, hBias);

        backend->AddBiasBroadcast(buf.Data(), bias.Data(), batchSize, width);

        std::vector<float> out;
        buf.CopyToHost(out);

        bool ok = CheckClose(out, expected, backendName);
        printf("  [%s] AddBiasBroadcast: %s (got [%.1f, %.1f, %.1f, %.1f, ...])\n",
               backendName, ok ? "PASSED" : "FAILED", out[0], out[1], out[2], out[3]);
        return ok;
    }
}

int main()
{
    bool allPassed = true;

    printf("--- CPUBackend ---\n");
    allPassed &= RunTest(DeviceType::DEVICE_CPU, "CPU");

#ifdef DEEPITY_USE_CUDA
    printf("\n--- CUDABackend ---\n");
    allPassed &= RunTest(DeviceType::DEVICE_GPU, "CUDA");
#else
    printf("\n--- CUDABackend skipped (DEEPITY_USE_CUDA not defined) ---\n");
#endif

    if (!allPassed)
    {
        printf("\nFAILED: AddBiasBroadcast produced incorrect results on at least one backend.\n");
        return 1;
    }

    printf("\nPASSED: AddBiasBroadcast correct on all tested backends.\n");
    return 0;
}