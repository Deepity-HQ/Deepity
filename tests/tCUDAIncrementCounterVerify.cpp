/**
 * @file tCUDAIncrementCounterVerify.cpp
 * @brief Combined test: does IncrementCounter() work on real GPU
 * hardware? Basic int* copy roundtrip already confirmed working
 * (tCUDAIntRoundtripVerify.cpp, PASS on a0344/A100). Every earlier
 * attempt at this exact test tonight ran on the OSC login node (no
 * GPU, Allocate() silently failing) -- this is the first genuine test
 * on real hardware.
 */
#include <deepity/backend/Backend.h>
#include <cuda_runtime.h>
#include <cstdio>

using namespace Deep;

int main()
{
    auto gpuBackend = CreateBackend(DeviceType::DEVICE_GPU);

    // --- Part 1: plain call, synchronous CopyToHost readback (same
    // pattern every other working kernel tonight used successfully) ---
    int *dT = reinterpret_cast<int *>(gpuBackend->Allocate(1));
    int zero = 0;
    gpuBackend->CopyFromHost(reinterpret_cast<float *>(dT), reinterpret_cast<float *>(&zero), 1);

    int readback1 = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback1), reinterpret_cast<float *>(dT), 1);
    printf("After init (expect 0): %d\n", readback1);

    gpuBackend->IncrementCounter(dT);

    int readback2 = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback2), reinterpret_cast<float *>(dT), 1);
    printf("After ONE IncrementCounter call, plain CopyToHost (expect 1): %d\n", readback2);

    // --- Part 2: same, but with explicit cudaDeviceSynchronize() ---
    gpuBackend->IncrementCounter(dT);
    cudaError_t syncErr = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize() after 2nd IncrementCounter: %s\n", cudaGetErrorString(syncErr));

    int readback3 = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback3), reinterpret_cast<float *>(dT), 1);
    printf("After TWO total calls + explicit sync (expect 2): %d\n", readback3);

    bool pass = (readback1 == 0) && (readback2 == 1) && (readback3 == 2);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    gpuBackend->Free(reinterpret_cast<float *>(dT));
    return pass ? 0 : 1;
}