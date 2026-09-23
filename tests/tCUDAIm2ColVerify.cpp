/**
 * @file tCUDAIncrementCounterSyncVerify.cpp
 * @brief Same as tCUDAIncrementCounterVerify.cpp, but with an explicit
 * cudaDeviceSynchronize() inserted right after IncrementCounter(),
 * before the readback. IncrementCounterKernel is launched as
 * <<<1,1,0,stream>>> -- exactly one block, one thread -- structurally
 * different from every other kernel tested tonight (all used a real
 * blocks/BLOCK_SIZE calculation with many threads). Testing whether
 * this specific, minimal launch configuration has a genuine
 * stream-timing issue the synchronous CopyToHost isn't actually
 * catching, despite that same pattern working for every other kernel.
 */
#include <deepity/backend/Backend.h>
#include <cuda_runtime.h>
#include <cstdio>

using namespace Deep;

int main()
{
    auto gpuBackend = CreateBackend(DeviceType::DEVICE_GPU);

    int *dT = reinterpret_cast<int *>(gpuBackend->Allocate(1));
    int zero = 0;
    gpuBackend->CopyFromHost(reinterpret_cast<float *>(dT), reinterpret_cast<float *>(&zero), 1);

    gpuBackend->IncrementCounter(dT);

    // THE ONLY CHANGE from tCUDAIncrementCounterVerify.cpp:
    cudaError_t syncErr = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize() after IncrementCounter: %s\n", cudaGetErrorString(syncErr));

    int readback = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback), reinterpret_cast<float *>(dT), 1);
    printf("After ONE IncrementCounter call + explicit sync (expect 1): %d\n", readback);

    bool pass = (readback == 1);
    printf("\n%s\n", pass ? "PASS (explicit sync fixed it -- real stream-timing bug found)"
                          : "FAIL (still wrong even with explicit sync -- bug is elsewhere)");

    gpuBackend->Free(reinterpret_cast<float *>(dT));
    return pass ? 0 : 1;
}