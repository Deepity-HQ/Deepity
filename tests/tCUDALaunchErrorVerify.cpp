/**
 * @file tCUDALaunchErrorVerify.cpp
 * @brief Checks cudaGetLastError() immediately after IncrementCounter's
 * kernel launch, before any synchronization -- cudaDeviceSynchronize()
 * reported "no error" in tCUDAIncrementCounterVerify.cpp, but that only
 * confirms EXECUTION had no fault; a silent LAUNCH-time failure (e.g.
 * an invalid stream handle from CUDABackend's constructor, which calls
 * cudaStreamCreate() with no error checking at all) can report
 * differently and might not surface the same way through
 * cudaDeviceSynchronize() alone.
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

    // Clear any pre-existing sticky error state before the launch we
    // actually care about.
    cudaGetLastError();

    gpuBackend->IncrementCounter(dT);

    // Check IMMEDIATELY, before any sync -- catches a launch-time error
    // (invalid stream, invalid configuration, etc.) directly.
    cudaError_t launchErr = cudaGetLastError();
    printf("cudaGetLastError() immediately after launch: %s (%d)\n",
           cudaGetErrorString(launchErr), (int)launchErr);

    cudaError_t syncErr = cudaDeviceSynchronize();
    printf("cudaDeviceSynchronize() after launch: %s (%d)\n",
           cudaGetErrorString(syncErr), (int)syncErr);

    int readback = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback), reinterpret_cast<float *>(dT), 1);
    printf("Readback (expect 1): %d\n", readback);

    // Separately: does a fresh, independent cudaStreamCreate() work at
    // all on this node/driver, outside the backend entirely? Rules out
    // "something about this specific node's CUDA context is broken" as
    // an explanation distinct from CUDABackend's own stream handling.
    cudaStream_t testStream;
    cudaError_t streamErr = cudaStreamCreate(&testStream);
    printf("\nIndependent cudaStreamCreate() test: %s (%d)\n",
           cudaGetErrorString(streamErr), (int)streamErr);
    if (streamErr == cudaSuccess)
        cudaStreamDestroy(testStream);

    return (readback == 1) ? 0 : 1;
}