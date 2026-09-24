/**
 * @file tCUDAIncrementCounterVerify.cpp
 * @brief Minimal isolation: does a SINGLE IncrementCounter() call on
 * GPU actually take effect at all? tCUDAAdamWStepVerify.cpp showed
 * t staying at 0 after 10 calls -- this checks whether even one call
 * works, separate from any loop or AdamWStep interaction, to narrow
 * down whether the bug is in IncrementCounter itself or something
 * about repeated calls / the later readback.
 */
#include <deepity/backend/Backend.h>
#include <cstdio>

using namespace Deep;

int main()
{
    auto gpuBackend = CreateBackend(DeviceType::DEVICE_GPU);

    int *dT = reinterpret_cast<int *>(gpuBackend->Allocate(1));
    int zero = 0;
    gpuBackend->CopyFromHost(reinterpret_cast<float *>(dT), reinterpret_cast<float *>(&zero), 1);

    int readback1 = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback1), reinterpret_cast<float *>(dT), 1);
    printf("After init (expect 0): %d\n", readback1);

    gpuBackend->IncrementCounter(dT);

    int readback2 = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback2), reinterpret_cast<float *>(dT), 1);
    printf("After ONE IncrementCounter call (expect 1): %d\n", readback2);

    gpuBackend->IncrementCounter(dT);
    gpuBackend->IncrementCounter(dT);

    int readback3 = -999;
    gpuBackend->CopyToHost(reinterpret_cast<float *>(&readback3), reinterpret_cast<float *>(dT), 1);
    printf("After THREE total IncrementCounter calls (expect 3): %d\n", readback3);

    bool pass = (readback1 == 0) && (readback2 == 1) && (readback3 == 3);
    printf("\n%s\n", pass ? "PASS" : "FAIL");

    gpuBackend->Free(reinterpret_cast<float *>(dT));
    return pass ? 0 : 1;
}