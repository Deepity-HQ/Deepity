#pragma once
#include <cuda_runtime.h>

class DeviceMemoryArena {
private:
    float *base_ptr;
    size_t capacity_bytes;
    size_t offset_bytes;

public:
    DeviceMemoryArena(size_t total_floats) {
        capacity_bytes = total_floats * sizeof(float);
        cudaMalloc(&base_ptr, capacity_bytes);
        offset_bytes = 0;
    }

    ~DeviceMemoryArena() {
        cudaFree(base_ptr);
    }

    float *AllocateFloats(size_t num_floats) {
        float *chunk = base_ptr + (offset_bytes / sizeof(float));
        offset_bytes += num_floats * sizeof(float);
        return chunk;
    }
};