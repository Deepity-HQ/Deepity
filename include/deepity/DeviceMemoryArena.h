#pragma once
#if defined(DEEPITY_ENABLE_CUDA)
#include <cuda_runtime.h>

/**
 * @file DeviceMemoryArena.h
 * @brief A simple bump-pointer allocator over a single cudaMalloc'd
 * device buffer, mirroring this codebase's host-side MemoryArena but for
 * GPU memory.
 *
 * @warning No bounds checking: AllocateFloats() never verifies the
 * running offset stays within capacity_bytes, so over-allocating past the
 * arena's total_floats silently hands out a pointer past the end of the
 * cudaMalloc'd buffer. Callers are responsible for summing every
 * required allocation up front (mirroring how MemoryArena-based layers
 * report GetRequiredFloats() before BindMemory()).
 *
 * @warning cudaMalloc()'s return value is not checked in the constructor.
 * If the allocation fails, base_ptr is left null/uninitialized and every
 * subsequent AllocateFloats() call will silently return an invalid
 * pointer rather than failing loudly.
 *
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

/// @brief A bump-pointer allocator over a single contiguous block of CUDA
/// device memory. Only compiled when DEEPITY_ENABLE_CUDA is defined.
class DeviceMemoryArena
{
private:
    /// @brief Base address of the underlying cudaMalloc'd device buffer.
    float *base_ptr;
    /// @brief Total capacity of the arena, in bytes.
    size_t capacity_bytes;
    /// @brief Current allocation offset from base_ptr, in bytes. Advances
    /// monotonically with each AllocateFloats() call and is never reset
    /// or reclaimed.
    size_t offset_bytes;

public:
    /// @brief Allocates a single device buffer large enough to hold
    /// `total_floats` floats.
    /// @param total_floats Total number of floats this arena can hand
    /// out across all future AllocateFloats() calls combined.
    /// @warning Does not check cudaMalloc()'s return value -- see
    /// file-level warning.
    DeviceMemoryArena(size_t total_floats)
    {
        capacity_bytes = total_floats * sizeof(float);
        cudaMalloc(&base_ptr, capacity_bytes);
        offset_bytes = 0;
    }

    /// @brief Frees the underlying device buffer via cudaFree().
    ~DeviceMemoryArena()
    {
        cudaFree(base_ptr);
    }

    /// @brief Hands out a chunk of `num_floats` floats from the arena via
    /// simple pointer-bump allocation.
    /// @param num_floats Number of floats to allocate from the arena.
    /// @return Pointer to the start of the allocated chunk, valid for the
    /// lifetime of this DeviceMemoryArena.
    /// @warning No bounds checking against capacity_bytes -- see
    /// file-level warning. Individual chunks are also never freed
    /// independently; the entire arena is released at once in the
    /// destructor.
    float *AllocateFloats(size_t num_floats)
    {
        float *chunk = base_ptr + (offset_bytes / sizeof(float));
        offset_bytes += num_floats * sizeof(float);
        return chunk;
    }
};
#endif