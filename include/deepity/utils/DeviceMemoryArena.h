#pragma once
#if defined(DEEPITY_USE_CUDA)
#include <deepity/backend/IComputeBackend.h>
#include <new>
#include <stdexcept>
#include <iostream>

/**
 * @file DeviceMemoryArena.h
 * @brief A 64-byte-aligned bump-pointer allocator over a single device
 * buffer, mirroring this codebase's host-side MemoryArena, but backed by
 * IComputeBackend::Allocate()/Free() rather than raw cudaMalloc/cudaFree
 * directly -- the same allocation path Tensor already uses and
 * CUDAFunctionsVerify already exercises, rather than a second, separate,
 * unverified call site.
 *
 * Deliberately NOT routed through IComputeBackend on the CPU side
 * (MemoryArena stays exactly as it is): CPUBackend::Allocate() has no
 * huge-pages parameter, so wiring MemoryArena through it would mean
 * either silently losing that real, measured feature, or bolting a
 * CPU-only concept onto IComputeBackend's interface for no benefit,
 * since nothing ever needs to swap MemoryArena's allocator
 * polymorphically at runtime. The GPU side has no such feature to lose,
 * and the virtual call only happens once per arena (construction and
 * destruction), not per AllocateFloats() call -- a clean win with no
 * real cost.
 *
 * @warning Individual chunks handed out by AllocateFloats() cannot be
 * freed independently -- the entire arena is released at once in the
 * destructor, matching MemoryArena's bump-pointer/no-reclaim design.
 * @version 1.1
 * @date 2026-09-05
 */

namespace Deep
{
    /// @brief A 64-byte-aligned bump-pointer allocator over a single
    /// contiguous block of device memory, obtained via a supplied
    /// IComputeBackend (expected to be a CUDABackend in practice, though
    /// this class itself doesn't hard-require that specific type).
    class DeviceMemoryArena
    {
    private:
        /// @brief Backend used for the underlying allocation and its
        /// eventual release. Non-owning -- must outlive this arena.
        IComputeBackend *backend;
        /// @brief Base address of the underlying backend-allocated buffer.
        float *base_ptr;
        /// @brief Total capacity of the arena, in bytes, rounded up to a
        /// multiple of 64.
        size_t capacity_bytes;
        /// @brief Current allocation offset from base_ptr, in bytes.
        /// Advances monotonically with each AllocateFloats() call and is
        /// never reset or reclaimed.
        size_t offset_bytes;

    public:
        /// @brief Allocates a single 64-byte-aligned device buffer large
        /// enough to hold `total_floats` floats, rounded up to the
        /// nearest 64-byte boundary.
        /// @param backend The backend to allocate from (and later free
        /// through). Must outlive this DeviceMemoryArena. Passing a
        /// CPUBackend here would allocate host memory under a class
        /// named "device" -- that's a caller contract, not something
        /// this class checks at runtime, mirroring Tensor's own
        /// backend/device pairing contract.
        /// @param total_floats Total number of floats this arena can
        /// hand out across all future AllocateFloats() calls combined.
        /// @throws std::bad_alloc if the underlying allocation fails.
        DeviceMemoryArena(IComputeBackend *backend, size_t total_floats)
            : backend(backend)
        {
            capacity_bytes = (total_floats * sizeof(float) + 63) & ~(size_t)63;

            base_ptr = backend->Allocate(capacity_bytes / sizeof(float));
            if (!base_ptr)
            {
                throw std::bad_alloc();
            }
            offset_bytes = 0;
        }

        /// @brief Frees the underlying device buffer via the same
        /// backend it was allocated from.
        ~DeviceMemoryArena()
        {
            if (base_ptr)
            {
                backend->Free(base_ptr);
            }
        }

        // Delete copy/move constructors to prevent double-free corruption
        // -- the original version of this class had no such guard, a real
        // gap relative to MemoryArena's own established protection.

        /// @brief Deleted: DeviceMemoryArena owns a single device
        /// allocation, so copying would risk a double-free.
        DeviceMemoryArena(const DeviceMemoryArena &) = delete;
        /// @brief Deleted: DeviceMemoryArena owns a single device
        /// allocation, so copy-assignment would risk a double-free.
        DeviceMemoryArena &operator=(const DeviceMemoryArena &) = delete;

        /// @brief Allocates a 64-byte aligned chunk of floats from the
        /// arena.
        /// @param num_floats Number of floats to allocate from the
        /// arena. The actual reservation is rounded up to the nearest
        /// 64-byte boundary, matching MemoryArena's own guarantee (the
        /// original version of this class had no such rounding at all).
        /// @return Pointer to the start of the allocated chunk, valid
        /// for the lifetime of this DeviceMemoryArena. 64-byte aligned.
        /// @throws std::runtime_error if the requested allocation would
        /// exceed the arena's total capacity -- the original version of
        /// this class had no bounds checking at all.
        /// @warning Individual chunks are never freed independently; the
        /// entire arena is released at once in the destructor.
        float *AllocateFloats(size_t num_floats)
        {
            size_t allocation_size = (num_floats * sizeof(float) + 63) & ~(size_t)63;

            if (offset_bytes + allocation_size > capacity_bytes)
            {
                throw std::runtime_error("Fatal: DeviceMemoryArena capacity exceeded during allocation.");
            }

            float *chunk = reinterpret_cast<float *>(
                reinterpret_cast<char *>(base_ptr) + offset_bytes);

            offset_bytes += allocation_size;
            return chunk;
        }

        /// @brief Returns how many bytes have been allocated from the
        /// arena so far.
        size_t GetUsedBytes() const { return offset_bytes; }
        /// @brief Returns the arena's total capacity.
        size_t GetCapacityBytes() const { return capacity_bytes; }
    };
}
#endif