#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#if defined(_WIN32)
#include <malloc.h>
#endif

/**
 * @file MemoryArena.h
 * @brief A 64-byte-aligned bump-pointer allocator over a single
 * heap-allocated block, used to pack every layer's beliefs/errors/
 * weights/scratch buffers into one contiguous region for cache locality
 * (see the "Contiguous Arena Allocator" design note in the project
 * README). Host-side counterpart to DeviceMemoryArena.h's CUDA version.
 *
 * @note Individual chunks handed out by AllocateFloats() cannot be freed
 * independently -- the entire arena is released at once in the
 * destructor, matching the bump-pointer/no-reclaim design used
 * throughout this codebase's memory arenas.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    namespace detail
    {
        // std::aligned_alloc isn't reliably available on Windows/MinGW: the
        // standard requires memory from it be freed with plain free(), but
        // Windows' native aligned allocator needs the opposite pairing
        // (_aligned_malloc / _aligned_free), so many MinGW toolchains simply
        // omit std::aligned_alloc rather than violate the standard's contract.

        /// @brief Allocates a block of memory aligned to @p alignment bytes,
        /// using whichever aligned-allocation API is actually reliable on the
        /// current platform (see the comment above for why this can't just be
        /// std::aligned_alloc everywhere).
        /// @param alignment Required alignment, in bytes. Must be a valid
        /// alignment for the platform's underlying allocator (e.g. a power of
        /// two).
        /// @param size Number of bytes to allocate.
        /// @return Pointer to the allocated block, or nullptr on failure.
        /// @warning The pointer returned must be freed with
        /// AlignedFreePortable(), never plain free() or delete -- the
        /// underlying allocator differs by platform and the two must be
        /// paired correctly.
        inline void *AlignedAllocPortable(size_t alignment, size_t size)
        {
#if defined(_WIN32)
            return _aligned_malloc(size, alignment); // note: (size, alignment) -- reversed vs aligned_alloc
#else
            return std::aligned_alloc(alignment, size);
#endif
        }

        /// @brief Frees a block previously returned by AlignedAllocPortable().
        /// @param ptr Pointer to free; must have been returned by
        /// AlignedAllocPortable(), not by plain malloc()/new or any other
        /// allocator.
        inline void AlignedFreePortable(void *ptr)
        {
#if defined(_WIN32)
            _aligned_free(ptr);
#else
            std::free(ptr);
#endif
        }
    } // namespace detail

    /// @brief A 64-byte-aligned bump-pointer allocator over a single
    /// contiguous block of host memory. Every chunk handed out via
    /// AllocateFloats() is itself 64-byte aligned, satisfying this
    /// codebase's strict alignment requirement for AVX2/AVX-512 loads.
    class MemoryArena
    {
    private:
        /// @brief Base address of the underlying aligned allocation.
        float *base_ptr;
        /// @brief Total capacity of the arena, in bytes, rounded up to a
        /// multiple of 64.
        size_t capacity_bytes;
        /// @brief Current allocation offset from base_ptr, in bytes.
        /// Advances monotonically with each AllocateFloats() call and is
        /// never reset or reclaimed.
        size_t offset_bytes;

    public:
        /// @brief Allocates a single 64-byte-aligned buffer large enough
        /// to hold `total_floats` floats, rounded up to the nearest
        /// 64-byte boundary.
        /// @param total_floats Total number of floats this arena can
        /// hand out across all future AllocateFloats() calls combined.
        /// @throws std::bad_alloc if the underlying aligned allocation
        /// fails.
        MemoryArena(size_t total_floats)
        {
            // Guarantee total capacity is a multiple of 64 bytes
            capacity_bytes = (total_floats * sizeof(float) + 63) & ~63;
            base_ptr = static_cast<float *>(detail::AlignedAllocPortable(64, capacity_bytes));

            if (!base_ptr)
            {
                throw std::bad_alloc();
            }
            offset_bytes = 0;
        }

        /// @brief Frees the underlying aligned allocation, if any.
        ~MemoryArena()
        {
            if (base_ptr)
            {
                detail::AlignedFreePortable(base_ptr);
            }
        }

        // Delete copy/move constructors to prevent double-free corruption

        /// @brief Deleted: MemoryArena owns a single aligned allocation,
        /// so copying would risk a double-free.
        MemoryArena(const MemoryArena &) = delete;
        /// @brief Deleted: MemoryArena owns a single aligned allocation,
        /// so copy-assignment would risk a double-free.
        MemoryArena &operator=(const MemoryArena &) = delete;

        /// @brief Allocates a 64-byte aligned chunk of floats from the arena
        /// @param num_floats Number of floats to allocate from the arena.
        /// The actual reservation is rounded up to the nearest 64-byte
        /// boundary, so the returned chunk may have more usable capacity
        /// than requested.
        /// @return Pointer to the start of the allocated chunk, valid for
        /// the lifetime of this MemoryArena. 64-byte aligned.
        /// @throws std::runtime_error if the requested allocation would
        /// exceed the arena's total capacity.
        /// @warning Individual chunks are never freed independently; the
        /// entire arena is released at once in the destructor.
        float *AllocateFloats(size_t num_floats)
        {
            // Calculate how many bytes we need, padded to the nearest 64-byte boundary
            size_t allocation_size = (num_floats * sizeof(float) + 63) & ~63;

            if (offset_bytes + allocation_size > capacity_bytes)
            {
                throw std::runtime_error("Fatal: MemoryArena capacity exceeded during allocation.");
            }

            float *chunk = reinterpret_cast<float *>(
                reinterpret_cast<char *>(base_ptr) + offset_bytes);

            offset_bytes += allocation_size;
            return chunk;
        }

        /// @brief Returns how many bytes have been allocated from the
        /// arena so far.
        /// @return The current offset, in bytes.
        size_t GetUsedBytes() const { return offset_bytes; }
        /// @brief Returns the arena's total capacity.
        /// @return The capacity, in bytes, rounded up to a multiple of 64.
        size_t GetCapacityBytes() const { return capacity_bytes; }
    };
} // namespace Deep