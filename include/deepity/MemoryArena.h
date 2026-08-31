#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>
#if defined(_WIN32)
#include <malloc.h>
#include <windows.h>
#else
#include <sys/mman.h>
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

        /// @brief Attempts to allocate @p size bytes backed by huge pages.
        /// This is genuinely opt-in and best-effort: on Linux, requests
        /// Transparent Huge Pages via madvise(MADV_HUGEPAGE) on a normal
        /// anonymous mmap -- this is a HINT to the kernel, not a
        /// guarantee, and requires zero system-level configuration
        /// (unlike explicit hugetlbfs, which needs pages pre-reserved via
        /// /proc/sys/vm/nr_hugepages). On Windows, requests
        /// MEM_LARGE_PAGES via VirtualAlloc, which requires the process
        /// to hold SeLockMemoryPrivilege -- NOT granted by default, so
        /// this will typically fail unless explicitly configured by an
        /// administrator.
        /// @param size Number of bytes to allocate.
        /// @param[out] succeeded Set to true if huge-page-backed memory
        /// was actually obtained, false otherwise. Never throws on
        /// failure -- callers should fall back to AlignedAllocPortable()
        /// when this is false.
        /// @return Pointer to the allocated block if succeeded is true;
        /// nullptr otherwise.
        /// @warning A pointer returned here (when succeeded=true) must be
        /// freed with HugePageFree(), never AlignedFreePortable() or
        /// plain free() -- the underlying allocator (mmap/VirtualAlloc)
        /// is incompatible with aligned_alloc's pairing.
        inline void *HugePageAllocPortable(size_t size, bool &succeeded)
        {
            succeeded = false;
#if defined(_WIN32)
            void *ptr = VirtualAlloc(nullptr, size, MEM_RESERVE | MEM_COMMIT | MEM_LARGE_PAGES, PAGE_READWRITE);
            if (ptr)
                succeeded = true;
            return ptr;
#else
            void *ptr = mmap(nullptr, size, PROT_READ | PROT_WRITE,
                              MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
            if (ptr == MAP_FAILED)
                return nullptr;

            // Best-effort hint -- if this fails, we still have valid,
            // page-aligned (though not huge-page-backed) memory from the
            // mmap itself, so this isn't treated as a hard failure.
            madvise(ptr, size, MADV_HUGEPAGE);
            succeeded = true;
            return ptr;
#endif
        }

        /// @brief Frees a block previously returned by
        /// HugePageAllocPortable() with succeeded=true.
        /// @param ptr Pointer to free.
        /// @param size The exact size originally passed to
        /// HugePageAllocPortable() -- required on POSIX systems, where
        /// munmap() needs to know the mapping's length.
        inline void HugePageFreePortable(void *ptr, size_t size)
        {
#if defined(_WIN32)
            VirtualFree(ptr, 0, MEM_RELEASE);
#else
            munmap(ptr, size);
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
        /// @brief True if base_ptr was obtained via HugePageAllocPortable()
        /// and must therefore be freed with HugePageFreePortable(), not
        /// AlignedFreePortable(). Huge pages are best-effort and opt-in
        /// (see the constructor) -- this can be false even when huge
        /// pages were requested, if the request wasn't honored.
        bool used_huge_pages;

    public:
        /// @brief Allocates a single 64-byte-aligned buffer large enough
        /// to hold `total_floats` floats, rounded up to the nearest
        /// 64-byte boundary.
        /// @param total_floats Total number of floats this arena can
        /// hand out across all future AllocateFloats() calls combined.
        /// @param use_huge_pages Opt-in, best-effort request for
        /// huge-page-backed memory (see HugePageAllocPortable() for the
        /// platform-specific caveats). Defaults to false, preserving
        /// existing behavior exactly for anyone not explicitly opting
        /// in. Falls back safely to the standard aligned allocator if
        /// the request isn't honored -- this never causes a hard
        /// failure on its own.
        /// @throws std::bad_alloc if the underlying allocation fails
        /// (including the fallback path, if huge pages were requested
        /// but unavailable).
        explicit MemoryArena(size_t total_floats, bool use_huge_pages = false)
            : used_huge_pages(false)
        {
            // Guarantee total capacity is a multiple of 64 bytes
            capacity_bytes = (total_floats * sizeof(float) + 63) & ~63;

            if (use_huge_pages)
            {
                bool succeeded = false;
                base_ptr = static_cast<float *>(detail::HugePageAllocPortable(capacity_bytes, succeeded));
                used_huge_pages = succeeded;
            }

            if (!used_huge_pages)
            {
                // Either huge pages weren't requested, or the request
                // wasn't honored (mmap/VirtualAlloc failed, or the
                // system lacks the required configuration/privilege) --
                // fall back to the standard aligned allocator rather
                // than treating this as a hard failure.
                base_ptr = static_cast<float *>(detail::AlignedAllocPortable(64, capacity_bytes));
            }

            if (!base_ptr)
            {
                throw std::bad_alloc();
            }
            offset_bytes = 0;
        }

        /// @brief Frees the underlying allocation, via whichever
        /// deallocator matches the allocator actually used (see
        /// used_huge_pages).
        ~MemoryArena()
        {
            if (base_ptr)
            {
                if (used_huge_pages)
                    detail::HugePageFreePortable(base_ptr, capacity_bytes);
                else
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
        /// @brief Returns whether this arena's memory is actually
        /// huge-page-backed. Since huge pages are requested on a
        /// best-effort basis, this can be false even if use_huge_pages
        /// was passed to the constructor, if the request wasn't honored
        /// (system lacks configuration/privilege, or the allocation
        /// call failed) and the arena silently fell back to the
        /// standard aligned allocator instead.
        bool UsedHugePages() const { return used_huge_pages; }
    };
} // namespace Deep
