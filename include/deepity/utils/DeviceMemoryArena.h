#pragma once
#if defined(DEEPITY_USE_CUDA)
#include <deepity/backend/IComputeBackend.h>
#include <iostream>
#include <new>
#include <stdexcept>

/**
 * @file DeviceMemoryArena.h
 * @brief A 64-byte-aligned bump-pointer allocator over a single device
 * buffer, mirroring this codebase's host-side MemoryArena, but backed
 * by IComputeBackend::Allocate()/Free() rather than raw
 * cudaMalloc/cudaFree directly.
 *
 * @warning Individual chunks handed out by AllocateFloats() cannot be
 * freed independently; the entire arena is released at once in the
 * destructor.
 * @version 1.1
 * @date 2026-09-05
 */

namespace Deep
{
/// @brief A 64-byte-aligned bump-pointer allocator over a single
/// contiguous block of device memory, obtained via a supplied
/// IComputeBackendT<T, AccumT> (expected to be a CUDABackend/
/// CUDABackendT in practice, though this class itself doesn't
/// hard-require that specific type).
///
/// @tparam T Element type this arena allocates, matching the backend's
/// own T. `DeviceMemoryArena` (see the alias below) is
/// `DeviceMemoryArenaT<float, float>`.
/// @tparam AccumT Matches the backend's own AccumT; defaults to T, as
/// in IComputeBackendT itself.
template <typename T, typename AccumT = T>
class DeviceMemoryArenaT
{
private:
  /// @brief Backend used for the underlying allocation and its
  /// eventual release. Non-owning, must outlive this arena.
  IComputeBackendT<T, AccumT>* backend;
  /// @brief Base address of the underlying backend-allocated buffer.
  T* base_ptr;
  /// @brief Total capacity of the arena, in bytes, rounded up to a
  /// multiple of 64.
  size_t capacity_bytes;
  /// @brief Current allocation offset from base_ptr, in bytes.
  /// Advances monotonically with each Allocate() call and is
  /// never reset or reclaimed.
  size_t offset_bytes;

public:
  /// @brief Allocates a single 64-byte-aligned device buffer large
  /// enough to hold `total_elements` elements of T, rounded up to the
  /// nearest 64-byte boundary.
  /// @param backend The backend to allocate from (and later free
  /// through). Must outlive this DeviceMemoryArenaT.
  /// @param total_elements Total number of elements this arena can
  /// hand out across all future Allocate() calls combined.
  /// @throws std::bad_alloc if the underlying allocation fails.
  DeviceMemoryArenaT(IComputeBackendT<T, AccumT>* backend, size_t total_elements)
      : backend(backend)
  {
    capacity_bytes = (total_elements * sizeof(T) + 63) & ~(size_t)63;

    base_ptr = backend->Allocate(capacity_bytes / sizeof(T));
    if (!base_ptr)
    {
      throw std::bad_alloc();
    }
    offset_bytes = 0;
  }

  /// @brief Frees the underlying device buffer via the same
  /// backend it was allocated from.
  ~DeviceMemoryArenaT()
  {
    if (base_ptr)
    {
      backend->Free(base_ptr);
    }
  }

  // Delete copy/move constructors to prevent double-free corruption.

  /// @brief Deleted: DeviceMemoryArenaT owns a single device
  /// allocation, so copying would risk a double-free.
  DeviceMemoryArenaT(const DeviceMemoryArenaT&) = delete;
  /// @brief Deleted: DeviceMemoryArenaT owns a single device
  /// allocation, so copy-assignment would risk a double-free.
  DeviceMemoryArenaT& operator=(const DeviceMemoryArenaT&) = delete;

  /// @brief Allocates a 64-byte aligned chunk of elements from the
  /// arena.
  /// @param num_elements Number of elements to allocate from the
  /// arena. The actual reservation is rounded up to the nearest
  /// 64-byte boundary, matching MemoryArenaT's own guarantee.
  /// @return Pointer to the start of the allocated chunk, valid
  /// for the lifetime of this DeviceMemoryArenaT. 64-byte aligned.
  /// @throws std::runtime_error if the requested allocation would
  /// exceed the arena's total capacity.
  /// @warning Individual chunks are never freed independently; the
  /// entire arena is released at once in the destructor.
  T* Allocate(size_t num_elements)
  {
    size_t allocation_size = (num_elements * sizeof(T) + 63) & ~(size_t)63;

    if (offset_bytes + allocation_size > capacity_bytes)
    {
      throw std::runtime_error("Fatal: DeviceMemoryArenaT capacity exceeded during allocation.");
    }

    T* chunk = reinterpret_cast<T*>(reinterpret_cast<char*>(base_ptr) + offset_bytes);

    offset_bytes += allocation_size;
    return chunk;
  }

  /// @brief Alias for Allocate(). Only meaningful when T=float.
  T* AllocateFloats(size_t num_elements)
  {
    return Allocate(num_elements);
  }

  /// @brief Returns how many bytes have been allocated from the
  /// arena so far.
  size_t GetUsedBytes() const
  {
    return offset_bytes;
  }
  /// @brief Returns the arena's total capacity.
  size_t GetCapacityBytes() const
  {
    return capacity_bytes;
  }
};

/// @brief Non-templated alias: `DeviceMemoryArena` is
/// `DeviceMemoryArenaT<float, float>`.
using DeviceMemoryArena = DeviceMemoryArenaT<float, float>;
} // namespace Deep
#endif