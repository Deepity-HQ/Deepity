#pragma once
#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace Deep {

    class MemoryArena {
    private:
        float* base_ptr;
        size_t capacity_bytes;
        size_t offset_bytes;

    public:
        MemoryArena(size_t total_floats) {
            // Guarantee total capacity is a multiple of 64 bytes
            capacity_bytes = (total_floats * sizeof(float) + 63) & ~63;
            base_ptr = static_cast<float*>(std::aligned_alloc(64, capacity_bytes));
            
            if (!base_ptr) {
                throw std::bad_alloc();
            }
            offset_bytes = 0;
        }

        ~MemoryArena() {
            if (base_ptr) {
                std::free(base_ptr);
            }
        }

        // Delete copy/move constructors to prevent double-free corruption
        MemoryArena(const MemoryArena&) = delete;
        MemoryArena& operator=(const MemoryArena&) = delete;

        /// @brief Allocates a 64-byte aligned chunk of floats from the arena
        float* AllocateFloats(size_t num_floats) {
            // Calculate how many bytes we need, padded to the nearest 64-byte boundary
            size_t allocation_size = (num_floats * sizeof(float) + 63) & ~63;
            
            if (offset_bytes + allocation_size > capacity_bytes) {
                throw std::runtime_error("Fatal: MemoryArena capacity exceeded during allocation.");
            }
            
            float* chunk = reinterpret_cast<float*>(
                reinterpret_cast<char*>(base_ptr) + offset_bytes
            );
            
            offset_bytes += allocation_size;
            return chunk;
        }

        size_t GetUsedBytes() const { return offset_bytes; }
        size_t GetCapacityBytes() const { return capacity_bytes; }
    };

} // namespace Deep