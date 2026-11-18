#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

/**
 * @file Blake3.h
 * @brief Streaming BLAKE3 (default 32-byte hash mode) used for model-file integrity hashes.
 * The compression kernels are Google Highway code in src/io/Blake3.cpp, dispatched at run time
 * to the best SIMD target the CPU supports. Adapted from Universal-Blake3 (github.com/Ra4ster).
 */

namespace Deep::io
{
namespace detail
{
/// @brief Plain streaming state. Layout is shared with the dispatched kernels.
struct Blake3State
{
  alignas(64) uint32_t cvStack[54 * 8];
  alignas(64) uint8_t chunkBuffer[1024];
  uint64_t chunkCounter = 0;
  size_t bufferLen = 0;
  uint32_t stackLen = 0;
};
} // namespace detail

class Blake3
{
public:
  /// @brief Absorbs `len` bytes. May be called any number of times.
  void Update(const void* data, size_t len) noexcept;

  /// @brief Returns the 32-byte digest as 64 lowercase hex characters. Does not change the state.
  std::string Hex() const;

private:
  detail::Blake3State state;
};
} // namespace Deep::io
