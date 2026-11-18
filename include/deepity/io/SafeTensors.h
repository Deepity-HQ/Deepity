#pragma once
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * @file SafeTensors.h
 * @brief Standalone reader and writer for the safetensors container that Deepity model files use
 * (see docs/model-format.md). It knows nothing about networks: it moves named, typed, shaped byte
 * ranges to and from a file, validates everything it reads, and never trusts the header.
 */

namespace Deep::io
{
/// @brief Thrown for any malformed, truncated or unsafe file, and for write failures.
class SafeTensorsError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

/// @brief One tensor's header entry. Offsets are relative to the start of the data region.
struct TensorEntry
{
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;

  size_t ByteSize() const noexcept
  {
    return static_cast<size_t>(end - begin);
  }
};

/// @brief Bytes per element for an allowed dtype name, or 0 if the dtype is not on the allow-list.
size_t DTypeSize(const std::string& dtype) noexcept;

/// @brief Metadata key under which the data-region BLAKE3 is stored.
inline constexpr const char* kHashKey = "deepity.content_blake3";

/// @brief Collects tensors and metadata, then writes one file atomically.
///
/// Layout written: 8-byte size, JSON header padded with spaces so the data region starts on a
/// 64-byte boundary, then the tensors back to back (no gaps, as stock safetensors requires),
/// ordered by decreasing element size so every tensor is naturally aligned.
class SafeTensorsWriter
{
public:
  /// @brief Required alignment of the data region's start within the file.
  static constexpr size_t kAlignment = 64;

  void SetMetadata(const std::string& key, const std::string& value);
  /// @brief Writes the BLAKE3 hash of the data region into the metadata. On by default.
  void SetHash(bool enabled) noexcept
  {
    hash = enabled;
  }
  /// @brief Registers a tensor. `data` is not copied and must stay valid until Write() returns.
  /// @throws SafeTensorsError for an unknown dtype, a duplicate name, or a size that does not
  /// equal prod(shape) * element size.
  void AddTensor(const std::string& name, const std::string& dtype, const std::vector<size_t>& shape,
                 const void* data, size_t bytes);
  /// @brief Writes to `path` through a temporary file and a rename, so a crash never leaves a
  /// half-written model at `path`.
  void Write(const std::string& path) const;

private:
  struct Pending
  {
    TensorEntry entry;
    const void* data;
  };
  std::vector<Pending> tensors;
  std::map<std::string, std::string> metadata;
  bool hash = true;
};

/// @brief Opens and fully validates a file, then reads tensors on request. Not thread-safe.
///
/// Reads both files written by SafeTensorsWriter and files from other safetensors writers
/// (alignment is not required on read).
class SafeTensorsReader
{
public:
  /// @brief Largest header the reader accepts.
  static constexpr uint64_t kMaxHeaderBytes = 16ull << 20;

  /// @param verifyHash when true and the file carries `deepity.content_blake3`, the data region
  /// is hashed and compared. Pass false to skip the pass for a trusted local file.
  /// @throws SafeTensorsError if the file is missing, truncated, malformed or unsafe.
  explicit SafeTensorsReader(const std::string& path, bool verifyHash = true);

  const std::map<std::string, std::string>& Metadata() const noexcept
  {
    return metadata;
  }
  /// @brief Tensors in file order.
  const std::vector<TensorEntry>& Tensors() const noexcept
  {
    return tensors;
  }
  /// @brief Returns the entry named `name`, or nullptr.
  const TensorEntry* Find(const std::string& name) const noexcept;
  /// @brief Absolute file offset of the data region.
  uint64_t DataOffset() const noexcept
  {
    return dataOffset;
  }
  /// @brief Reads exactly `entry.ByteSize()` bytes into `dst`.
  /// @throws SafeTensorsError if `dstBytes` differs from the tensor's size or the read fails.
  void Read(const TensorEntry& entry, void* dst, size_t dstBytes) const;

private:
  std::string path;
  mutable std::ifstream in;
  std::map<std::string, std::string> metadata;
  std::vector<TensorEntry> tensors;
  uint64_t dataOffset = 0;
  uint64_t dataSize = 0;
};
} // namespace Deep::io
