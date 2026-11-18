#pragma once
#include <cstddef>
#include <cstdint>
#include <deepity/io/DType.h>
#include <deepity/io/MappedFile.h>
#include <fstream>
#include <map>
#include <memory>
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

/// @brief One tensor's entry. `dtype` and `shape` are the logical ones (a packed 4-bit tensor reports
/// its real type and shape, not the `U8` bytes in the header). `begin`/`end` are on-disk byte
/// offsets relative to the start of the data region.
struct TensorEntry
{
  std::string name;
  std::string dtype;
  std::vector<size_t> shape;
  uint64_t begin = 0;
  uint64_t end = 0;
  /// @brief True when the bytes use a packed layout (see DType.h), so callers must unpack them.
  bool packed = false;

  size_t ByteSize() const noexcept
  {
    return static_cast<size_t>(end - begin);
  }
};

/// @brief Bytes per element as laid out in a file written by a Standard dtype, or 0 if the dtype is
/// unknown. Packed dtype names report 1, the layout of 0.1.0a8 files that stored them unpacked.
size_t DTypeSize(const std::string& dtype) noexcept;

/// @brief Metadata key under which the data-region BLAKE3 is stored.
inline constexpr const char* kHashKey = "deepity.content_blake3";
/// @brief Metadata key mapping tensor name to its logical dtype and shape, for packed tensors.
inline constexpr const char* kDTypesKey = "deepity.dtypes";

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
  /// @brief Registers a tensor. `data` must stay valid until Write() returns. For a Packed dtype
  /// (see DType.h) `data` is already in the packed on-disk layout and `bytes` is StorageBytes();
  /// the writer stores it as `U8` and records the logical type in `deepity.dtypes`. Standard
  /// dtypes are given in host byte order and written little endian.
  /// @throws SafeTensorsError for an unknown dtype, a duplicate name, or a size that does not
  /// match the dtype and shape.
  void AddTensor(const std::string& name, const std::string& dtype, const std::vector<size_t>& shape,
                 const void* data, size_t bytes);
  /// @brief Writes to `path` through a temporary file and a rename, so a crash never leaves a
  /// half-written model at `path`.
  void Write(const std::string& path) const;

private:
  struct Pending
  {
    TensorEntry entry;
    const void* data = nullptr;
    std::vector<char> owned; ///< Byte-swapped copy on big-endian hosts; `data` points into it.
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
  /// @param useMmap map the file (the default); falls back to buffered reads if mapping fails.
  explicit SafeTensorsReader(const std::string& path, bool verifyHash = true, bool useMmap = true);

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
  /// @brief True when the file is memory mapped (reads copy from the mapping; Bytes() works).
  bool IsMapped() const noexcept
  {
    return mapping != nullptr;
  }
  /// @brief Zero-copy pointer to a tensor's on-disk bytes, or nullptr when the file is not mapped
  /// or the tensor would need conversion (byte swap on a big-endian host, packed types are fine
  /// as raw bytes). Valid while `Mapping()` or this reader is alive.
  const void* Bytes(const TensorEntry& entry) const noexcept;
  /// @brief Shared owner of the mapping, so a view (for example a numpy array) can outlive the reader.
  std::shared_ptr<const MappedFile> Mapping() const noexcept
  {
    return mapping;
  }
  /// @brief Reads exactly `entry.ByteSize()` bytes into `dst`.
  /// @throws SafeTensorsError if `dstBytes` differs from the tensor's size or the read fails.
  void Read(const TensorEntry& entry, void* dst, size_t dstBytes) const;

private:
  std::string path;
  mutable std::ifstream in;
  std::shared_ptr<const MappedFile> mapping;
  std::map<std::string, std::string> metadata;
  std::vector<TensorEntry> tensors;
  uint64_t dataOffset = 0;
  uint64_t dataSize = 0;
};
} // namespace Deep::io
