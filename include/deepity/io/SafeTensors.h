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
  /// @brief Which shard holds the tensor (0 for a single file). Set by the reader.
  size_t shard = 0;

  size_t ByteSize() const noexcept
  {
    return static_cast<size_t>(end - begin);
  }
};

/// @brief Bytes per element as laid out in a file written by a Standard dtype, or 0 if the dtype is
/// unknown. Packed dtype names report 1, the layout of 0.1.0a8 files that stored them unpacked.
size_t DTypeSize(const std::string& dtype) noexcept;

/// @brief Suffix of a shard index file.
inline constexpr const char* kIndexSuffix = ".safetensors.index.json";
/// @brief Metadata key under which the data-region BLAKE3 is stored.
inline constexpr const char* kHashKey = "deepity.content_blake3";
/// @brief Metadata key mapping tensor name to its logical dtype and shape, for packed tensors.
inline constexpr const char* kDTypesKey = "deepity.dtypes";

/// @brief Metadata key holding the Deepity model format version (a positive integer in a string).
inline constexpr const char* kFormatVersionKey = "deepity.format_version";
/// @brief The newest model format version this build reads, and the one it writes.
inline constexpr int kModelFormatVersion = 1;

/// @brief Throws unless `metadata` has no format version (a foreign or pre-versioned file) or one
/// this build supports. Names both versions when the file is newer, and rejects values that are not
/// a positive integer. Unknown keys and tensors are never an error; only the version is.
inline void CheckFormatVersion(const std::map<std::string, std::string>& metadata, const std::string& path)
{
  const auto it = metadata.find(kFormatVersionKey);
  if (it == metadata.end()) return;
  const std::string& text = it->second;
  long long version = 0;
  bool valid = !text.empty() && text.size() <= 9;
  for (const char c : text)
  {
    valid = valid && c >= '0' && c <= '9';
    if (valid) version = version * 10 + (c - '0');
  }
  if (!valid || version < 1)
    throw SafeTensorsError("SafeTensors: " + path + " has an invalid " + kFormatVersionKey + " '" + text +
                           "'; it must be a positive integer.");
  if (version > kModelFormatVersion)
    throw SafeTensorsError("SafeTensors: " + path + " was written with model format version " + text +
                           ", but this build of Deepity reads up to version " + std::to_string(kModelFormatVersion) +
                           ". Upgrade Deepity to load it.");
}

/// @brief Testing only: makes every reader and writer behave as on a big-endian host, so the byte-swap
/// paths can be exercised on a little-endian machine. In this mode tensor memory passed to
/// SafeTensorsWriter, and read back by SafeTensorsReader::Read(), is in big-endian order and the
/// files are still little endian. Not thread safe against concurrent readers and writers.
void SimulateBigEndianHost(bool enabled) noexcept;

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
  /// @brief Splits the tensors over several files named `<stem>-00001-of-0000N.safetensors` next to
  /// `indexPath` (which must end in `.safetensors.index.json`), then writes that index. Tensors
  /// keep their order and a shard closes once the next tensor would push it past `maxShardBytes`
  /// (a tensor larger than the limit gets a shard of its own). Every shard carries the full
  /// metadata and its own hash; the index goes last, so it never names a missing shard.
  /// @throws SafeTensorsError on a bad path or limit, or any write failure.
  void WriteSharded(const std::string& indexPath, uint64_t maxShardBytes) const;

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
  ///
  /// A path ending in `.index.json` opens a sharded model: the index and every shard it names are
  /// validated (shard names must be plain file names beside the index, each tensor must be in
  /// exactly the shard the index says), and the reader then behaves as one file.
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
  /// @brief Shared owner of the mapping holding `entry`, so a view (for example a numpy array) can
  /// outlive the reader. Null when the file is not mapped.
  std::shared_ptr<const MappedFile> Mapping(const TensorEntry& entry) const noexcept
  {
    return shards.empty() ? mapping : shards[entry.shard]->mapping;
  }
  /// @brief Number of files behind this reader: 1 for a plain file, N for a shard index.
  size_t ShardCount() const noexcept
  {
    return shards.empty() ? 1 : shards.size();
  }
  /// @brief Reads exactly `entry.ByteSize()` bytes into `dst`.
  /// @throws SafeTensorsError if `dstBytes` differs from the tensor's size or the read fails.
  void Read(const TensorEntry& entry, void* dst, size_t dstBytes) const;

private:
  std::string path;
  mutable std::ifstream in;
  std::shared_ptr<const MappedFile> mapping;
  std::vector<std::shared_ptr<SafeTensorsReader>> shards; ///< Non-empty only for a shard index.
  std::vector<std::string> shardFiles;
  void OpenSharded(bool verifyHash, bool useMmap);
  std::map<std::string, std::string> metadata;
  std::vector<TensorEntry> tensors;
  uint64_t dataOffset = 0;
  uint64_t dataSize = 0;
};
} // namespace Deep::io
