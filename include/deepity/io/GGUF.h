#pragma once
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <deepity/io/MappedFile.h>

/**
 * @file GGUF.h
 * @brief Reader for GGUF files (the llama.cpp model format), versions 2 and 3.
 *
 * Reads the key-value header and the tensor table, validates every offset and size against the
 * file, and decodes tensors to float32 or integers. Written for untrusted input: counts, string
 * lengths and array lengths are bounded by the file size, and nothing is read outside the mapping.
 */

namespace Deep::io
{
/// @brief Thrown for a missing, malformed, truncated or unsupported GGUF file.
class GGUFError : public std::runtime_error
{
public:
  using std::runtime_error::runtime_error;
};

/// @brief ggml tensor type ids as stored in the file.
enum class GgmlType : uint32_t
{
  F32 = 0, F16 = 1, Q4_0 = 2, Q4_1 = 3, Q5_0 = 6, Q5_1 = 7, Q8_0 = 8,
  I8 = 24, I16 = 25, I32 = 26, I64 = 27, F64 = 28, BF16 = 30
};

struct GGUFTensor
{
  std::string name;
  uint32_t ggmlType = 0;
  std::string typeName;         ///< "F32", "Q8_0", ... or "type_<id>" for one this reader does not know.
  std::vector<size_t> shape;    ///< Row-major order (GGUF stores the fastest axis first; reversed here).
  uint64_t offset = 0;          ///< Byte offset from the start of the file.
  uint64_t byteSize = 0;
  bool integer = false;         ///< True for I8..I64; everything else decodes to floating point.
  size_t Elements() const noexcept;
};

class GGUFReader
{
public:
  /// @brief Opens and validates `path`.
  /// @throws GGUFError for a missing file, bad magic, unsupported version, or any size, offset or
  /// count that does not fit the file.
  explicit GGUFReader(const std::string& path);

  uint32_t Version() const noexcept
  {
    return version;
  }
  /// @brief Key-value pairs as a JSON object string. Arrays longer than 256 elements (token
  /// lists, merges) are replaced by `{"array_length": n}`.
  const std::string& MetadataJson() const noexcept
  {
    return metadata;
  }
  const std::vector<GGUFTensor>& Tensors() const noexcept
  {
    return tensors;
  }
  /// @brief True when this reader can decode the tensor (everything but the K-quants and others).
  static bool CanDecode(const GGUFTensor& t) noexcept;
  /// @brief Decodes a floating-point tensor to float32 (`dst` holds `t.Elements()` floats).
  /// @throws GGUFError for an integer or unsupported type, naming it.
  void ReadFloat32(const GGUFTensor& t, float* dst) const;
  /// @brief Copies an integer tensor as int64 (`dst` holds `t.Elements()` values).
  void ReadInt64(const GGUFTensor& t, int64_t* dst) const;

private:
  std::string path;
  MappedFile file;
  uint32_t version = 0;
  std::string metadata;
  std::vector<GGUFTensor> tensors;
};
} // namespace Deep::io
