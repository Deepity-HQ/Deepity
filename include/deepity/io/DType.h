#pragma once
#include <cstddef>
#include <cstdint>
#include <string>

/**
 * @file DType.h
 * @brief Registry of the element types model files can hold, and the codecs for types whose
 * on-disk layout differs from their in-memory one (4-bit codes packed two per byte).
 *
 * A type has a logical name (what the network holds), a block shape (`elementsPerBlock` elements
 * occupy `bytesPerBlock` bytes on disk) and a storage dtype (what the safetensors header says).
 * Standard types store as themselves. Packed types store as `U8`, with the logical type and shape
 * in the file's `deepity.dtypes` metadata, so stock safetensors readers still open the file.
 */

namespace Deep::io
{
enum class DTypeKind
{
  Standard, ///< Stored as itself, one element per block.
  Packed    ///< Stored as U8 blocks; needs the codec below to move to and from memory.
};

/// @brief Converts between the in-memory form (one byte per element, code in the low bits) and the
/// on-disk packed form. `count` is the number of elements; a final partial block is zero padded.
using PackFn = void (*)(const uint8_t* codes, size_t count, uint8_t* packed);
using UnpackFn = void (*)(const uint8_t* packed, size_t count, uint8_t* codes);

struct DTypeInfo
{
  const char* name;
  DTypeKind kind;
  size_t elementsPerBlock; ///< Elements in one block (1 for every Standard type).
  size_t bytesPerBlock;    ///< Bytes one block takes on disk.
  size_t memBytes;         ///< Bytes per element in memory.
  const char* storage;     ///< Header dtype: the name itself for Standard, "U8" for Packed.
  PackFn pack;             ///< Null for Standard.
  UnpackFn unpack;         ///< Null for Standard.
};

/// @brief Returns the registry entry for `name`, or nullptr if the dtype is unknown.
const DTypeInfo* FindDType(const std::string& name) noexcept;

/// @brief On-disk bytes for `elements` elements of `info`: whole blocks, the last one padded.
size_t StorageBytes(const DTypeInfo& info, size_t elements) noexcept;

/// @brief Byte order of model files. Always little endian.
///
/// Swaps each `elemBytes`-wide element of `count` elements in place when `hostIsBigEndian` is true,
/// and does nothing otherwise. The host flag is a parameter, not a compile-time test, so the swap
/// logic is testable on any machine; the reader and writer pass `std::endian::native == big`.
void ConvertEndianness(void* data, size_t elemBytes, size_t count, bool hostIsBigEndian) noexcept;
} // namespace Deep::io
