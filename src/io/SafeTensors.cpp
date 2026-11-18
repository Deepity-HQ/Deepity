#include "deepity/io/SafeTensors.h"
#include "deepity/io/Blake3.h"
#include <algorithm>
#include <bit>
#include <cstring>
#include <filesystem>
#include <limits>
#include <nlohmann/json.hpp>

namespace Deep::io
{
namespace
{
constexpr const char* kMetadataKey = "__metadata__";
constexpr size_t kChunkBytes = 1u << 20;
constexpr bool kHostIsBigEndian = std::endian::native == std::endian::big;

/// @brief prod(shape) * elemBytes, or false on overflow.
bool CheckedBytes(const std::vector<size_t>& shape, size_t elemBytes, uint64_t* out)
{
  uint64_t total = elemBytes;
  for (size_t dim : shape)
  {
    if (dim != 0 && total > std::numeric_limits<uint64_t>::max() / dim)
      return false;
    total *= dim;
  }
  *out = total;
  return true;
}

/// @brief Element size of what the header will say: 1 for a packed tensor (stored as U8).
size_t StorageElemBytes(const TensorEntry& e) noexcept
{
  return e.packed ? 1 : DTypeSize(e.dtype);
}

[[noreturn]] void Fail(const std::string& path, const std::string& why)
{
  throw SafeTensorsError("SafeTensors: " + path + ": " + why);
}
} // namespace

size_t DTypeSize(const std::string& dtype) noexcept
{
  const DTypeInfo* info = FindDType(dtype);
  if (info == nullptr)
    return 0;
  return info->kind == DTypeKind::Standard ? info->memBytes : 1;
}

// ---------------------------------------------------------------- writer

void SafeTensorsWriter::SetMetadata(const std::string& key, const std::string& value)
{
  if (key == kMetadataKey || key == kHashKey || key == kDTypesKey)
    throw SafeTensorsError("SafeTensors: metadata key '" + key + "' is reserved.");
  metadata[key] = value;
}

void SafeTensorsWriter::AddTensor(const std::string& name, const std::string& dtype,
                                  const std::vector<size_t>& shape, const void* data, size_t bytes)
{
  const DTypeInfo* info = FindDType(dtype);
  if (info == nullptr)
    throw SafeTensorsError("SafeTensors: tensor '" + name + "' has unsupported dtype '" + dtype + "'.");
  if (name.empty() || name == kMetadataKey)
    throw SafeTensorsError("SafeTensors: invalid tensor name '" + name + "'.");
  for (const auto& t : tensors)
    if (t.entry.name == name)
      throw SafeTensorsError("SafeTensors: duplicate tensor name '" + name + "'.");

  uint64_t elements = 0, expected = 0;
  if (!CheckedBytes(shape, 1, &elements) || elements > std::numeric_limits<uint64_t>::max() - 64)
    throw SafeTensorsError("SafeTensors: tensor '" + name + "' has an overflowing shape.");
  const bool packed = info->kind == DTypeKind::Packed;
  if (packed)
    expected = StorageBytes(*info, static_cast<size_t>(elements));
  else if (!CheckedBytes(shape, info->memBytes, &expected))
    throw SafeTensorsError("SafeTensors: tensor '" + name + "' has an overflowing shape.");
  if (expected != bytes)
    throw SafeTensorsError("SafeTensors: tensor '" + name + "' has " + std::to_string(bytes) +
                           " bytes, but its shape and dtype need " + std::to_string(expected) + ".");
  if (bytes > 0 && data == nullptr)
    throw SafeTensorsError("SafeTensors: tensor '" + name + "' has a null data pointer.");

  Pending p;
  p.entry.name = name;
  p.entry.dtype = dtype;
  p.entry.shape = shape;
  p.entry.packed = packed;
  p.data = data;
  if (!packed && kHostIsBigEndian && info->memBytes > 1 && bytes > 0)
  {
    p.owned.assign(static_cast<const char*>(data), static_cast<const char*>(data) + bytes);
    ConvertEndianness(p.owned.data(), info->memBytes, bytes / info->memBytes, kHostIsBigEndian);
    p.data = p.owned.data();
  }
  tensors.push_back(std::move(p));
  tensors.back().entry.end = bytes; // temporarily the byte size; real offsets are set in Write()
}

void SafeTensorsWriter::Write(const std::string& path) const
{
  // Largest elements first: every tensor then starts at a multiple of its own element size.
  std::vector<const Pending*> order;
  for (const auto& t : tensors)
    order.push_back(&t);
  std::stable_sort(order.begin(), order.end(), [](const Pending* a, const Pending* b)
                   { return StorageElemBytes(a->entry) > StorageElemBytes(b->entry); });

  std::map<std::string, std::string> meta = metadata;
  if (hash)
  {
    Blake3 hasher;
    for (const Pending* t : order)
      if (t->entry.end > 0)
        hasher.Update(t->data, static_cast<size_t>(t->entry.end));
    meta[kHashKey] = hasher.Hex();
  }

  nlohmann::json header = nlohmann::json::object();
  nlohmann::json packedTypes = nlohmann::json::object();
  uint64_t offset = 0;
  for (const Pending* t : order)
  {
    const uint64_t bytes = t->entry.end;
    if (t->entry.packed)
    {
      packedTypes[t->entry.name] = {{"dtype", t->entry.dtype}, {"shape", t->entry.shape}};
      header[t->entry.name] = {{"dtype", "U8"},
                               {"shape", std::vector<uint64_t>{bytes}},
                               {"data_offsets", {offset, offset + bytes}}};
    }
    else
      header[t->entry.name] = {{"dtype", t->entry.dtype},
                               {"shape", t->entry.shape},
                               {"data_offsets", {offset, offset + bytes}}};
    offset += bytes;
  }
  if (!packedTypes.empty())
    meta[kDTypesKey] = packedTypes.dump();
  if (!meta.empty())
    header[kMetadataKey] = meta;

  std::string headerText = header.dump();
  const size_t total = 8 + headerText.size();
  const size_t padded = (total + kAlignment - 1) / kAlignment * kAlignment;
  headerText.append(padded - total, ' ');
  const uint64_t headerSize = headerText.size();

  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out)
      Fail(path, "cannot open '" + tmp + "' for writing");
    out.write(reinterpret_cast<const char*>(&headerSize), sizeof(headerSize));
    out.write(headerText.data(), static_cast<std::streamsize>(headerText.size()));
    for (const Pending* t : order)
      if (t->entry.end > 0)
        out.write(static_cast<const char*>(t->data), static_cast<std::streamsize>(t->entry.end));
    out.flush();
    if (!out)
    {
      out.close();
      std::error_code ignore;
      std::filesystem::remove(tmp, ignore);
      Fail(path, "write failed (disk full or permission problem)");
    }
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec)
  {
    std::filesystem::remove(tmp, ec);
    Fail(path, "cannot move the finished file into place");
  }
}

// ---------------------------------------------------------------- reader

SafeTensorsReader::SafeTensorsReader(const std::string& filePath, bool verifyHash, bool useMmap) : path(filePath)
{
  std::error_code ec;
  const uint64_t fileSize = std::filesystem::file_size(path, ec);
  if (ec)
    Fail(path, "cannot open file: " + ec.message());
  in.open(path, std::ios::binary);
  if (!in)
    Fail(path, "cannot open file");
  if (fileSize < 8)
    Fail(path, "file is smaller than the 8-byte header size field");

  uint64_t headerSize = 0;
  in.read(reinterpret_cast<char*>(&headerSize), sizeof(headerSize));
  if (!in)
    Fail(path, "cannot read the header size");
  if (headerSize > kMaxHeaderBytes)
    Fail(path, "header size " + std::to_string(headerSize) + " exceeds the limit of " +
                   std::to_string(kMaxHeaderBytes));
  if (8 + headerSize > fileSize)
    Fail(path, "header size " + std::to_string(headerSize) + " runs past the end of the file");

  std::string headerText(static_cast<size_t>(headerSize), '\0');
  in.read(headerText.data(), static_cast<std::streamsize>(headerSize));
  if (!in)
    Fail(path, "cannot read the header");

  nlohmann::json header = nlohmann::json::parse(headerText, nullptr, /*allow_exceptions=*/false);
  if (header.is_discarded() || !header.is_object())
    Fail(path, "header is not a valid JSON object");

  dataOffset = 8 + headerSize;
  dataSize = fileSize - dataOffset;

  for (auto it = header.begin(); it != header.end(); ++it)
  {
    const std::string& key = it.key();
    const nlohmann::json& value = it.value();
    if (key == kMetadataKey)
    {
      if (!value.is_object())
        Fail(path, "__metadata__ is not an object");
      for (auto m = value.begin(); m != value.end(); ++m)
      {
        if (!m.value().is_string())
          Fail(path, "metadata value for '" + m.key() + "' is not a string");
        metadata[m.key()] = m.value().get<std::string>();
      }
      continue;
    }

    if (!value.is_object() || !value.contains("dtype") || !value["dtype"].is_string() ||
        !value.contains("shape") || !value["shape"].is_array() || !value.contains("data_offsets") ||
        !value["data_offsets"].is_array() || value["data_offsets"].size() != 2)
      Fail(path, "tensor '" + key + "' has a malformed header entry");

    TensorEntry e;
    e.name = key;
    e.dtype = value["dtype"].get<std::string>();
    const size_t elem = DTypeSize(e.dtype);
    if (elem == 0)
      Fail(path, "tensor '" + key + "' has unsupported dtype '" + e.dtype + "'");
    for (const auto& d : value["shape"])
    {
      if (!d.is_number_unsigned())
        Fail(path, "tensor '" + key + "' has a non-integer or negative dimension");
      e.shape.push_back(d.get<size_t>());
    }
    const auto& off = value["data_offsets"];
    if (!off[0].is_number_unsigned() || !off[1].is_number_unsigned())
      Fail(path, "tensor '" + key + "' has non-integer data_offsets");
    e.begin = off[0].get<uint64_t>();
    e.end = off[1].get<uint64_t>();
    if (e.begin > e.end || e.end > dataSize)
      Fail(path, "tensor '" + key + "' data_offsets [" + std::to_string(e.begin) + ", " +
                     std::to_string(e.end) + "] lie outside the " + std::to_string(dataSize) +
                     "-byte data region");
    uint64_t expected = 0;
    if (!CheckedBytes(e.shape, elem, &expected) || expected != e.end - e.begin)
      Fail(path, "tensor '" + key + "' has " + std::to_string(e.end - e.begin) +
                     " bytes, which does not match its shape and dtype");
    tensors.push_back(std::move(e));
  }

  std::stable_sort(tensors.begin(), tensors.end(), [](const TensorEntry& a, const TensorEntry& b)
                   { return a.begin != b.begin ? a.begin < b.begin : a.end < b.end; });
  uint64_t cursor = 0;
  for (const auto& t : tensors)
  {
    if (t.begin != cursor)
      Fail(path, "tensor '" + t.name + "' " + (t.begin < cursor ? "overlaps the previous tensor"
                                                                 : "leaves a gap before it in the data region"));
    cursor = t.end;
  }
  if (cursor != dataSize)
    Fail(path, "the data region is " + std::to_string(dataSize) + " bytes, but tensors cover " +
                   std::to_string(cursor));

  const auto typesIt = metadata.find(kDTypesKey);
  if (typesIt != metadata.end())
  {
    nlohmann::json overlay = nlohmann::json::parse(typesIt->second, nullptr, /*allow_exceptions=*/false);
    if (overlay.is_discarded() || !overlay.is_object())
      Fail(path, std::string(kDTypesKey) + " is not a valid JSON object");
    for (auto it = overlay.begin(); it != overlay.end(); ++it)
    {
      const std::string& name = it.key();
      const nlohmann::json& v = it.value();
      auto entry = std::find_if(tensors.begin(), tensors.end(), [&](const TensorEntry& t) { return t.name == name; });
      if (entry == tensors.end())
        Fail(path, std::string(kDTypesKey) + " names '" + name + "', which is not a tensor in the file");
      if (!v.is_object() || !v.contains("dtype") || !v["dtype"].is_string() || !v.contains("shape") ||
          !v["shape"].is_array())
        Fail(path, std::string(kDTypesKey) + " entry for '" + name + "' is malformed");
      const DTypeInfo* info = FindDType(v["dtype"].get<std::string>());
      if (info == nullptr || info->kind != DTypeKind::Packed)
        Fail(path, "tensor '" + name + "' declares logical dtype '" + v["dtype"].get<std::string>() +
                       "', which is not a packed dtype this build knows");
      if (entry->dtype != "U8" || entry->shape.size() != 1)
        Fail(path, "packed tensor '" + name + "' must be stored as a 1-D U8 tensor");
      std::vector<size_t> logicalShape;
      for (const auto& d : v["shape"])
      {
        if (!d.is_number_unsigned())
          Fail(path, "packed tensor '" + name + "' has a non-integer or negative logical dimension");
        logicalShape.push_back(d.get<size_t>());
      }
      uint64_t elements = 0;
      if (!CheckedBytes(logicalShape, 1, &elements) || elements > std::numeric_limits<uint64_t>::max() - 64 ||
          StorageBytes(*info, static_cast<size_t>(elements)) != entry->ByteSize())
        Fail(path, "packed tensor '" + name + "' has " + std::to_string(entry->ByteSize()) +
                       " bytes, which does not match its logical shape and dtype");
      entry->dtype = info->name;
      entry->shape = std::move(logicalShape);
      entry->packed = true;
    }
  }

  if (useMmap)
  {
    MappedFile m = MappedFile::Open(path);
    if (m.Data() != nullptr && m.Size() == fileSize) mapping = std::make_shared<const MappedFile>(std::move(m));
  }

  const auto hashIt = metadata.find(kHashKey);
  if (verifyHash && hashIt != metadata.end() && mapping != nullptr)
  {
    Blake3 hasher;
    hasher.Update(mapping->Data() + dataOffset, static_cast<size_t>(dataSize));
    if (hasher.Hex() != hashIt->second)
      Fail(path, "content hash mismatch: the file is corrupted or was modified");
  }
  else if (verifyHash && hashIt != metadata.end())
  {
    Blake3 hasher;
    std::vector<char> chunk(kChunkBytes);
    in.seekg(static_cast<std::streamoff>(dataOffset));
    for (uint64_t left = dataSize; left > 0;)
    {
      const size_t take = static_cast<size_t>(std::min<uint64_t>(left, kChunkBytes));
      in.read(chunk.data(), static_cast<std::streamsize>(take));
      if (!in)
        Fail(path, "cannot read the data region to verify its hash");
      hasher.Update(chunk.data(), take);
      left -= take;
    }
    if (hasher.Hex() != hashIt->second)
      Fail(path, "content hash mismatch: the file is corrupted or was modified");
  }
}

const TensorEntry* SafeTensorsReader::Find(const std::string& name) const noexcept
{
  for (const auto& t : tensors)
    if (t.name == name)
      return &t;
  return nullptr;
}

const void* SafeTensorsReader::Bytes(const TensorEntry& entry) const noexcept
{
  if (mapping == nullptr) return nullptr;
  if (!entry.packed && kHostIsBigEndian && DTypeSize(entry.dtype) > 1) return nullptr;
  return mapping->Data() + dataOffset + entry.begin;
}

void SafeTensorsReader::Read(const TensorEntry& entry, void* dst, size_t dstBytes) const
{
  if (dstBytes != entry.ByteSize())
    Fail(path, "tensor '" + entry.name + "' is " + std::to_string(entry.ByteSize()) +
                   " bytes, but the destination holds " + std::to_string(dstBytes));
  if (dstBytes == 0)
    return;
  if (mapping != nullptr)
    std::memcpy(dst, mapping->Data() + dataOffset + entry.begin, dstBytes);
  else
  {
    in.clear();
    in.seekg(static_cast<std::streamoff>(dataOffset + entry.begin));
    in.read(static_cast<char*>(dst), static_cast<std::streamsize>(dstBytes));
    if (!in)
      Fail(path, "cannot read tensor '" + entry.name + "'");
  }
  if (!entry.packed)
  {
    const size_t elemBytes = DTypeSize(entry.dtype);
    ConvertEndianness(dst, elemBytes, dstBytes / elemBytes, kHostIsBigEndian);
  }
}
} // namespace Deep::io
