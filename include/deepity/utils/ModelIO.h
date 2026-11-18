#pragma once
#include <algorithm>
#include <cstdint>
#include <deepity/backend/DeviceType.h>
#include <deepity/io/SafeTensors.h>
#include <deepity/utils/DTypeTag.h>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

/**
 * @file ModelIO.h
 * @brief Saves/loads a network's weights to/from a safetensors file (see docs/model-format.md),
 * through the validating reader/writer in deepity/io/SafeTensors.h. Templated on NetworkType, so
 * it works across every PC network variant that exposes GetLayers(). Each tensor's element type
 * comes from its own `.data` pointer (via DTypeTag<T>). Load() checks the file's dtype and shape
 * against the calling network and throws std::runtime_error on a mismatch.
 */

namespace Deep
{
namespace detail
{
// Detects structurally whether LayerPtr has a SyncStateDictFromDevice/
// ToDevice method (virtual override or plain method of that name),
// and calls it if so. No-op for a layer type with neither.
template <typename LayerPtr> void SyncFromDeviceIfSupported(LayerPtr& layerPtr)
{
  if constexpr (requires { layerPtr->SyncStateDictFromDevice(); })
    layerPtr->SyncStateDictFromDevice();
}

template <typename LayerPtr> void SyncToDeviceIfSupported(LayerPtr& layerPtr)
{
  if constexpr (requires { layerPtr->SyncStateDictToDevice(); })
    layerPtr->SyncStateDictToDevice();
}

/// @brief Row count implied by `shape` (product of every dimension but
/// the last) and that last dimension's own width, for a tensor whose
/// `rowStride` differs from it (see TensorDescriptor's own comment).
inline std::pair<size_t, size_t> RowsAndWidth(const std::vector<size_t>& shape)
{
  size_t width = shape.empty() ? 0 : shape.back();
  size_t rows = 1;
  for (size_t i = 0; i + 1 < shape.size(); ++i)
    rows *= shape[i];
  return {rows, width};
}

/// @brief True when the tensor is stored with padding between rows.
inline bool IsPadded(const std::vector<size_t>& shape, size_t rowStride)
{
  return rowStride != 0 && rowStride != RowsAndWidth(shape).second;
}

/// @brief Stable on-disk name for a layer's state-dict entry ("W" becomes "weight", and so on).
inline std::string TensorFileName(size_t layer, const std::string& name)
{
  std::string base = name == "W" ? "weight" : name == "b" ? "bias" : name == "Psi" ? "psi" : name;
  return "layers." + std::to_string(layer) + "." + base;
}

/// @brief Name used by files written before the versioned format (0.1.0a8 and earlier).
inline std::string LegacyTensorFileName(size_t layer, const std::string& name)
{
  return "layer_" + std::to_string(layer) + "." + name;
}

/// @brief Adds `tensor` to `writer` under `fileName`, packing padded rows into `packed` (the writer
/// keeps pointers until Write() returns, so `packed` must outlive it).
template <typename Descriptor>
void AddTensorPacked(io::SafeTensorsWriter& writer, std::vector<std::vector<char>>& packed,
                     const std::string& fileName, const Descriptor& tensor)
{
  using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
  const auto [rows, width] = RowsAndWidth(tensor.shape);
  const size_t rowBytes = width * sizeof(ElementType);
  const void* source = tensor.data;
  if (IsPadded(tensor.shape, tensor.rowStride))
  {
    packed.emplace_back(rows * rowBytes);
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(reinterpret_cast<const char*>(tensor.data + r * tensor.rowStride), rowBytes,
                  packed.back().data() + r * rowBytes);
    source = packed.back().data();
  }
  writer.AddTensor(fileName, DTypeTag<ElementType>::Name(), tensor.shape, source, rows * rowBytes);
}

/// @brief Reads `entry` (whose shape is `readShape`) into `tensor`, spreading rows over the
/// tensor's row stride when it is padded.
template <typename Descriptor>
void ReadTensorInto(const io::SafeTensorsReader& reader, const io::TensorEntry& entry,
                    const std::vector<size_t>& readShape, const Descriptor& tensor)
{
  using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
  const auto [rows, width] = RowsAndWidth(readShape);
  if (!IsPadded(readShape, tensor.rowStride))
  {
    reader.Read(entry, tensor.data, rows * width * sizeof(ElementType));
    return;
  }
  std::vector<ElementType> tight(rows * width);
  reader.Read(entry, tight.data(), tight.size() * sizeof(ElementType));
  for (size_t r = 0; r < rows; ++r)
    std::copy_n(tight.data() + r * width, width, tensor.data + r * tensor.rowStride);
}

/// @brief Throws unless every layer can describe all of its optimizer state, and the network
/// lives in host memory (device-resident moments are not readable from here).
template <typename NetworkType> void RequireCheckpointSupport(const NetworkType& net)
{
  if constexpr (requires { net.GetDevice(); })
    if (net.GetDevice() == DeviceType::DEVICE_GPU)
      throw std::runtime_error("ModelIO: checkpoints of GPU networks are not supported yet.");
  size_t index = 0;
  for (const auto& layerPtr : net.GetLayers())
  {
    bool supported = false;
    if constexpr (requires { layerPtr->SupportsCheckpoint(); })
      supported = layerPtr->SupportsCheckpoint();
    if (!supported)
      throw std::runtime_error("ModelIO: layer " + std::to_string(index) +
                               " does not support checkpoints (no optimizer state accessors).");
    ++index;
  }
}

inline std::string OptimizerTensorName(size_t layer, const std::string& name)
{
  return "optimizer.layers." + std::to_string(layer) + "." + name;
}
} // namespace detail

/// @brief Static save/load helpers for any network type exposing
/// GetLayers(), templated so it works across every PC network
/// variant without each one reimplementing serialization.
class ModelIO
{
public:
  /// @brief Writes `net`'s weights to a single safetensors file at `filepath`, atomically, with
  /// a content hash and `deepity.*` metadata. Padded tensors are written without their padding.
  /// @return true. Failure is reported by exception, so the reason is never lost.
  /// @throws io::SafeTensorsError if the file can't be written (message names the path and cause).
  /// @param metadata extra string entries stored in `__metadata__` (for example `deepity.model`,
  /// the architecture JSON). Keys must not collide with the ones Save() sets itself.
  template <typename NetworkType>
  static bool Save(const NetworkType& net, const std::string& filepath,
                   const std::map<std::string, std::string>& metadata = {})
  {
    return Write(net, filepath, metadata, /*checkpoint=*/false);
  }

  /// @brief Like Save(), but also writes everything needed to resume training: Adam/AdamW
  /// moments and step counters for every layer. The file has `deepity.kind = "checkpoint"`;
  /// Load() still reads its weights, LoadCheckpoint() restores the optimizer state too.
  /// Caller state (epoch, step, learning-rate position, RNG) goes in `metadata`.
  /// @throws std::runtime_error for a GPU network or a layer without optimizer-state accessors.
  template <typename NetworkType>
  static bool SaveCheckpoint(const NetworkType& net, const std::string& filepath,
                             const std::map<std::string, std::string>& metadata = {})
  {
    detail::RequireCheckpointSupport(net);
    return Write(net, filepath, metadata, /*checkpoint=*/true);
  }

  /// @brief Returns the `__metadata__` string map of a model file after validating it, without
  /// hashing the data. Use it to read the architecture before building a network.
  /// @throws io::SafeTensorsError for a missing, malformed or truncated file, or a file written
  /// with a newer `deepity.format_version` than this build understands.
  static std::map<std::string, std::string> ReadMetadata(const std::string& filepath)
  {
    const io::SafeTensorsReader reader(filepath, /*verifyHash=*/false);
    const auto& meta = reader.Metadata();
    const auto it = meta.find("deepity.format_version");
    if (it != meta.end() && it->second != "1")
      throw io::SafeTensorsError("ModelIO: " + filepath + " has deepity.format_version " + it->second +
                                 ", but this build only reads version 1.");
    return meta;
  }

  /// @brief Reads a file written by Save() (or by 0.1.0a8 and earlier) back into `net`'s
  /// already-compiled weight arena. The whole file is validated first, including its hash.
  /// A checkpoint file loads too; its optimizer state is ignored.
  /// @return true. Failure is reported by exception, so the reason is never lost.
  /// @throws io::SafeTensorsError for a missing, malformed, truncated or corrupted file.
  /// @throws std::runtime_error for a dtype or shape that doesn't match the network.
  template <typename NetworkType> static bool Load(NetworkType& net, const std::string& filepath)
  {
    const io::SafeTensorsReader reader(filepath);
    ReadWeights(net, reader);
    return true;
  }

  /// @brief Loads a checkpoint written by SaveCheckpoint(): weights, then Adam/AdamW moments and
  /// step counters, so training continues as if it had never stopped.
  /// @throws std::runtime_error if the file is not a checkpoint, lacks optimizer state for a
  /// layer, or does not match the network, plus everything Load() throws.
  template <typename NetworkType>
  static bool LoadCheckpoint(NetworkType& net, const std::string& filepath)
  {
    detail::RequireCheckpointSupport(net);
    const io::SafeTensorsReader reader(filepath);
    const auto kind = reader.Metadata().find("deepity.kind");
    if (kind == reader.Metadata().end() || kind->second != "checkpoint")
      throw std::runtime_error("ModelIO::LoadCheckpoint: " + filepath +
                               " is not a checkpoint (deepity.kind is '" +
                               (kind == reader.Metadata().end() ? std::string("missing") : kind->second) +
                               "'). Use Load() for a weights-only file.");
    ReadWeights(net, reader);

    size_t layerIdx = 0;
    for (auto& layerPtr : net.GetLayers())
    {
      if constexpr (requires { layerPtr->GetOptimizerStateDict(); })
      {
      for (const auto& [name, tensor] : layerPtr->GetOptimizerStateDict())
      {
        const std::string key = detail::OptimizerTensorName(layerIdx, name);
        const io::TensorEntry* entry = reader.Find(key);
        if (entry == nullptr)
          throw std::runtime_error("ModelIO::LoadCheckpoint: '" + key + "' is missing from " + filepath + ".");
        using MomentType = std::remove_pointer_t<decltype(tensor.data)>;
        if (entry->dtype != DTypeTag<MomentType>::Name() || entry->shape != tensor.shape)
          throw std::runtime_error("ModelIO::LoadCheckpoint: dtype or shape mismatch for '" + key + "'.");
        detail::ReadTensorInto(reader, *entry, tensor.shape, tensor);
      }
      for (const auto& [name, counter] : layerPtr->GetOptimizerCounters())
      {
        const std::string key = detail::OptimizerTensorName(layerIdx, name);
        const io::TensorEntry* entry = reader.Find(key);
        if (entry == nullptr || entry->dtype != "I32" || entry->shape != std::vector<size_t>{1})
          throw std::runtime_error("ModelIO::LoadCheckpoint: '" + key + "' is missing or malformed in " + filepath + ".");
        reader.Read(*entry, counter, sizeof(int32_t));
      }
      }
      ++layerIdx;
    }
    return true;
  }

private:
  template <typename NetworkType>
  static bool Write(const NetworkType& net, const std::string& filepath,
                    const std::map<std::string, std::string>& metadata, bool checkpoint)
  {
    io::SafeTensorsWriter writer;
    writer.SetMetadata("deepity.format_version", "1");
    for (const auto& [key, value] : metadata)
      writer.SetMetadata(key, value);
    writer.SetMetadata("deepity.kind", checkpoint ? "checkpoint" : "inference");

    std::vector<std::vector<char>> packed;
    size_t layerIdx = 0;
    for (const auto& layerPtr : net.GetLayers())
    {
      // Pushes a device-resident layer's current state into its host
      // staging buffers before GetStateDict() reads them. No-op for a
      // CPU layer.
      detail::SyncFromDeviceIfSupported(layerPtr);
      for (const auto& [name, tensor] : layerPtr->GetStateDict())
        detail::AddTensorPacked(writer, packed, detail::TensorFileName(layerIdx, name), tensor);
      // Layer types without the accessors never reach here with `checkpoint` set, see
      // RequireCheckpointSupport(); the requires-check just keeps them compiling.
      if constexpr (requires { layerPtr->GetOptimizerStateDict(); })
        if (checkpoint)
        {
          for (const auto& [name, tensor] : layerPtr->GetOptimizerStateDict())
            detail::AddTensorPacked(writer, packed, detail::OptimizerTensorName(layerIdx, name), tensor);
          for (const auto& [name, counter] : layerPtr->GetOptimizerCounters())
            writer.AddTensor(detail::OptimizerTensorName(layerIdx, name), "I32", {1}, counter, sizeof(int32_t));
        }
      ++layerIdx;
    }
    writer.Write(filepath);
    return true;
  }

  template <typename NetworkType>
  static void ReadWeights(NetworkType& net, const io::SafeTensorsReader& reader)
  {
    size_t layerIdx = 0;
    for (auto& layerPtr : net.GetLayers())
    {
      auto stateDict = layerPtr->GetStateDict();
      for (auto& [name, tensor] : stateDict)
      {
        using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
        const std::string key = detail::TensorFileName(layerIdx, name);
        const io::TensorEntry* entry = reader.Find(key);
        if (entry == nullptr)
          entry = reader.Find(detail::LegacyTensorFileName(layerIdx, name));
        if (entry == nullptr)
          continue;

        const char* expectedDtype = DTypeTag<ElementType>::Name();
        if (entry->dtype != expectedDtype)
          throw std::runtime_error("ModelIO::Load: dtype mismatch for '" + key + "' -- file has " +
                                   entry->dtype + ", network expects " + expectedDtype + ".");

        // Files written before 0.1.0a9 stored Psi as [nextSize, terminal] although the buffer is
        // [size, terminal], dropping rows. Read what such a file has; anything else must match.
        std::vector<size_t> readShape = tensor.shape;
        if (entry->shape != tensor.shape)
        {
          const bool legacyTruncatedPsi = name == "Psi" && entry->shape.size() == 2 &&
                                          tensor.shape.size() == 2 &&
                                          entry->shape[1] == tensor.shape[1] &&
                                          entry->shape[0] < tensor.shape[0];
          if (!legacyTruncatedPsi)
            throw std::runtime_error("ModelIO::Load: shape mismatch for '" + key + "'.");
          readShape = entry->shape;
        }
        detail::ReadTensorInto(reader, *entry, readShape, tensor);
      }
      // Pushes the reads above from host staging buffers on to the
      // actual device-resident parameters. No-op for a CPU layer.
      detail::SyncToDeviceIfSupported(layerPtr);
      ++layerIdx;
    }
  }
};
} // namespace Deep
