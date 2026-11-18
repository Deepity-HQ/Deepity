#pragma once
#include <algorithm>
#include <cstdint>
#include <deepity/backend/DeviceType.h>
#include <deepity/backend/IComputeBackend.h>
#include <deepity/layers/Layer.h>
#include <deepity/io/SafeTensors.h>
#include <deepity/utils/DTypeTag.h>
#include <deque>
#include <map>
#include <stdexcept>
#include <functional>
#include <set>
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

template <typename LayerPtr> IComputeBackend* StateBackendOf(const LayerPtr& layerPtr)
{
  if constexpr (requires { layerPtr->DeviceStateBackend(); })
    return layerPtr->DeviceStateBackend();
  return nullptr;
}

template <typename LayerPtr> IComputeBackend* CounterBackendOf(const LayerPtr& layerPtr)
{
  if constexpr (requires { layerPtr->DeviceCounterBackend(); })
    return layerPtr->DeviceCounterBackend();
  return nullptr;
}

/// @brief Element count from the first element to the last, rows included.
inline size_t TensorExtent(const std::vector<size_t>& shape, size_t rowStride)
{
  size_t rows = 1, width = shape.empty() ? 0 : shape.back();
  for (size_t i = 0; i + 1 < shape.size(); ++i)
    rows *= shape[i];
  const size_t stride = rowStride == 0 ? width : rowStride;
  return rows == 0 || width == 0 ? 0 : (rows - 1) * stride + width;
}

/// @brief Host tensors need no staging.
template <typename Descriptor>
Descriptor StagedForSave(IComputeBackend*, const Descriptor& tensor, std::vector<std::vector<float>>&)
{
  return tensor;
}

/// @brief Copies a device tensor into a host buffer kept alive in `keep` and returns a descriptor for it.
inline TensorDescriptor StagedForSave(IComputeBackend* dev, const TensorDescriptor& tensor,
                                      std::vector<std::vector<float>>& keep)
{
  if (dev == nullptr)
    return tensor;
  keep.emplace_back(TensorExtent(tensor.shape, tensor.rowStride));
  dev->Synchronize();
  dev->CopyToHost(keep.back().data(), tensor.data, keep.back().size());
  TensorDescriptor host = tensor;
  host.data = keep.back().data();
  return host;
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

/// @brief Adds `tensor` to `writer` under `fileName`. Padded rows are packed tight, and types whose
/// on-disk layout differs (4-bit codes) are encoded; both go into `scratch`, which must outlive the
/// writer's Write() because the writer keeps pointers.
template <typename Descriptor>
void AddTensorPacked(io::SafeTensorsWriter& writer, std::vector<std::vector<char>>& scratch,
                     const std::string& fileName, const Descriptor& tensor)
{
  using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
  const char* dtypeName = DTypeTag<ElementType>::Name();
  const io::DTypeInfo* info = io::FindDType(dtypeName);
  const auto [rows, width] = RowsAndWidth(tensor.shape);
  const size_t count = rows * width, rowBytes = width * sizeof(ElementType);

  const void* tight = tensor.data;
  if (IsPadded(tensor.shape, tensor.rowStride))
  {
    scratch.emplace_back(count * sizeof(ElementType));
    for (size_t r = 0; r < rows; ++r)
      std::copy_n(reinterpret_cast<const char*>(tensor.data + r * tensor.rowStride), rowBytes,
                  scratch.back().data() + r * rowBytes);
    tight = scratch.back().data();
  }
  if (info != nullptr && info->kind == io::DTypeKind::Packed)
  {
    if (sizeof(ElementType) != info->memBytes)
      throw std::runtime_error(std::string("ModelIO: in-memory size of ") + dtypeName + " does not match the registry.");
    scratch.emplace_back(io::StorageBytes(*info, count));
    info->pack(static_cast<const uint8_t*>(tight), count, reinterpret_cast<uint8_t*>(scratch.back().data()));
    writer.AddTensor(fileName, dtypeName, tensor.shape, scratch.back().data(), scratch.back().size());
    return;
  }
  writer.AddTensor(fileName, dtypeName, tensor.shape, tight, count * sizeof(ElementType));
}

template <typename Descriptor>
void ReadTensorInto(const io::SafeTensorsReader& reader, const io::TensorEntry& entry,
                    const std::vector<size_t>& readShape, const Descriptor& tensor);

/// @brief Reads into `tensor`, going through a host buffer and the backend's copy when the tensor is
/// device memory. The buffer starts as the device contents so padding between rows is kept.
template <typename Descriptor>
void ReadTensorIntoStaged(IComputeBackend* dev, const io::SafeTensorsReader& reader, const io::TensorEntry& entry,
                          const std::vector<size_t>& readShape, const Descriptor& tensor)
{
  ReadTensorInto(reader, entry, readShape, tensor);
}

inline void ReadTensorIntoStaged(IComputeBackend* dev, const io::SafeTensorsReader& reader,
                                 const io::TensorEntry& entry, const std::vector<size_t>& readShape,
                                 const TensorDescriptor& tensor)
{
  if (dev == nullptr)
  {
    ReadTensorInto(reader, entry, readShape, tensor);
    return;
  }
  std::vector<float> host(TensorExtent(tensor.shape, tensor.rowStride));
  dev->Synchronize();
  dev->CopyToHost(host.data(), tensor.data, host.size());
  TensorDescriptor staged = tensor;
  staged.data = host.data();
  ReadTensorInto(reader, entry, readShape, staged);
  dev->CopyFromHost(tensor.data, host.data(), host.size());
}

/// @brief Reads `entry` (whose shape is `readShape`) into `tensor`, decoding packed types and
/// spreading rows over the tensor's row stride when it is padded.
template <typename Descriptor>
void ReadTensorInto(const io::SafeTensorsReader& reader, const io::TensorEntry& entry,
                    const std::vector<size_t>& readShape, const Descriptor& tensor)
{
  using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
  const auto [rows, width] = RowsAndWidth(readShape);
  const size_t count = rows * width;
  if (!entry.packed && !IsPadded(readShape, tensor.rowStride))
  {
    reader.Read(entry, tensor.data, count * sizeof(ElementType));
    return;
  }
  std::vector<ElementType> tight(count);
  if (entry.packed)
  {
    const io::DTypeInfo* info = io::FindDType(entry.dtype);
    if (sizeof(ElementType) != info->memBytes)
      throw std::runtime_error("ModelIO::Load: '" + entry.name + "' holds " + entry.dtype +
                               ", which does not match the network's element type.");
    std::vector<uint8_t> bytes(entry.ByteSize());
    reader.Read(entry, bytes.data(), bytes.size());
    info->unpack(bytes.data(), count, reinterpret_cast<uint8_t*>(tight.data()));
  }
  else
    reader.Read(entry, tight.data(), count * sizeof(ElementType));
  const size_t stride = tensor.rowStride == 0 ? width : tensor.rowStride;
  for (size_t r = 0; r < rows; ++r)
    std::copy_n(tight.data() + r * width, width, tensor.data + r * stride);
}

/// @brief Throws unless every layer can describe all of its optimizer state.
template <typename NetworkType> void RequireCheckpointSupport(const NetworkType& net)
{
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

  /// @brief Like Save(), but splits the file over several shards of at most `maxShardBytes` each
  /// (see SafeTensorsWriter::WriteSharded). `indexPath` must end in `.safetensors.index.json`;
  /// Load(), LoadCheckpoint() and ReadMetadata() accept that index path as they do a file path.
  template <typename NetworkType>
  static bool SaveSharded(const NetworkType& net, const std::string& indexPath, uint64_t maxShardBytes,
                          const std::map<std::string, std::string>& metadata = {})
  {
    return Write(net, indexPath, metadata, /*checkpoint=*/false, maxShardBytes);
  }

  /// @brief Sharded form of SaveCheckpoint().
  template <typename NetworkType>
  static bool SaveCheckpointSharded(const NetworkType& net, const std::string& indexPath, uint64_t maxShardBytes,
                                    const std::map<std::string, std::string>& metadata = {})
  {
    detail::RequireCheckpointSupport(net);
    return Write(net, indexPath, metadata, /*checkpoint=*/true, maxShardBytes);
  }

  /// @brief Like Save(), but also writes everything needed to resume training: Adam/AdamW
  /// moments and step counters for every layer. The file has `deepity.kind = "checkpoint"`;
  /// Load() still reads its weights, LoadCheckpoint() restores the optimizer state too.
  /// Caller state (epoch, step, learning-rate position, RNG) goes in `metadata`.
  /// @throws std::runtime_error for a layer without optimizer-state accessors.
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
    ReadWeights(net, reader, nullptr, nullptr);
    return true;
  }

  /// @brief Which names did not line up in a LoadWeights() call.
  struct LoadReport
  {
    std::vector<std::string> missing;    ///< Network tensors the file had no tensor for (left unchanged).
    std::vector<std::string> unexpected; ///< File tensors no network tensor used (optimizer state excluded).
  };

  /// @brief Loads weights from any safetensors file, for example one exported from PyTorch.
  ///
  /// `rename` maps each file tensor name to the Deepity name it should fill (`layers.0.weight`);
  /// it may be empty if the names already match. With `strict`, any missing or unexpected tensor
  /// throws, naming all of them. Without it, they are reported and the rest still loads. Dtype and
  /// shape must match either way.
  template <typename NetworkType>
  static LoadReport LoadWeights(NetworkType& net, const std::string& filepath, bool strict = true,
                                const std::function<std::string(const std::string&)>& rename = {})
  {
    const io::SafeTensorsReader reader(filepath);
    LoadReport report;
    ReadWeights(net, reader, rename ? &rename : nullptr, &report);
    if (strict && (!report.missing.empty() || !report.unexpected.empty()))
    {
      std::string msg = "ModelIO::LoadWeights: names do not match.";
      for (const auto& n : report.missing)
        msg += " Missing from file: " + n + ".";
      for (const auto& n : report.unexpected)
        msg += " Not used by the network: " + n + ".";
      throw std::runtime_error(msg);
    }
    return report;
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
    ReadWeights(net, reader, nullptr, nullptr);

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
        detail::ReadTensorIntoStaged(detail::StateBackendOf(layerPtr), reader, *entry, tensor.shape, tensor);
      }
      for (const auto& [name, counter] : layerPtr->GetOptimizerCounters())
      {
        const std::string key = detail::OptimizerTensorName(layerIdx, name);
        const io::TensorEntry* entry = reader.Find(key);
        if (entry == nullptr || entry->dtype != "I32" || entry->shape != std::vector<size_t>{1})
          throw std::runtime_error("ModelIO::LoadCheckpoint: '" + key + "' is missing or malformed in " + filepath + ".");
        if (IComputeBackend* counterDev = detail::CounterBackendOf(layerPtr); counterDev != nullptr)
        {
          int32_t value = 0;
          reader.Read(*entry, &value, sizeof(int32_t));
          counterDev->CopyCounterFromHost(counter, &value);
        }
        else
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
                    const std::map<std::string, std::string>& metadata, bool checkpoint, uint64_t maxShardBytes = 0)
  {
    io::SafeTensorsWriter writer;
    writer.SetMetadata("deepity.format_version", "1");
    for (const auto& [key, value] : metadata)
      writer.SetMetadata(key, value);
    writer.SetMetadata("deepity.kind", checkpoint ? "checkpoint" : "inference");

    std::vector<std::vector<char>> packed;
    std::vector<std::vector<float>> deviceCopies; // host copies of device tensors, alive until Write()
    std::deque<int32_t> counterCopies;
    size_t layerIdx = 0, tensorCount = 0;
    for (const auto& layerPtr : net.GetLayers())
    {
      // Pushes a device-resident layer's current state into its host
      // staging buffers before GetStateDict() reads them. No-op for a
      // CPU layer.
      detail::SyncFromDeviceIfSupported(layerPtr);
      IComputeBackend* const dev = detail::StateBackendOf(layerPtr);
      for (const auto& [name, tensor] : layerPtr->GetStateDict())
      {
        detail::AddTensorPacked(writer, packed, detail::TensorFileName(layerIdx, name),
                                detail::StagedForSave(dev, tensor, deviceCopies));
        ++tensorCount;
      }
      // Layer types without the accessors never reach here with `checkpoint` set, see
      // RequireCheckpointSupport(); the requires-check just keeps them compiling.
      if constexpr (requires { layerPtr->GetOptimizerStateDict(); })
        if (checkpoint)
        {
          for (const auto& [name, tensor] : layerPtr->GetOptimizerStateDict())
            detail::AddTensorPacked(writer, packed, detail::OptimizerTensorName(layerIdx, name),
                                    detail::StagedForSave(dev, tensor, deviceCopies));
          IComputeBackend* const counterDev = detail::CounterBackendOf(layerPtr);
          for (const auto& [name, counter] : layerPtr->GetOptimizerCounters())
          {
            const int* source = counter;
            if (counterDev != nullptr)
            {
              counterCopies.emplace_back(0);
              counterDev->CopyCounterToHost(&counterCopies.back(), counter);
              source = &counterCopies.back();
            }
            writer.AddTensor(detail::OptimizerTensorName(layerIdx, name), "I32", {1}, source, sizeof(int32_t));
          }
        }
      ++layerIdx;
    }
    if (layerIdx > 0 && tensorCount == 0)
      throw std::runtime_error("ModelIO: this network exposes no weights to save (its layers have no "
                               "state dict yet), so the file would be empty.");
    if (maxShardBytes > 0)
      writer.WriteSharded(filepath, maxShardBytes);
    else
      writer.Write(filepath);
    return true;
  }

  template <typename NetworkType>
  static void ReadWeights(NetworkType& net, const io::SafeTensorsReader& reader,
                          const std::function<std::string(const std::string&)>* rename, LoadReport* report)
  {
    // File name (after renaming) to entry, so renamed and plain files share one lookup.
    std::map<std::string, const io::TensorEntry*> byName;
    std::set<std::string> used;
    if (rename != nullptr)
      for (const auto& t : reader.Tensors())
        if (!byName.emplace((*rename)(t.name), &t).second)
          throw std::runtime_error("ModelIO::LoadWeights: rename maps two file tensors to '" + (*rename)(t.name) + "'.");
    auto find = [&](const std::string& key) -> const io::TensorEntry*
    {
      if (rename == nullptr)
        return reader.Find(key);
      const auto it = byName.find(key);
      return it == byName.end() ? nullptr : it->second;
    };
    size_t layerIdx = 0;
    for (auto& layerPtr : net.GetLayers())
    {
      auto stateDict = layerPtr->GetStateDict();
      for (auto& [name, tensor] : stateDict)
      {
        using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
        const std::string key = detail::TensorFileName(layerIdx, name);
        const io::TensorEntry* entry = find(key);
        if (entry == nullptr)
          entry = find(detail::LegacyTensorFileName(layerIdx, name));
        if (entry == nullptr)
        {
          if (report != nullptr)
            report->missing.push_back(key);
          continue;
        }
        used.insert(entry->name);

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
        detail::ReadTensorIntoStaged(detail::StateBackendOf(layerPtr), reader, *entry, readShape, tensor);
      }
      // Pushes the reads above from host staging buffers on to the
      // actual device-resident parameters. No-op for a CPU layer.
      detail::SyncToDeviceIfSupported(layerPtr);
      // Layers that cache log(precision) must rebuild it from the p they just loaded.
      if constexpr (requires { layerPtr->ResyncLogPrecision(); })
        layerPtr->ResyncLogPrecision();
      ++layerIdx;
    }
    if (report != nullptr)
      for (const auto& t : reader.Tensors())
        if (used.count(t.name) == 0 && t.name.rfind("optimizer.", 0) != 0)
          report->unexpected.push_back(t.name);
  }
};
} // namespace Deep
