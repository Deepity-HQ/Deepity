#pragma once
#include <algorithm>
#include <cstdint>
#include <deepity/io/SafeTensors.h>
#include <deepity/utils/DTypeTag.h>
#include <filesystem>
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
} // namespace detail

/// @brief Static save/load helpers for any network type exposing
/// GetLayers(), templated so it works across every PC network
/// variant without each one reimplementing serialization.
class ModelIO
{
public:
  /// @brief Writes `net`'s weights to a single safetensors file at `filepath`, atomically, with
  /// a content hash and `deepity.*` metadata. Padded tensors are written without their padding.
  /// @return false if `filepath` couldn't be written; true on success.
  template <typename NetworkType>
  static bool Save(const NetworkType& net, const std::string& filepath)
  {
    io::SafeTensorsWriter writer;
    writer.SetMetadata("deepity.format_version", "1");
    writer.SetMetadata("deepity.kind", "inference");

    // Padded tensors are packed here; the writer keeps pointers until Write() returns.
    std::vector<std::vector<char>> packed;
    size_t layerIdx = 0;
    for (const auto& layerPtr : net.GetLayers())
    {
      // Pushes a device-resident layer's current state into its host
      // staging buffers before GetStateDict() reads them. No-op for a
      // CPU layer.
      detail::SyncFromDeviceIfSupported(layerPtr);
      for (const auto& [name, tensor] : layerPtr->GetStateDict())
      {
        using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
        const auto [rows, width] = detail::RowsAndWidth(tensor.shape);
        const size_t rowBytes = width * sizeof(ElementType);
        const void* source = tensor.data;
        if (detail::IsPadded(tensor.shape, tensor.rowStride))
        {
          packed.emplace_back(rows * rowBytes);
          for (size_t r = 0; r < rows; ++r)
            std::copy_n(reinterpret_cast<const char*>(tensor.data + r * tensor.rowStride), rowBytes,
                        packed.back().data() + r * rowBytes);
          source = packed.back().data();
        }
        try
        {
          writer.AddTensor(detail::TensorFileName(layerIdx, name), DTypeTag<ElementType>::Name(),
                           tensor.shape, source, rows * rowBytes);
        }
        catch (const io::SafeTensorsError&)
        {
          return false;
        }
      }
      ++layerIdx;
    }

    try
    {
      writer.Write(filepath);
    }
    catch (const io::SafeTensorsError&)
    {
      return false;
    }
    return true;
  }

  /// @brief Reads a file written by Save() (or by 0.1.0a8 and earlier) back into `net`'s
  /// already-compiled weight arena. The whole file is validated first, including its hash.
  /// @return false if `filepath` doesn't exist; true on success.
  /// @throws io::SafeTensorsError for a malformed, truncated or corrupted file.
  /// @throws std::runtime_error for a dtype or shape that doesn't match the network.
  template <typename NetworkType> static bool Load(NetworkType& net, const std::string& filepath)
  {
    if (!std::filesystem::exists(filepath))
      return false;
    const io::SafeTensorsReader reader(filepath);

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

        const auto [rows, width] = detail::RowsAndWidth(readShape);
        if (!detail::IsPadded(readShape, tensor.rowStride))
        {
          reader.Read(*entry, tensor.data, rows * width * sizeof(ElementType));
          continue;
        }
        std::vector<ElementType> tight(rows * width);
        reader.Read(*entry, tight.data(), tight.size() * sizeof(ElementType));
        for (size_t r = 0; r < rows; ++r)
          std::copy_n(tight.data() + r * width, width, tensor.data + r * tensor.rowStride);
      }
      // Pushes the reads above from host staging buffers on to the
      // actual device-resident parameters. No-op for a CPU layer.
      detail::SyncToDeviceIfSupported(layerPtr);
      ++layerIdx;
    }
    return true;
  }
};
} // namespace Deep
