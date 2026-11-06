#pragma once
#include <cstdint>
#include <deepity/utils/DTypeTag.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>

/**
 * @file ModelIO.h
 * @brief Saves/loads a network's compiled weight arena to/from disk,
 * as a JSON header (per-layer shapes/offsets/dtypes) plus a raw
 * binary data blob. Templated on NetworkType, so it works across
 * every PC network variant that exposes GetLayers(). The element
 * type backing each tensor is derived from each tensor's own `.data`
 * pointer (via DTypeTag<T>), not assumed to be `float`. Load() checks
 * the file's dtype against what the calling network's own tensors
 * expect and throws std::runtime_error on a mismatch.
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

/// @brief Writes a tensor's real (unpadded) elements to `out`, one row
/// at a time when rowStride requires it, so a padded buffer like a
/// DirectKPPCLayer run's Psi never has its padding columns serialized.
template <typename ElementType>
void WriteTensorData(std::ofstream& out, const ElementType* data, const std::vector<size_t>& shape,
                     size_t rowStride)
{
  auto [rows, width] = RowsAndWidth(shape);
  if (rowStride == 0 || rowStride == width)
  {
    out.write(reinterpret_cast<const char*>(data), rows * width * sizeof(ElementType));
    return;
  }
  for (size_t r = 0; r < rows; ++r)
    out.write(reinterpret_cast<const char*>(data + r * rowStride), width * sizeof(ElementType));
}

/// @brief Counterpart to WriteTensorData() for Load().
template <typename ElementType>
void ReadTensorData(std::ifstream& in, ElementType* data, const std::vector<size_t>& shape,
                    size_t rowStride)
{
  auto [rows, width] = RowsAndWidth(shape);
  if (rowStride == 0 || rowStride == width)
  {
    in.read(reinterpret_cast<char*>(data), rows * width * sizeof(ElementType));
    return;
  }
  for (size_t r = 0; r < rows; ++r)
    in.read(reinterpret_cast<char*>(data + r * rowStride), width * sizeof(ElementType));
}
} // namespace detail

/// @brief Static save/load helpers for any network type exposing
/// GetLayers(), templated so it works across every PC network
/// variant without each one reimplementing serialization.
class ModelIO
{
public:
  /// @brief Writes `net`'s weights to a single file at `filepath`:
  /// a JSON header (per-layer tensor shapes/offsets) followed by
  /// the raw float data, back to back.
  /// @return false if `filepath` couldn't be opened for writing;
  /// true on success.
  template <typename NetworkType>
  static bool Save(const NetworkType& net, const std::string& filepath)
  {
    nlohmann::json header;
    size_t current_offset = 0;
    int layer_idx = 0;
    std::string fileDtype = "F32"; // stays "F32" for an empty network, matching old behavior.

    for (const auto& layerPtr : net.GetLayers())
    {
      // Pushes a device-resident layer's current state into its host
      // staging buffers before GetStateDict() reads them. No-op for a
      // CPU layer.
      detail::SyncFromDeviceIfSupported(layerPtr);
      auto state_dict = layerPtr->GetStateDict();

      for (const auto& [name, tensor] : state_dict)
      {
        using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
        fileDtype = DTypeTag<ElementType>::Name();

        std::string key = "layer_" + std::to_string(layer_idx) + "." + name;

        size_t num_elements = 1;
        for (size_t dim : tensor.shape)
        {
          num_elements *= dim;
        }
        size_t byte_size = num_elements * sizeof(ElementType);

        header[key] = {{"dtype", fileDtype},
                       {"shape", tensor.shape},
                       {"data_offsets", {current_offset, current_offset + byte_size}}};
        current_offset += byte_size;
      }
      layer_idx++;
    }

    header["__metadata__"] = {{"format", "deepity_safetensors"}, {"dtype", fileDtype}};

    std::string header_str = header.dump();
    uint64_t header_size = header_str.size();

    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open())
      return false;

    out.write(reinterpret_cast<const char*>(&header_size), sizeof(uint64_t));
    out.write(header_str.data(), header_size);

    for (const auto& layerPtr : net.GetLayers())
    {
      for (const auto& [name, tensor] : layerPtr->GetStateDict())
      {
        using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
        detail::WriteTensorData(out, tensor.data, tensor.shape, tensor.rowStride);
      }
    }
    return true;
  }

  /// @brief Reads a file written by Save() back into `net`'s
  /// already-compiled weight arena, matching each tensor by the
  /// same "layer_N.name" key Save() wrote it under.
  /// @return false if `filepath` couldn't be opened for reading;
  /// true on success.
  template <typename NetworkType> static bool Load(NetworkType& net, const std::string& filepath)
  {
    std::ifstream in(filepath, std::ios::binary);
    if (!in.is_open())
      return false;

    uint64_t header_size;
    in.read(reinterpret_cast<char*>(&header_size), sizeof(uint64_t));

    std::string header_str(header_size, '\0');
    in.read(&header_str[0], header_size);

    auto header = nlohmann::json::parse(header_str);
    size_t base_offset = sizeof(uint64_t) + header_size;

    int layer_idx = 0;
    for (auto& layerPtr : net.GetLayers())
    {
      auto state_dict = layerPtr->GetStateDict();
      for (auto& [name, tensor] : state_dict)
      {
        using ElementType = std::remove_pointer_t<decltype(tensor.data)>;
        std::string key = "layer_" + std::to_string(layer_idx) + "." + name;

        if (header.contains(key))
        {
          std::string fileDtype = header[key].value("dtype", std::string());
          const char* expectedDtype = DTypeTag<ElementType>::Name();
          if (!fileDtype.empty() && fileDtype != expectedDtype)
          {
            throw std::runtime_error("ModelIO::Load: dtype mismatch for '" + key +
                                     "' -- file "
                                     "has " +
                                     fileDtype + ", network expects " + expectedDtype + ".");
          }

          size_t start_offset = header[key]["data_offsets"][0];

          in.seekg(base_offset + start_offset, std::ios::beg);
          detail::ReadTensorData(in, tensor.data, tensor.shape, tensor.rowStride);
        }
      }
      // Pushes the reads above from host staging buffers on to the
      // actual device-resident parameters. No-op for a CPU layer.
      detail::SyncToDeviceIfSupported(layerPtr);
      layer_idx++;
    }
    return true;
  }
};
} // namespace Deep