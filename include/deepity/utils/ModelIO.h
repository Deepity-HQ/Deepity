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
 * @brief Saves/loads a network's compiled weight arena to/from disk, as a
 * JSON header (per-layer shapes/offsets/dtypes) plus a raw binary data
 * blob. Templated on NetworkType the same as before (works across every
 * PC network variant that exposes GetLayers()), but the element type
 * backing each tensor is no longer assumed to be `float`: Save() derives
 * it from each tensor's own `.data` pointer (via DTypeTag<T>), so a
 * FullPCNetworkT<double, ...> writes real "F64" tensors at their real
 * 8-byte stride, not truncated through float. Load() checks that
 * dtype against what the CALLING network's own tensors expect and
 * throws std::runtime_error on a mismatch -- strictly new behavior:
 * the previous version never checked "dtype" on read at all.
 */

namespace Deep
{
namespace detail
{
// FullPCLayerT<T, AccumT> is free-standing (doesn't inherit Layer, see
// FullPCLayerT.h's own doc comment on why), so it has no
// SyncStateDictFromDevice/ToDevice at all -- only Layer declares those.
// These two helpers let Save()/Load() call them where they exist
// (every Layer-derived network) and skip them where they don't, instead
// of forcing every NetworkType's layer to carry a method it has no use
// for. NOTE: this means a FullPCLayerT running on a CUDA device has no
// sync path here at all -- unlike FullConvPCLayer, it was never given
// host-staging buffers in the first place, so a GetStateDict() call on
// a device-resident FullPCLayerT is not proven safe. Not fixed here;
// flagging it is the point of this comment.
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
      // A device-resident layer's GetStateDict() reads from host staging
      // buffers it owns, not from the actual device memory -- those
      // buffers only hold current data once this sync runs. No-op for a
      // CPU layer, whose GetStateDict() already returns live pointers.
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
        size_t num_elements = 1;
        for (size_t dim : tensor.shape)
        {
          num_elements *= dim;
        }
        out.write(reinterpret_cast<const char*>(tensor.data), num_elements * sizeof(ElementType));
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
          size_t end_offset = header[key]["data_offsets"][1];
          size_t byte_size = end_offset - start_offset;

          in.seekg(base_offset + start_offset, std::ios::beg);
          in.read(reinterpret_cast<char*>(tensor.data), byte_size);
        }
      }
      // Mirror of Save()'s SyncStateDictFromDevice(): for a device-resident
      // layer, the reads above only landed in its host staging buffers --
      // this pushes them on to the actual device-resident parameters. No-op
      // for a CPU layer, whose GetStateDict() pointers are the real buffers.
      detail::SyncToDeviceIfSupported(layerPtr);
      layer_idx++;
    }
    return true;
  }
};
} // namespace Deep