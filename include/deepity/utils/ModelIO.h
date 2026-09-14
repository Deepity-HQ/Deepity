#pragma once
#include <string>
#include <fstream>
#include <cstdint>
#include <nlohmann/json.hpp>

namespace Deep
{
    class ModelIO
    {
    public:
        template <typename NetworkType>
        static bool Save(const NetworkType &net, const std::string &filepath)
        {
            nlohmann::json header;
            size_t current_offset = 0;
            int layer_idx = 0;

            for (const auto &layerPtr : net.GetLayers())
            {
                auto state_dict = layerPtr->GetStateDict();

                for (const auto &[name, tensor] : state_dict)
                {
                    std::string key = "layer_" + std::to_string(layer_idx) + "." + name;

                    size_t num_elements = 1;
                    for (size_t dim : tensor.shape)
                    {
                        num_elements *= dim;
                    }
                    size_t byte_size = num_elements * sizeof(float);

                    header[key] = {
                        {"dtype", "F32"},
                        {"shape", tensor.shape},
                        {"data_offsets", {current_offset, current_offset + byte_size}}};
                    current_offset += byte_size;
                }
                layer_idx++;
            }

            header["__metadata__"] = {{"format", "deepity_safetensors"}};

            std::string header_str = header.dump();
            uint64_t header_size = header_str.size();

            std::ofstream out(filepath, std::ios::binary);
            if (!out.is_open())
                return false;

            out.write(reinterpret_cast<const char *>(&header_size), sizeof(uint64_t));
            out.write(header_str.data(), header_size);

            for (const auto &layerPtr : net.GetLayers())
            {
                for (const auto &[name, tensor] : layerPtr->GetStateDict())
                {
                    size_t num_elements = 1;
                    for (size_t dim : tensor.shape)
                    {
                        num_elements *= dim;
                    }
                    out.write(reinterpret_cast<const char *>(tensor.data), num_elements * sizeof(float));
                }
            }
            return true;
        }

        template <typename NetworkType>
        static bool Load(NetworkType &net, const std::string &filepath)
        {
            std::ifstream in(filepath, std::ios::binary);
            if (!in.is_open())
                return false;

            uint64_t header_size;
            in.read(reinterpret_cast<char *>(&header_size), sizeof(uint64_t));

            std::string header_str(header_size, '\0');
            in.read(&header_str[0], header_size);

            auto header = nlohmann::json::parse(header_str);
            size_t base_offset = sizeof(uint64_t) + header_size;

            int layer_idx = 0;
            for (auto &layerPtr : net.GetLayers())
            {
                auto state_dict = layerPtr->GetStateDict();
                for (auto &[name, tensor] : state_dict)
                {
                    std::string key = "layer_" + std::to_string(layer_idx) + "." + name;

                    if (header.contains(key))
                    {
                        size_t start_offset = header[key]["data_offsets"][0];
                        size_t end_offset = header[key]["data_offsets"][1];
                        size_t byte_size = end_offset - start_offset;

                        in.seekg(base_offset + start_offset, std::ios::beg);
                        in.read(reinterpret_cast<char *>(tensor.data), byte_size);
                    }
                }
                layer_idx++;
            }
            return true;
        }
    };
}