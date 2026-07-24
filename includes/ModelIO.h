#pragma once
#include <string>
#include "DiscriminativePCNetwork.h"

namespace Deep
{
    class ModelIO
    {
    public:
        // Saves the network into a structured directory (manifest.json, weights.bin, README.md)
        static bool Save(const DiscriminativePCNetwork &net, const std::string &dirPath);

        // Loads a network state from a structured directory
        static bool Load(DiscriminativePCNetwork &net, const std::string &dirPath);

    private:
        static std::string ActivationToString(ActivationType type);
        static ActivationType StringToActivation(const std::string &str);
    };
}