#pragma once
#include <string>
#include <deepity/networks/DiscriminativePCNetwork.h>

/**
 * @file ModelIO.h
 * @brief Save/load support for persisting a DiscriminativePCNetwork's
 * trained state to and from disk, as a structured directory rather than a
 * single opaque binary blob.
 *
 * @note Layer weights and activation configuration are stored separately:
 * numeric weights go into a flat binary buffer (weights.bin), while
 * per-layer metadata (sizes, activation types, hyperparameters) is
 * expected to live in a human-readable manifest (manifest.json),
 * with a generated README.md alongside for a human-readable summary.
 * ActivationToString()/StringToActivation() are the round-trip
 * conversion used when writing/reading that manifest.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    /// @brief Static utility class for saving and loading
    /// DiscriminativePCNetwork model state to/from a structured directory
    /// on disk.
    class ModelIO
    {
    public:
        /// @brief Saves the network into a structured directory
        /// (manifest.json, weights.bin, README.md).
        /// @param net The network whose state (weights, biases, and
        /// per-layer configuration) should be saved.
        /// @param dirPath Path to the directory to save into; created if
        /// it doesn't already exist. Any existing contents at this path
        /// may be overwritten.
        /// @return true if every file was written successfully, false
        /// otherwise.
        static bool Save(const DiscriminativePCNetwork &net, const std::string &dirPath);

        /// @brief Loads a network state from a structured directory.
        /// @param net The network to load state into. Its layer
        /// configuration is expected to already match (or be rebuilt
        /// from) what's recorded in the manifest; existing weights are
        /// overwritten with the loaded values.
        /// @param dirPath Path to a directory previously written by
        /// Save().
        /// @return true if the network was loaded successfully, false
        /// otherwise (e.g. missing/malformed files).
        static bool Load(DiscriminativePCNetwork &net, const std::string &dirPath);

    private:
        /// @brief Converts an ActivationType to its manifest string
        /// representation.
        /// @param type The activation type to convert.
        /// @return The string form of @p type, as written into
        /// manifest.json.
        static std::string ActivationToString(ActivationType type);

        /// @brief Converts a manifest string representation back into an
        /// ActivationType. Inverse of ActivationToString().
        /// @param str The string to convert, as read from manifest.json.
        /// @return The corresponding ActivationType.
        static ActivationType StringToActivation(const std::string &str);
    };
}