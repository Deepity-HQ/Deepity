#include "ModelIO.h"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <iostream>

namespace fs = std::filesystem;

namespace Deep
{
    std::string ModelIO::ActivationToString(ActivationType type)
    {
        switch (type)
        {
            case ActivationType::TANH: return "tanh";
            case ActivationType::SIGMOID: return "sigmoid";
            case ActivationType::RELU: return "relu";
            case ActivationType::LINEAR: return "linear";
            default: return "relu";
        }
    }

    bool ModelIO::Save(const DiscriminativePCNetwork &net, const std::string &dirPath)
    {
        try
        {
            // 1. Ensure directory exists
            fs::create_directories(dirPath);

            fs::path manifestPath = fs::path(dirPath) / "manifest.json";
            fs::path weightsPath  = fs::path(dirPath) / "weights.bin";
            fs::path readmePath   = fs::path(dirPath) / "README.md";

            const auto &layers = net.GetLayers();
            uint64_t totalParameters = 0;

            // -------------------------------------------------------------
            // A. WRITE BINARY WEIGHTS (High-Speed Sequential Stream)
            // -------------------------------------------------------------
            std::ofstream wStream(weightsPath, std::ios::binary);
            if (!wStream) return false;

            // Large 1MB buffer to prevent OS context switching during multi-GB offloading
            constexpr size_t BUFFER_SIZE = 1024 * 1024;
            std::vector<char> writeBuffer(BUFFER_SIZE);
            wStream.rdbuf()->pubsetbuf(writeBuffer.data(), BUFFER_SIZE);

            for (const auto *layer : layers)
            {
                size_t inputSize  = layer->GetInputSize();
                size_t outputSize = layer->GetOutputSize();

                // Dump Precisions (size floats)
                wStream.write(reinterpret_cast<const char *>(layer->GetPrecisions()), sizeof(float) * inputSize);

                if (outputSize > 0)
                {
                    size_t wCount = inputSize * outputSize;
                    size_t bCount = outputSize;

                    // Dump Weights and Biases
                    wStream.write(reinterpret_cast<const char *>(layer->GetWeights()), sizeof(float) * wCount);
                    wStream.write(reinterpret_cast<const char *>(layer->GetBiases()), sizeof(float) * bCount);

                    totalParameters += wCount + bCount;
                }
            }
            wStream.close();

            // -------------------------------------------------------------
            // B. WRITE MANIFEST (JSON Metadata)
            // -------------------------------------------------------------
            std::ofstream mStream(manifestPath);
            if (!mStream) return false;

            mStream << "{\n";
            mStream << "  \"format\": \"DeepityCheckpoint\",\n";
            mStream << "  \"version\": 1,\n";
            mStream << "  \"batch_size\": " << net.GetBatchSize() << ",\n";
            mStream << "  \"layer_count\": " << layers.size() << ",\n";
            mStream << "  \"total_parameters\": " << totalParameters << ",\n";
            mStream << "  \"layers\": [\n";

            for (size_t i = 0; i < layers.size(); ++i)
            {
                const auto *layer = layers[i];
                mStream << "    {\n";
                mStream << "      \"index\": " << i << ",\n";
                mStream << "      \"input_size\": " << layer->GetInputSize() << ",\n";
                mStream << "      \"output_size\": " << layer->GetOutputSize() << ",\n";
                mStream << "      \"learning_rate\": " << layer->GetLearningRate() << ",\n";
                mStream << "      \"inference_rate\": " << layer->GetInferenceRate() << ",\n";
                mStream << "      \"precision_rate\": " << layer->GetPrecisionRate() << ",\n";
                mStream << "      \"lambda\": " << layer->GetLambda() << ",\n";
                mStream << "      \"activation\": \"" << ActivationToString(layer->GetActivationType()) << "\"\n";
                mStream << "    }" << (i + 1 < layers.size() ? "," : "") << "\n";
            }
            mStream << "  ]\n";
            mStream << "}\n";
            mStream.close();

            // -------------------------------------------------------------
            // C. WRITE HUMAN-READABLE README.md
            // -------------------------------------------------------------
            std::ofstream rStream(readmePath);
            if (!rStream) return false;

            double footprintMB = static_cast<double>(totalParameters * sizeof(float)) / (1024.0 * 1024.0);

            rStream << "# Deepity Model Checkpoint\n\n";
            rStream << "Auto-generated model summary.\n\n";
            rStream << "## Overview\n";
            rStream << "- **Total Parameters:** " << totalParameters << "\n";
            rStream << "- **Weight Footprint:** " << std::fixed << std::setprecision(2) << footprintMB << " MB\n";
            rStream << "- **Batch Size:** " << net.GetBatchSize() << "\n";
            rStream << "- **Layer Hierarchy:** " << layers.size() << " layers\n\n";
            rStream << "## Layer Architecture\n\n";
            rStream << "| Layer | Input Dim | Output Dim | Activation | Learning Rate | Inference Rate |\n";
            rStream << "|---|---|---|---|---|---|\n";

            for (size_t i = 0; i < layers.size(); ++i)
            {
                const auto *layer = layers[i];
                rStream << "| " << i << " | "
                        << layer->GetInputSize() << " | "
                        << layer->GetOutputSize() << " | "
                        << ActivationToString(layer->GetActivationType()) << " | "
                        << layer->GetLearningRate() << " | "
                        << layer->GetInferenceRate() << " |\n";
            }
            rStream.close();

            return true;
        }
        catch (const std::exception &e)
        {
            std::cerr << "ModelIO Save Error: " << e.what() << "\n";
            return false;
        }
    }

    bool ModelIO::Load(DiscriminativePCNetwork &net, const std::string &dirPath)
    {
        fs::path weightsPath = fs::path(dirPath) / "weights.bin";

        std::ifstream wStream(weightsPath, std::ios::binary);
        if (!wStream) return false;

        constexpr size_t BUFFER_SIZE = 1024 * 1024;
        std::vector<char> readBuffer(BUFFER_SIZE);
        wStream.rdbuf()->pubsetbuf(readBuffer.data(), BUFFER_SIZE);

        // Read binary weight buffers directly into pre-allocated layer memory
        for (auto *layer : net.GetLayers())
        {
            size_t inputSize  = layer->GetInputSize();
            size_t outputSize = layer->GetOutputSize();

            // Read Precisions
            wStream.read(reinterpret_cast<char *>(const_cast<float *>(layer->GetPrecisions())), sizeof(float) * inputSize);

            if (outputSize > 0)
            {
                size_t wCount = inputSize * outputSize;
                size_t bCount = outputSize;

                // Stream directly into contiguous layer pointers
                wStream.read(reinterpret_cast<char *>(layer->GetWeights()), sizeof(float) * wCount);
                wStream.read(reinterpret_cast<char *>(layer->GetBiases()), sizeof(float) * bCount);
            }
        }

        return wStream.good() || wStream.eof();
    }
}