#include "PCNNetwork.h"
#include <cassert>
#include <cblas.h>
#include <iostream>
#include <new>
#include <cstdio>

using namespace Deep;

PCNetwork::~PCNetwork()
{
    if (masterArena != nullptr)
    {
        ::operator delete[](masterArena, std::align_val_t{64});
        masterArena = nullptr;
    }
}

void PCNetwork::Compile() noexcept
{
    arenaSize = 0;
    for (const auto &layer : layers)
    {
        arenaSize += layer.GetTotalSize();
    }

    masterArena = new (std::align_val_t{64}) float[arenaSize];
    std::fill(masterArena, masterArena + arenaSize, 0.0f);

    size_t currentOffset = 0;
    for (auto &layer : layers)
    {
        layer.Attach(masterArena + currentOffset);
        currentOffset += layer.GetTotalSize();
    }

#ifdef _DEBUG
    std::cout << "[Deepity] Network Compiled. Total Memory Arena: "
              << (arenaSize * sizeof(float)) / 1024.0 / 1024.0 << " MB\n";
#endif
}

void PCNetwork::InferenceStep(float *x)
{
    if (layers.empty() || x == nullptr)
        return;

    layers.front().RunPrediction(x);

    // Track the batch to cascade data upwards
    pendingInferenceCount++;
    size_t B = layers.front().GetBatchSize();

    if (pendingInferenceCount == B)
    {
        // Cascade the beliefs (Z) upwards layer by layer.
        for (size_t l = 0; l < layers.size() - 1; ++l)
        {
            float *z_out = layers[l].GetBeliefs();

            layers[l + 1].RunBatchedPrediction(z_out, B);
        }
        pendingInferenceCount = 0; // Reset for the next batch
    }
}

void PCNetwork::GenerationStep(float *target)
{
    if (layers.empty())
        return;

    if (target != nullptr)
        layers.back().RunPrediction(target);

    pendingGenerationCount++;
    size_t B = layers.back().GetBatchSize();

    if (pendingGenerationCount == B)
    {
        for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l)
        {
            float *z_out = layers[l + 1].GetBeliefs();
            size_t out_dim = layers[l + 1].GetOutputSize();

            for (size_t b = 0; b < B; ++b)
                layers[l].RunPrediction(z_out + (b * out_dim));
        }
        pendingGenerationCount = 0;
    }
}

void PCNetwork::FlushInference()
{
    if (layers.empty() || pendingInferenceCount == 0)
        return;

    layers.front().Flush();

    // Cascade the partial batch upwards
    for (size_t l = 0; l < layers.size() - 1; ++l)
    {
        float *z_out = layers[l].GetBeliefs();
        size_t out_dim = layers[l].GetOutputSize();

        for (size_t b = 0; b < pendingInferenceCount; ++b)
        {
            layers[l + 1].RunPrediction(z_out + (b * out_dim));
        }
        // Flush the next layer immediately after feeding it the partial batch
        layers[l + 1].Flush();
    }

    pendingInferenceCount = 0;
}

void PCNetwork::FlushGeneration()
{
    if (layers.empty() || pendingGenerationCount == 0)
        return;

    layers.back().Flush();

    // Cascade the partial batch downwards
    for (int l = static_cast<int>(layers.size()) - 2; l >= 0; --l)
    {
        float *z_out = layers[l + 1].GetBeliefs();
        size_t out_dim = layers[l + 1].GetOutputSize();

        for (size_t b = 0; b < pendingGenerationCount; ++b)
        {
            layers[l].RunPrediction(z_out + (b * out_dim));
        }
        // Flush the next layer immediately after feeding it the partial batch
        layers[l].Flush();
    }

    pendingGenerationCount = 0;
}

float PCNetwork::GetTotalEnergy() const noexcept
{
#ifdef _DEBUG
    if (pendingInferenceCount > 0 || pendingGenerationCount > 0)
        std::cout << "[Deepity] WARNING: GetTotalEnergy() called with unflushed batch ("
                  << pendingInferenceCount << " inference, "
                  << pendingGenerationCount << " generation pending). "
                  << "Energy may be stale or zero.\n";
#endif

    float totalEnergy = 0.0f;

    for (const auto &layer : layers)
    {
        const float *err = layer.GetInferenceError();

        int total_elements = static_cast<int>(layer.GetBatchSize() * layer.GetInputSize());
        totalEnergy += cblas_sdot(total_elements, err, 1, err, 1); // err^2
    }

    return 0.5f * totalEnergy;
}

void PCNetwork::SaveModel(const char *path) const noexcept
{
    assert(path && "Cannot save uncompiled network!");

    FILE *fp = fopen(path, "wb");
    assert(fp && "Failed to open file for saving model.");

    fwrite(masterArena, sizeof(float), arenaSize, fp);

    fclose(fp);
}

void PCNetwork::LoadModel(const char *path) noexcept
{
    FILE *fp = fopen(path, "rb");
    assert(fp && "Failed to open file for loading model.");

    fread(masterArena, sizeof(float), arenaSize, fp);
    fclose(fp);
}

#ifdef _DEBUG
void PCNetwork::DebugStats()
{
    std::cout << "\n======================================================================================\n";
    std::cout << "                           DEEPITY NETWORK HEALTH REPORT                              \n";
    std::cout << "======================================================================================\n";

    if (layers.empty() || masterArena == nullptr)
    {
        std::cout << "[ERROR] Network is uncompiled or empty!\n";
        return;
    }

    std::cout << "Total Layers   : " << layers.size() << '\n';
    std::cout << "Memory Arena   : " << (arenaSize * sizeof(float)) / 1024.0 / 1024.0 << " MB (64-byte Aligned)\n";
    std::cout << "Global Energy  : " << GetTotalEnergy() << '\n';

    for (size_t i = 0; i < layers.size(); ++i)
    {
        layers[i].DebugStats(static_cast<int>(i));
    }
    std::cout << "======================================================================================\n\n";
}

#endif