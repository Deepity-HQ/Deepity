#pragma once
#include <cstddef>
#include <memory>
#ifdef __linux__
#include <fstream>
#elif defined(_WIN32)
#include <windows.h>
#endif

/**
 * @file Optimize.h
 * @brief Defines useful optimization functions of a PC model.
 *
 * This header currently only includes implementation of an L2 Cache size lookup.
 *
 * Usage:
 *  #include <Optimize.h>
 *
 * Example:
 *  size_t sizeofL2 = GetL2CacheBytes();
 *
 * @note Separate versions exist for Windows VS. Linux.
 * @version 1.0
 * @date 2026-06-21
 * @author Jack Rose
 */
namespace Deep
{
    inline size_t GetL2CacheBytes()
    {
#ifdef __linux__
        std::ifstream f("/sys/devices/system/cpu/cpu0/cache/index2/size");
        size_t kb = 0;
        char unit;
        f >> kb >> unit;
        return kb * 1024;
#elif defined(_WIN32)
        DWORD bufSize = 0;
        GetLogicalProcessorInformation(nullptr, &bufSize);
        auto buf = std::make_unique<SYSTEM_LOGICAL_PROCESSOR_INFORMATION[]>(bufSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        GetLogicalProcessorInformation(buf.get(), &bufSize);
        for (DWORD i = 0; i < bufSize / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION); i++)
        {
            if (buf[i].Relationship == RelationCache && buf[i].Cache.Level == 2)
                return buf[i].Cache.Size;
        }
        return 512 * 1024; // fallback
#else
        return 512 * 1024; // fallback for unknown platforms
#endif
    }

    inline size_t AutoBatchSize(size_t inSize, size_t outSize)
    {
        size_t l2 = GetL2CacheBytes();

        size_t wSize = inSize * outSize * sizeof(float);
        size_t bytesPerBatch = (outSize + inSize + inSize) * sizeof(float);

        size_t remaining = (l2 > wSize) ? l2 - wSize : l2 / 2;
        size_t B = remaining / bytesPerBatch;

        // Round down to nearest power of 2
        size_t pow2 = 1;
        while (pow2 * 2 <= B)
            pow2 *= 2;

        if (pow2 < 64)
            pow2 = 64;

        if (pow2 > 512)
            pow2 = 512;

        return pow2;
    }
}