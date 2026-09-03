#pragma once
#include <algorithm>
#include <cstddef>
#include <memory>
#include <vector>
#include <immintrin.h>
#include <omp.h>

#ifdef __linux__
#include <fstream>
#elif defined(_WIN32)
#include <windows.h>
#endif

#ifdef DEEPITY_USE_MKL
#include <mkl_service.h>
#else
extern "C" {
    void openblas_set_num_threads(int num_threads);
}
#endif

/**
 * @file Optimize.h
 * @brief Defines useful optimization functions of a PC model.
 *
 * This header currently only includes implementation of an L2 Cache size lookup.
 *
 * Usage:
 *  #include <deepity/utils/Optimize.h>
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

        // Computed once, by name, instead of repeating `bufSize /
        // sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION)` inline at both the
        // allocation site and the loop condition -- and std::vector instead
        // of make_unique<T[]>(count), since that's the far more common,
        // heavily-exercised pattern for a runtime-sized buffer like this.
        const size_t count = static_cast<size_t>(bufSize) / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buf(count);
        GetLogicalProcessorInformation(buf.data(), &bufSize);

        for (size_t i = 0; i < count; ++i)
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

    static inline void DynamicThread(int batchSize) noexcept
    {
        static int currentThreads = -1;

        const int THRESHOLD = 32;
        int maxProcs = omp_get_num_procs();

        // Confirmed via extended sweep (batch=1024/2048/4096, threads up to
        // 48): the optimum drifts mildly upward with batch size (8 -> ~12 ->
        // ~16) but the difference is small and noisy at this scale (~7%
        // between 8/12/16 at batch=4096) -- NOT worth a hard per-batch-size
        // table. What IS solid and unambiguous: 48 threads is always the
        // worst choice tested, by 1.8-2.9x, at every batch size -- likely
        // cross-CCD/Infinity Fabric synchronization cost on this many-core
        // part. A smooth, capped scaling curve fits the real trend better
        // than another single hardcoded number.
        int scaled = 8 + (batchSize / 1024) * 4;             // 8 @ <1024, ~12 @ 2048, ~16 @ 4096, etc.
        const int MAX_USEFUL_THREADS = std::min(scaled, 16); // cap -- 48 was confirmed worse
                                                             // every time, no reason to extrapolate past 16

        int targetThreads = (batchSize < THRESHOLD) ? 1 : std::min(maxProcs, MAX_USEFUL_THREADS);

        if (currentThreads != targetThreads)
        {
            omp_set_num_threads(targetThreads);
#ifdef DEEPITY_USE_MKL
            mkl_set_num_threads(targetThreads);
#else
            openblas_set_num_threads(targetThreads);
#endif
            currentThreads = targetThreads;
        }
    }

    static inline float hsum256_ps(__m256 x)
    {
        __m128 lo = _mm256_castps256_ps128(x);
        __m128 hi = _mm256_extractf128_ps(x, 1);
        lo = _mm_add_ps(lo, hi);

        __m128 shuf = _mm_movehdup_ps(lo);
        __m128 sums = _mm_add_ps(lo, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);

        return _mm_cvtss_f32(sums);
    }

    static inline float hsum128_ps(__m128 x)
    {
        __m128 shuf = _mm_movehdup_ps(x);
        __m128 sums = _mm_add_ps(x, shuf);
        shuf = _mm_movehl_ps(shuf, sums);
        sums = _mm_add_ss(sums, shuf);
        return _mm_cvtss_f32(sums);
    }
}
