#pragma once
#include <vector>
#include <random>
#include <cstring>
#include <stdexcept>

/**
 * @file StreamAlignedBatcher.h
 * @brief C++ port of the Python StreamAlignedBatcher -- builds batches with
 * a fixed number of examples per class, grouped together, required for
 * I_avg's per-class averaging (Eq. 7 of "Faster Predictive Coding Networks
 * via Better Initialization").
 *
 * Kept deliberately SEPARATE from DiscriminativePCNetwork/SimplePCNetwork/
 * ConvPCNetwork -- this is a standalone data-pipeline utility, not part of
 * the core numerical library. Any consumer (C++, or via future bindings)
 * uses it independently, feeding its output into whichever network class
 * they're using via the existing ClampInput/ClampState-style API.
 *
 * Does NOT own or copy the dataset -- holds pointers + stride info only.
 * Caller must keep X/Y/labels alive for the batcher's lifetime. GetBatch()
 * gathers selected rows into caller-provided output buffers, so the same
 * buffers can be reused across calls without repeated allocation.
 */

namespace Deep
{
    class StreamAlignedBatcher
    {
    public:
        /// @param X Flat, row-major input data, numSamples rows of xStride floats each. NOT owned.
        /// @param Y Flat, row-major target data, numSamples rows of yStride floats each. NOT owned.
        /// @param labels Integer class label per sample, length numSamples. NOT owned.
        /// @param numSamples Total number of samples in X/Y/labels.
        /// @param xStride Floats per X row.
        /// @param yStride Floats per Y row.
        /// @param numClasses Number of distinct classes (labels expected in [0, numClasses)).
        /// @param perClass Examples per class per batch (batch size = numClasses * perClass).
        StreamAlignedBatcher(const float *X, const float *Y, const int *labels,
                              size_t numSamples, size_t xStride, size_t yStride,
                              int numClasses, int perClass, unsigned seed = 42);

        /// @brief Number of complete batches obtainable per epoch, limited
        /// by the rarest class's sample count (matches the Python version's
        /// behavior -- some samples from more common classes go unused
        /// each epoch, rather than padding/repeating to force even usage).
        size_t NumBatchesPerEpoch() const noexcept;

        /// @brief Fills the next batch into caller-provided, pre-allocated
        /// output buffers. Rows are GROUPED by class: rows
        /// [0, perClass) are class 0, [perClass, 2*perClass) are class 1,
        /// etc. -- same convention as the Python version, so per-class
        /// slicing for cache-averaging is a trivial contiguous range.
        /// @param X_out Must be sized (numClasses*perClass) * xStride.
        /// @param Y_out Must be sized (numClasses*perClass) * yStride.
        /// @param labels_out Must be sized (numClasses*perClass).
        void GetBatch(float *X_out, float *Y_out, int *labels_out) noexcept;

        int GetBatchSize() const noexcept { return numClasses * perClass; }
        int GetPerClass() const noexcept { return perClass; }
        int GetNumClasses() const noexcept { return numClasses; }
        size_t GetXStride() const noexcept { return xStride; }
        size_t GetYStride() const noexcept { return yStride; }

    private:
        const float *X;
        const float *Y;
        const int *labels;
        size_t numSamples;
        size_t xStride;
        size_t yStride;
        int numClasses;
        int perClass;
        std::mt19937 rng;

        std::vector<std::vector<size_t>> classIndices; // per-class list of global row indices
        std::vector<size_t> pointers;                  // current read position within each class's list

        void ReshuffleClass(int c) noexcept;

        /// @brief Returns `n` global row indices for class `c`, advancing
        /// (and reshuffling/wrapping, if exhausted) that class's pointer.
        std::vector<size_t> NextClassSlice(int c, int n) noexcept;
    };
}
