#include <deepity/StreamAlignedBatcher.h>
#include <algorithm>

namespace Deep
{
    StreamAlignedBatcher::StreamAlignedBatcher(const float *X, const float *Y, const int *labels,
                                               size_t numSamples, size_t xStride, size_t yStride,
                                               int numClasses, int perClass, unsigned seed)
        : X(X), Y(Y), labels(labels), numSamples(numSamples), xStride(xStride), yStride(yStride),
          numClasses(numClasses), perClass(perClass), rng(seed)
    {
        classIndices.resize(numClasses);
        pointers.assign(numClasses, 0);

        for (size_t i = 0; i < numSamples; ++i)
        {
            int c = labels[i];
            if (c < 0 || c >= numClasses)
                throw std::out_of_range("StreamAlignedBatcher: label out of [0, numClasses) range");
            classIndices[c].push_back(i);
        }

        for (int c = 0; c < numClasses; ++c)
        {
            if ((int)classIndices[c].size() < perClass)
                throw std::invalid_argument("StreamAlignedBatcher: a class has fewer than perClass samples");
            ReshuffleClass(c);
        }
    }

    void StreamAlignedBatcher::ReshuffleClass(int c) noexcept
    {
        std::shuffle(classIndices[c].begin(), classIndices[c].end(), rng);
        pointers[c] = 0;
    }

    std::vector<size_t> StreamAlignedBatcher::NextClassSlice(int c, int n) noexcept
    {
        auto &idx = classIndices[c];
        size_t p = pointers[c];

        if (p + (size_t)n > idx.size())
        {
            ReshuffleClass(c);
            p = 0;
        }

        std::vector<size_t> slice(idx.begin() + p, idx.begin() + p + n);
        pointers[c] = p + n;
        return slice;
    }

    size_t StreamAlignedBatcher::NumBatchesPerEpoch() const noexcept
    {
        size_t minCount = classIndices[0].size();
        for (int c = 1; c < numClasses; ++c)
            minCount = std::min(minCount, classIndices[c].size());
        return minCount / (size_t)perClass;
    }

    void StreamAlignedBatcher::GetBatch(float *X_out, float *Y_out, int *labels_out) noexcept
    {
        int row = 0;
        for (int c = 0; c < numClasses; ++c)
        {
            std::vector<size_t> sliceIdx = NextClassSlice(c, perClass);

            for (size_t globalIdx : sliceIdx)
            {
                std::memcpy(X_out + (size_t)row * xStride, X + globalIdx * xStride, xStride * sizeof(float));
                std::memcpy(Y_out + (size_t)row * yStride, Y + globalIdx * yStride, yStride * sizeof(float));
                labels_out[row] = labels[globalIdx];
                ++row;
            }
        }
    }
}
