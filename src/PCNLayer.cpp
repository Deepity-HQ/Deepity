#include "PCNLayer.h"
#include <cblas.h>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <openrand/tyche.h>
#include <omp.h>

#include "Optimize.h"

using namespace Deep;
using RNG = openrand::Tyche;
uint64_t base_seed = 12345; // <- Consider changing

/**
 * Implementation Notes:
 * The sizes are of the following and in the following order:
 * - W is outSize * inSize
 * - z is outSize
 * - p is inSize
 * - err is inSize
 */

PCLayer::PCLayer(size_t inSize, size_t outSize, float lr, float ir, int stepSize, ActivationFn act)
    : inputSize(inSize), outputSize(outSize), lr(lr), ir(ir), stepSize(stepSize), activation(act)
{
    B = GetBatchSize();
#ifdef _DEBUG
    std::cout << "[Deepity] Initialized layer with batch size " << B << std::endl;
#endif
    auto align_elements = [](size_t count)
    {
        return (count + 15) & ~size_t(15);
    };

    size_t wSizeAligned = align_elements(outSize * inSize);
    size_t zSizeAligned = align_elements(B * outputSize);
    size_t pSizeAligned = align_elements(B * inputSize);
    size_t errSizeAligned = align_elements(B * inputSize);
    // Compute pointers using the aligned step sizes
    zBegin = wSizeAligned;
    pBegin = zBegin + zSizeAligned;
    errBegin = pBegin + pSizeAligned;
    totalSize = errBegin + errSizeAligned;

    this->arr = new (std::align_val_t{64}) float[totalSize];
    this->inputBuffer = std::make_unique<float[]>(B * inputSize);

#pragma omp parallel
    {
        uint64_t thread_seed = base_seed + (uint64_t)omp_get_thread_num();
        RNG rng(thread_seed, 0);

#pragma omp for
        for (size_t i = 0; i < pBegin; i++)
        {
            arr[i] = rng.rand<float>() * 0.02f - 0.01f;
        }
    }
}

PCLayer &PCLayer::operator=(PCLayer &&other) noexcept
{
    if (this == &other)
        return *this;

    // Free what we currently own before overwriting
    if (ownsArr && arr != nullptr)
    {
        ::operator delete[](arr, std::align_val_t{64});
    }

    inputSize = other.inputSize;
    outputSize = other.outputSize;
    B = other.B;
    lr = other.lr;
    ir = other.ir;
    stepSize = other.stepSize;
    activation = other.activation;
    arr = other.arr;
    ownsArr = other.ownsArr;
    zBegin = other.zBegin;
    pBegin = other.pBegin;
    errBegin = other.errBegin;
    totalSize = other.totalSize;
    inputBuffer = std::move(other.inputBuffer);
    pendingCount = other.pendingCount;

    other.arr = nullptr;
    other.ownsArr = false;

    return *this;
}

PCLayer::~PCLayer()
{
    if (ownsArr && arr != nullptr)
    {
        ::operator delete[](arr, std::align_val_t{64});
        arr = nullptr;
    }
}

PCLayer::PCLayer(PCLayer &&other) noexcept
    : inputSize(other.inputSize), outputSize(other.outputSize),
      B(other.B), lr(other.lr), ir(other.ir), stepSize(other.stepSize),
      activation(other.activation), arr(other.arr), ownsArr(other.ownsArr),
      zBegin(other.zBegin), pBegin(other.pBegin),
      errBegin(other.errBegin), totalSize(other.totalSize),
      inputBuffer(std::move(other.inputBuffer)),
      pendingCount(other.pendingCount)
{
    other.arr = nullptr;
    other.ownsArr = false;
}

void PCLayer::CalcPrediction(size_t batchSize) noexcept
{
    cblas_sgemm( // p = z * W^T
        CblasRowMajor, CblasNoTrans, CblasTrans,
        batchSize,  // M = batch size
        inputSize,  // N = output cols
        outputSize, // K = inner dim
        1.0f,
        arr + zBegin, outputSize, // Z is [B × outSize]
        arr, outputSize,          // W is [inSize × outSize], transposed
        0.0f,
        arr + pBegin, inputSize);
}

void PCLayer::CalcStepError(const float *x, size_t batchSize) noexcept
{
    assert(x != nullptr && "Cannot calculate error if x is null!");
    cblas_scopy(batchSize * inputSize, x, 1, arr + errBegin, 1); // E = X
    cblas_saxpy(batchSize * inputSize, -1.0f,                    // E -= p
                arr + pBegin, 1,
                arr + errBegin, 1);
}

void PCLayer::UpdateBeliefs(size_t batchSize) noexcept
{
    cblas_sgemm( // z += ir * err * W
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        batchSize,  // M = batch size
        outputSize, // N
        inputSize,  // K
        ir,
        arr + errBegin, inputSize, // E is [B × inSize]
        arr, outputSize,           // W is [inSize × outSize]
        1.0f,
        arr + zBegin, outputSize);

    activation(arr + zBegin, batchSize * outputSize); // z = f(z)
}

void PCLayer::RunBatchedPrediction(const float *x, size_t batchSize) noexcept
{
    for (int i = 0; i < stepSize; i++)
    {
        CalcPrediction(batchSize);
        CalcStepError(x, batchSize);
        UpdateBeliefs(batchSize);
    }
    UpdateWeights(batchSize);
}

void PCLayer::RunPrediction(const float *x) noexcept
{
    size_t slot = pendingCount * inputSize;
    cblas_scopy(inputSize, x, 1, inputBuffer.get() + slot, 1);
    pendingCount++;

    if (pendingCount == B)
    {
        RunBatchedPrediction(inputBuffer.get(), B);
        pendingCount = 0;
    }
}

void PCLayer::Flush() noexcept
{
    if (pendingCount > 0)
    {
        RunBatchedPrediction(inputBuffer.get(), pendingCount); // partial batch
        pendingCount = 0;
    }
}

void PCLayer::Attach(float *ptr) noexcept
{
    assert(ptr);

    if (ownsArr && arr != nullptr)
    {
        ::operator delete[](arr, std::align_val_t{64});
    }

    this->arr = ptr;
    ownsArr = false;

#pragma omp parallel
    {
        uint64_t thread_seed = base_seed + (uint64_t)omp_get_thread_num();
        RNG rng(thread_seed, 0);

#pragma omp for
        for (size_t i = 0; i < pBegin; i++)
        {
            arr[i] = rng.rand<float>() * 0.02f - 0.01f;
        }
    }

#ifdef _DEBUG
    std::cout << "[Deepity] Layer attached. Alignment check: "
              << (reinterpret_cast<uintptr_t>(arr) % 64 == 0 ? "PASSED" : "FAILED")
              << " | Total elements: " << totalSize << '\n';
#endif
}

void PCLayer::UpdateWeights(size_t batchSize) noexcept
{
    cblas_sgemm( // W += lr * e^T * z
        CblasRowMajor, CblasTrans, CblasNoTrans,
        inputSize,  // M
        outputSize, // N
        batchSize,  // K = batch size
        lr,
        arr + errBegin, inputSize,
        arr + zBegin, outputSize,
        1.0f,
        arr, outputSize);
}

#ifdef _DEBUG

#include <iomanip>

void PCLayer::DebugStats(int layerIndex) const
{
    auto print_region = [&](const char *name, size_t begin, size_t end)
    {
        size_t size = end - begin;
        if (size == 0)
            return;

        size_t nan_count = 0, inf_count = 0, zero_count = 0;
        float min_val = arr[begin];
        float max_val = arr[begin];
        double sum = 0.0, sq_sum = 0.0;

        for (size_t i = begin; i < end; ++i)
        {
            const float v = arr[i];

            if (std::isnan(v))
                nan_count++;
            else if (std::isinf(v))
                inf_count++;
            else
            {
                if (v == 0.0f)
                    zero_count++;
                min_val = std::min(min_val, v);
                max_val = std::max(max_val, v);
                sum += v;
                sq_sum += static_cast<double>(v) * v;
            }
        }

        double mean = sum / size;
        double variance = (sq_sum / size) - (mean * mean);
        double stddev = std::sqrt(std::max(0.0, variance));
        float sparsity = (static_cast<float>(zero_count) / size) * 100.0f;

        // Formatted Console Output
        std::cout << std::left << std::setw(5) << name
                  << " | N=" << std::setw(8) << size
                  << " | Min: " << std::setw(9) << min_val
                  << " | Max: " << std::setw(9) << max_val
                  << " | Mean: " << std::setw(9) << mean
                  << " | Std: " << std::setw(9) << stddev
                  << " | Zero: " << std::setw(5) << std::fixed << std::setprecision(1) << sparsity << "%";

        if (nan_count > 0 || inf_count > 0)
        {
            std::cout << "  [ALARM: " << nan_count << " NaNs, " << inf_count << " Infs!]";
        }
        std::cout << '\n';
        std::cout.unsetf(std::ios_base::floatfield); // reset precision
    };

    std::cout << "--------------------------------------------------------------------------------------\n";
    std::cout << " LAYER " << layerIndex << " DIAGNOSTICS (In: " << inputSize << ", Out: " << outputSize << ")\n";
    std::cout << "--------------------------------------------------------------------------------------\n";

    print_region("W", 0, zBegin);
    print_region("Z", zBegin, pBegin);
    print_region("P", pBegin, errBegin);
    print_region("Err", errBegin, totalSize);
}
#endif