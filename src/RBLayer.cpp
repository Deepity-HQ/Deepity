#include "RBLayer.h"
#include <cblas.h>
#include <iostream>
#include <random>

using namespace Deep;

RBLayer::RBLayer(size_t inSize, size_t outSize, float var, float var_td, float k_1, float k_2, float lmbda, float alpha, size_t batchSize, int stepSize, ActivationFn act, ActivationFn dAct)
    : var(var), var_td(var_td), k_1(k_1), k_2(k_2), lmbda(lmbda), alpha(alpha), batchSize(batchSize), stepSize(stepSize), act(act), dAct(dAct)
{
    this->size = inSize;
    this->nextSize = outSize;

    size_t total = GetTotalSize();
    float *arena = new (std::align_val_t{64}) float[total](); // zero-init
    Attach(arena);
    this->inputBuffer = std::make_unique<float[]>(batchSize * size);
    ownsMemory = true; // Attach() sets it false, so re-set after
}

RBLayer::~RBLayer()
{
    if (ownsMemory && r != nullptr)
    {
        ::operator delete[](r, std::align_val_t{64});
        r = nullptr;
    }
}

// --- MEMORY ORCHESTRATION ---
size_t RBLayer::GetTotalSize() const noexcept
{
    size_t total = 0;
    total += batchSize * nextSize; // r (beliefs)
    total += nextSize * size;    // U (weights)
    total += batchSize * size;  // e_bu
    total += batchSize * nextSize; // e_td
    total += batchSize * nextSize; // tmpBelief
    total += batchSize * size;  // tmpWeight1
    total += batchSize * size;  // tmpWeight2
    total += batchSize * size;  // tmpWeight3
    total += nextSize * size;    // tmpWeight4
    return total;
}

void RBLayer::Attach(float *ptr) noexcept
{
    if (ownsMemory && r != nullptr)
    {
        ::operator delete[](r, std::align_val_t{64});
    }

    r = ptr;
    ptr += batchSize * nextSize;
    U = ptr;
    ptr += nextSize * size;
    e_bu = ptr;
    ptr += batchSize * size;
    e_td = ptr;
    ptr += batchSize * nextSize;
    tmpBelief = ptr;
    ptr += batchSize * nextSize;
    tmpWeight1 = ptr;
    ptr += batchSize * size;
    tmpWeight2 = ptr;
    ptr += batchSize * size;
    tmpWeight3 = ptr;
    ptr += batchSize * size;
    tmpWeight4 = ptr;

    ownsMemory = false;

    size_t uSize = nextSize * size;
    std::mt19937 rng(42);
    float scale = std::sqrt(2.0f / (size + nextSize));
    std::normal_distribution<float> dist(0.0f, scale);
    for (size_t i = 0; i < uSize; ++i)
        U[i] = dist(rng);
}

// CORE EXECUTION
void RBLayer::RunPrediction(const float *input, size_t currentBatchSize) noexcept
{
    // FAST PATH: full batch, buffer empty
    if (currentBatchSize == batchSize && pendingCount == 0)
    {
        RunInferenceStep(input, nullptr, batchSize);
        return;
    }

    size_t samplesRemaining = currentBatchSize;
    size_t xOffset = 0;

    while (samplesRemaining > 0)
    {
        size_t spaceLeft = batchSize - pendingCount;
        size_t samplesToCopy = std::min(samplesRemaining, spaceLeft);

        cblas_scopy(samplesToCopy * size,
                    input + (xOffset * size), 1,
                    inputBuffer.get() + (pendingCount * size), 1);

        pendingCount += samplesToCopy;
        samplesRemaining -= samplesToCopy;
        xOffset += samplesToCopy;

        if (pendingCount == batchSize)
        {
            RunInferenceStep(inputBuffer.get(), nullptr, batchSize);
            pendingCount = 0;
        }
    }
}

void RBLayer::RunInferenceStep(const float *input, const float *topDown, size_t currentBatchSize) noexcept
{
    CacheInput(input, currentBatchSize);

    for (int i = 0; i < stepSize; i++)
    {
        CalcError(input, topDown, currentBatchSize);
        UpdateBeliefs(input, topDown, currentBatchSize);
    }

    UpdateWeights(currentBatchSize);
}

void RBLayer::CalcError(const float *bottomUp, const float *topDown, size_t currentBatchSize) noexcept
{
    if (currentBatchSize > this->batchSize)
    {
        std::cerr << "CRITICAL: Requested batch " << currentBatchSize
                  << " exceeds max capacity " << this->batchSize << std::endl;
        std::abort();
    }
    // -- BOTTOM UP (e_bu = I - r @ U) --
    if (bottomUp != nullptr)
    {
        cblas_scopy(currentBatchSize * size, bottomUp, 1, e_bu, 1);
    }

    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        currentBatchSize, size, nextSize,
        -1.0f, r, nextSize, U, size,
        1.0f, e_bu, size);

    // -- TOP DOWN (e_td = r_TopDown - r) --
    if (topDown != nullptr)
    {
        cblas_scopy(currentBatchSize * nextSize, topDown, 1, e_td, 1);  // e_td = r_td
        cblas_saxpy(currentBatchSize * nextSize, -1.0f, r, 1, e_td, 1); // e_td -= r
    }
    else
    {
        // No top-down signal: e_td = 0 - r = -r
        cblas_scopy(currentBatchSize * nextSize, r, 1, e_td, 1);
        cblas_sscal(currentBatchSize * nextSize, -1.0f, e_td, 1);
    }
}

void RBLayer::UpdateBeliefs(const float *bottomUp, const float *topDown, size_t currentBatchSize) noexcept
{
    if (currentBatchSize > this->batchSize)
    {
        std::cerr << "CRITICAL: Requested batch " << currentBatchSize
                  << " exceeds max capacity " << this->batchSize << std::endl;
        std::abort();
    }

    // r += (k_1 / var) * I @ U^T
    if (bottomUp != nullptr)
    {
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            currentBatchSize, nextSize, size,
            k_1 / var, bottomUp, size, U, size,
            1.0f, r, nextSize);
    }

    if (topDown != nullptr)
    {
        cblas_scopy(currentBatchSize * nextSize, topDown, 1, tmpBelief, 1);
    }
    else
    {
        std::fill(tmpBelief, tmpBelief + (currentBatchSize * nextSize), 0.0f);
    }
    cblas_saxpy(currentBatchSize * nextSize, -1.0f, r, 1, tmpBelief, 1);
    cblas_saxpy(currentBatchSize * nextSize, k_1 / var_td, tmpBelief, 1, r, 1);

    // r -= k_1 * alpha * r  (g'(r) = 2*alpha*r for Gaussian prior, factor of 2 absorbed)
    cblas_saxpy(currentBatchSize * nextSize, -1.0f * k_1 * alpha, r, 1, r, 1);

    // r -= (k_1 / var) * W @ r,  where W = U^T @ U (lateral inhibition)
    // tmpBelief = r @ U  (shape: [B x size])
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        currentBatchSize, size, nextSize,
        1.0f, r, nextSize, U, size,
        0.0f, tmpBelief, size);

    // r -= (k_1 / var) * tmpBelief @ U^T  (completes the W = U^T U application)
    cblas_sgemm(
        CblasRowMajor, CblasNoTrans, CblasTrans,
        currentBatchSize, nextSize, size,
        -1.0f * k_1 / var, tmpBelief, size, U, size,
        1.0f, r, nextSize);
}

void RBLayer::UpdateWeights(size_t currentBatchSize) noexcept
{
    // tmp2 = r @ U
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                currentBatchSize, size, nextSize,
                1.0f, r, nextSize, U, size,
                0.0f, tmpWeight2, size);

    // tmp3 = tmp2
    cblas_scopy(currentBatchSize * size, tmpWeight2, 1, tmpWeight3, 1);

    // Apply Activations
    if (act)
        act(tmpWeight2, currentBatchSize * size); // tmp2 = f(Ur)
    if (dAct)
        dAct(tmpWeight3, currentBatchSize * size); // tmp3 = f'(Ur)

    // tmp1 = I - f(Ur)
    cblas_saxpy(currentBatchSize * size, -1.0f, tmpWeight2, 1, tmpWeight1, 1);

    // tmp3 = f'(Ur) * (I - f(Ur))
    for (size_t i = 0; i < currentBatchSize * size; ++i)
        tmpWeight3[i] *= tmpWeight1[i];

    // tmp4 = r^T @ tmp3
    cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                nextSize, size, currentBatchSize,
                k_2 / var, r, nextSize, tmpWeight3, size,
                0.0f, tmpWeight4, size);

    // U += tmp4
    cblas_saxpy(nextSize * size, 1.0f, tmpWeight4, 1, U, 1);

    // Normalize each row of U to unit norm
    for (size_t o = 0; o < nextSize; o++)
    {
        float norm = cblas_snrm2(size, U + o * size, 1);
        if (norm > 1e-8f)
            cblas_sscal(size, 1.0f / norm, U + o * size, 1);
    }

    // U -= k_2 * lambda * U (weight decay)
    cblas_saxpy(nextSize * size, -1.0f * k_2 * lmbda, U, 1, U, 1);
}

void RBLayer::Flush() noexcept
{
    if (pendingCount > 0)
    {
        RunInferenceStep(inputBuffer.get(), nullptr, pendingCount);
        pendingCount = 0;
    }
}

void RBLayer::CacheInput(const float *input, size_t currentBatchSize) noexcept
{
    cachedInput = input;
    cachedBatchSize = currentBatchSize;
    if (input != nullptr)
        cblas_scopy(currentBatchSize * size, input, 1, tmpWeight1, 1);
}

float RBLayer::CalculateState() noexcept
{
    CalcError(cachedInput, nullptr, cachedBatchSize);
    return 0.5f * cblas_sdot(cachedBatchSize * size, e_bu, 1, e_bu, 1);
}

void RBLayer::UpdateState() noexcept
{
    UpdateBeliefs(cachedInput, nullptr, cachedBatchSize);
}

void RBLayer::UpdateWeights() noexcept
{
    UpdateWeights(cachedBatchSize);
}

// --- GETTERS ---
size_t RBLayer::GetBatchSize() const noexcept { return batchSize; }
size_t RBLayer::GetInputSize() const noexcept { return size; }
size_t RBLayer::GetOutputSize() const noexcept { return nextSize; }
float *RBLayer::GetBeliefs() noexcept { return r; }
const float *RBLayer::GetInferenceError() const noexcept { return e_bu; }

#ifdef _DEBUG
void RBLayer::DebugStats(int layerIndex) const
{
    std::cout << "  [Layer " << layerIndex << " - RBLayer]\n"
              << "    In: " << size << " | Out: " << nextSize << '\n'
              << "    Batch Size: " << batchSize << '\n'
              << "    k_1: " << k_1 << " | k_2: " << k_2 << " | var: " << var << '\n';
}
#endif