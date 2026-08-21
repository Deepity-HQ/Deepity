#include <deepity/layers/ConvPCLayer.h>
#include <cblas.h>
#include <omp.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace Deep
{
    ConvPCLayer::ConvPCLayer(int inChannels, int outChannels,
                             int inHeight, int inWidth,
                             int kernelH, int kernelW,
                             int strideH, int strideW,
                             int padH, int padW,
                             int batchSize,
                             float learningRate, float inferenceRate,
                             float precisionRate, float lmbda,
                             ActivationType aType, ActivationType dType)
        : inChannels(inChannels), outChannels(outChannels),
          inHeight(inHeight), inWidth(inWidth),
          kernelH(kernelH), kernelW(kernelW),
          strideH(strideH), strideW(strideW),
          padH(padH), padW(padW),
          batchSize(batchSize),
          lr(learningRate), ir(inferenceRate), pr(precisionRate), lmbda(lmbda),
          layerAbove(nullptr), layerBelow(nullptr), activationType(aType)
    {
        outHeight = (outChannels > 0) ? ConvOutDim(inHeight, kernelH, strideH, padH) : 0;
        outWidth = (outChannels > 0) ? ConvOutDim(inWidth, kernelW, strideW, padW) : 0;

        this->activation = To_Fn(aType);
        this->activationDerivative = To_dFn(dType);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    size_t ConvPCLayer::GetRequiredFloats() const noexcept
    {
        // Same 16-float rounding MemoryArena::AllocateFloats() applies per
        // call -- see the analogous fix in DiscriminativePCLayer.cpp.
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        total += pad16(ownStateSize) * 3; // z, e, dz_dt
        total += pad16(ownSize) * 2;      // p, log_p

        if (outChannels > 0)
        {
            size_t outSize = (size_t)outChannels * outHeight * outWidth;
            size_t outStateSize = (size_t)batchSize * outSize;
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;
            size_t colSize = colRows * colCols;

            total += pad16((size_t)outChannels * colRows);   // W
            total += pad16((size_t)outChannels);             // b
            total += pad16(outStateSize);                    // mu
            total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
            total += pad16(outStateSize);                    // bottom_up_cols
            total += pad16((size_t)batchSize * colSize);     // colsRepacked (same total size as colBuffer)
            total += pad16(outStateSize);                    // lgRepacked (same total size as bottom_up_cols)
            total += pad16(outStateSize);                    // muRepacked (same total size as mu)
        }

        return total;
    }

    void ConvPCLayer::BindMemory(MemoryArena &arena)
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        z = arena.AllocateFloats(ownStateSize);
        e = arena.AllocateFloats(ownStateSize);
        dz_dt = arena.AllocateFloats(ownStateSize);
        p = arena.AllocateFloats(ownSize);
        log_p = arena.AllocateFloats(ownSize);

        std::memset(z, 0, ownStateSize * sizeof(float));
        std::memset(e, 0, ownStateSize * sizeof(float));
        std::memset(dz_dt, 0, ownStateSize * sizeof(float));
        std::fill_n(p, ownSize, 1.0f);
        std::fill_n(log_p, ownSize, 0.0f);

        if (outChannels > 0)
        {
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;
            size_t outStateSize = (size_t)batchSize * outChannels * colCols;
            size_t colSize = (size_t)batchSize * colRows * colCols;

            W = arena.AllocateFloats((size_t)outChannels * colRows);
            b = arena.AllocateFloats(outChannels);
            mu = arena.AllocateFloats(outStateSize);
            colBuffer = arena.AllocateFloats(colSize);
            feedbackScratch = arena.AllocateFloats(colSize);
            bottom_up_cols = arena.AllocateFloats(outStateSize);
            colsRepacked = arena.AllocateFloats(colSize);
            lgRepacked = arena.AllocateFloats(outStateSize);
            muRepacked = arena.AllocateFloats(outStateSize);

            std::memset(b, 0, outChannels * sizeof(float));
            std::memset(mu, 0, outStateSize * sizeof(float));
            std::memset(colBuffer, 0, colSize * sizeof(float));
            std::memset(feedbackScratch, 0, colSize * sizeof(float));
            std::memset(bottom_up_cols, 0, outStateSize * sizeof(float));
            std::memset(colsRepacked, 0, colSize * sizeof(float));
            std::memset(lgRepacked, 0, outStateSize * sizeof(float));
            std::memset(muRepacked, 0, outStateSize * sizeof(float));
        }
        else
        {
            W = nullptr;
            b = nullptr;
            mu = nullptr;
            colBuffer = nullptr;
            feedbackScratch = nullptr;
            bottom_up_cols = nullptr;
            colsRepacked = nullptr;
            lgRepacked = nullptr;
            muRepacked = nullptr;
        }

        if (localArena && localArena.get() != &arena)
            localArena.reset();
    }

    void ConvPCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        if (outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t Wsz = (size_t)outChannels * colRows;
        float limit = std::sqrt(2.0f / (float)colRows); // He-style, fan-in = true receptive-field fan-in

        std::uniform_int_distribution<uint32_t> seedDist;
        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(seedGenerator);

#pragma omp parallel
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
            std::normal_distribution<float> dist(0.0f, limit);
#pragma omp for
            for (ptrdiff_t i = 0; i < (ptrdiff_t)Wsz; ++i)
                W[i] = dist(rng);
        }
    }

    float ConvPCLayer::CalculateState() noexcept
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        if (layerBelow == nullptr)
        {
            std::memset(e, 0, ownStateSize * sizeof(float));
        }
        else
        {
            cblas_scopy((int)ownStateSize, z, 1, e, 1);
            cblas_saxpy((int)ownStateSize, -1.0f, layerBelow->mu, 1, e, 1);
        }

        float totalEnergy = 0.0f;
#pragma omp parallel for schedule(static) reduction(+ : totalEnergy) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            for (size_t i = 0; i < ownSize; ++i)
            {
                float precision = std::max(p[i], 1e-8f);
                float err = e[(size_t)batch * ownSize + i];
                totalEnergy += 0.5f * precision * err * err;
                totalEnergy -= 0.5f * std::log(precision);
            }
        }

        if (outChannels > 0)
        {
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;

            // REVERTED to the original per-batch-item loop (2025 session
            // note): a repack-then-single-GEMM version of this forward pass
            // was tried and measured SLOWER (554s vs 300s/epoch) than this
            // version. Unlike UpdateState()'s feedback loop (which had ZERO
            // parallelization and got a genuine free win from adding it),
            // this loop was ALREADY parallelized from the start -- the only
            // remaining lever was reducing GEMM call-count via repacking,
            // and for these data sizes (e.g. layer with colRows=400,
            // colCols=49, batchSize=256 needs a ~19.6MB repack buffer,
            // copied twice per call) the repack/scatter memory-bandwidth
            // cost exceeded the GEMM-count savings. Do not re-attempt this
            // exact restructure without first profiling to confirm repack
            // cost vs GEMM-count savings actually favors batching at the
            // specific shapes involved.
#pragma omp parallel for schedule(static)
            for (int batch = 0; batch < batchSize; ++batch)
            {
                const float *z_item = z + (size_t)batch * ownSize;
                float *cols_item = colBuffer + (size_t)batch * colRows * colCols;

                Im2Col(z_item, inChannels, inHeight, inWidth,
                       kernelH, kernelW, strideH, strideW, padH, padW,
                       cols_item);

                float *mu_item = mu + (size_t)batch * outChannels * colCols;

                cblas_sgemm(
                    CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    outChannels, (int)colCols, (int)colRows,
                    1.0f,
                    W, (int)colRows,
                    cols_item, (int)colCols,
                    0.0f,
                    mu_item, (int)colCols);

                for (int oc = 0; oc < outChannels; ++oc)
                {
                    float bias = b[oc];
                    float *row = mu_item + (size_t)oc * colCols;
                    for (size_t j = 0; j < colCols; ++j)
                        row[j] += bias;
                }
            }

            size_t outTotal = (size_t)batchSize * outChannels * colCols;
            activation(mu, outTotal);
        }

        return totalEnergy;
    }

    void ConvPCLayer::UpdateState() noexcept
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;
        size_t colCols = (outChannels > 0) ? (size_t)outHeight * outWidth : 0;
        size_t colRows = (outChannels > 0) ? (size_t)inChannels * kernelH * kernelW : 0;

        if (outChannels > 0)
        {
            size_t outTotal = (size_t)batchSize * outChannels * colCols;
            activationDerivative(mu, outTotal, true);
        }

        if (isClamped)
            return;

        // 1. Bulletproof initialization
        std::memset(dz_dt, 0, ownStateSize * sizeof(float));

        // 2. Compute top-down feedback into dz_dt FIRST
        if (layerAbove != nullptr && outChannels > 0)
        {
            const float *e_above = layerAbove->GetErrors();
            const float *p_above = layerAbove->GetPrecisions();
            size_t outSize = (size_t)outChannels * colCols;

#pragma omp parallel for schedule(static) collapse(2)
            for (int batch = 0; batch < batchSize; ++batch)
            {
                for (size_t f = 0; f < outSize; ++f)
                {
                    size_t idx = (size_t)batch * outSize + f;
                    bottom_up_cols[idx] = e_above[idx] * p_above[f] * mu[idx];
                }
            }

#pragma omp parallel for schedule(static)
            for (int batch = 0; batch < batchSize; ++batch)
            {
                const float *lg_item = bottom_up_cols + (size_t)batch * outChannels * colCols;
                float *scratch_item = feedbackScratch + (size_t)batch * colRows * colCols;

                cblas_sgemm(
                    CblasRowMajor, CblasTrans, CblasNoTrans,
                    (int)colRows, (int)colCols, outChannels,
                    1.0f,
                    W, (int)colRows,
                    lg_item, (int)colCols,
                    0.0f,
                    scratch_item, (int)colCols);

                float *dz_item = dz_dt + (size_t)batch * ownSize;
                Col2Im(scratch_item, inChannels, inHeight, inWidth,
                       kernelH, kernelW, strideH, strideW, padH, padW,
                       dz_item);
            }
        }

        // 3. Now SUBTRACT the own term (-p * e) safely
#pragma omp parallel for schedule(static) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            for (size_t i = 0; i < ownSize; ++i)
            {
                size_t idx = (size_t)batch * ownSize + i;
                dz_dt[idx] -= p[i] * e[idx];
            }
        }

        cblas_saxpy((int)ownStateSize, ir, dz_dt, 1, z, 1);
    }
    void ConvPCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t colCols = (size_t)outHeight * outWidth;
        size_t outSize = (size_t)outChannels * colCols;

        const float *e_above = layerAbove->GetErrors();
        const float *p_above = layerAbove->GetPrecisions();

#pragma omp parallel for schedule(static) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            for (size_t f = 0; f < outSize; ++f)
            {
                size_t idx = (size_t)batch * outSize + f;
                bottom_up_cols[idx] = e_above[idx] * p_above[f] * mu[idx];
            }
        }

        if (lmbda > 0.0f)
            cblas_sscal((int)((size_t)outChannels * colRows), 1.0f - lmbda, W, 1);

        float lr_batch = lr / batchSize;

        // Repack colBuffer (batch-major: batch,row,col) into colsRepacked
        // (row,batch,col), and bottom_up_cols into lgRepacked the same way
        // -- this is what lets a SINGLE GEMM compute what used to be 256
        // separate small GEMM calls, via the identity sum_b(A_b @ B_b^T)
        // == A_stacked @ B_stacked^T. Restored here explicitly: an earlier
        // version of this function relied on CalculateState() producing
        // colsRepacked directly, but that forward-pass restructure was
        // reverted (see CalculateState()'s comment -- it regressed
        // performance), so this function must repack colBuffer itself
        // again, or it would silently read stale data.

        const ptrdiff_t maxRows = static_cast<ptrdiff_t>(colRows);
        const ptrdiff_t maxBatch1 = static_cast<ptrdiff_t>(batchSize);
#pragma omp parallel for schedule(static) collapse(2)
        for (ptrdiff_t row = 0; row < maxRows; ++row)
        {
            for (ptrdiff_t batch = 0; batch < maxBatch1; ++batch)
            {
                size_t u_row = static_cast<size_t>(row);
                size_t u_batch = static_cast<size_t>(batch);

                const float *src = colBuffer + u_batch * colRows * colCols + u_row * colCols;
                float *dst = colsRepacked + u_row * batchSize * colCols + u_batch * colCols;
                std::memcpy(dst, src, colCols * sizeof(float));
            }
        }

        const int maxOc = static_cast<int>(outChannels);
        const int maxBatch2 = static_cast<int>(batchSize);

#pragma omp parallel for schedule(static) collapse(2)
        for (int oc = 0; oc < maxOc; ++oc)
        {
            for (int batch = 0; batch < maxBatch2; ++batch)
            {
                const float *src = bottom_up_cols + (size_t)batch * outSize + (size_t)oc * colCols;
                float *dst = lgRepacked + (size_t)oc * batchSize * colCols + (size_t)batch * colCols;
                std::memcpy(dst, src, colCols * sizeof(float));
            }
        }

        // dW(outC,colRows) += lr_batch * LG(outC, batchSize*colCols) @ COLS(colRows, batchSize*colCols)^T
        // ONE GEMM replaces the previous 256-iteration serial loop.
        size_t M = (size_t)batchSize * colCols;
        cblas_sgemm(
            CblasRowMajor, CblasNoTrans, CblasTrans,
            outChannels, (int)colRows, (int)M,
            lr_batch,
            lgRepacked, (int)M,
            colsRepacked, (int)M,
            1.0f, // accumulate onto existing W
            W, (int)colRows);

        // Bias gradient: unchanged -- a cheap reduction, not a GEMM, was
        // never the bottleneck. Reads from the original (unrepacked)
        // bottom_up_cols directly.
        for (int batch = 0; batch < batchSize; ++batch)
        {
            const float *lg_item = bottom_up_cols + (size_t)batch * outSize;
            for (int oc = 0; oc < outChannels; ++oc)
            {
                const float *row = lg_item + (size_t)oc * colCols;
                float sum = 0.0f;
                for (size_t j = 0; j < colCols; ++j)
                    sum += row[j];
                b[oc] += lr_batch * sum;
            }
        }
    }

    void ConvPCLayer::UpdatePrecision() noexcept
    {
        if (layerBelow == nullptr)
            return;

        size_t ownSize = (size_t)inChannels * inHeight * inWidth;

#pragma omp parallel for schedule(static)
        for (ptrdiff_t i = 0; i < (ptrdiff_t)ownSize; ++i)
        {
            float grad = 0.0f;
            for (int batch = 0; batch < batchSize; ++batch)
            {
                float err = e[(size_t)batch * ownSize + i];
                grad += 0.5f * (p[i] * err * err - 1.0f);
            }
            grad /= batchSize;
            log_p[i] -= pr * grad;
            log_p[i] = std::max(-5.0f, std::min(log_p[i], 5.0f));
            p[i] = std::exp(log_p[i]);
        }
    }

    void ConvPCLayer::ResetState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        std::memset(z, 0, ownStateSize * sizeof(float));
    }

    void ConvPCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        size_t copySize = std::min(inputData.size(), ownStateSize) * sizeof(float);
        std::memcpy(z, inputData.data(), copySize);
        isClamped = true;
    }

    void ConvPCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    void ConvPCLayer::ResyncLogPrecision() noexcept
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        for (size_t i = 0; i < ownSize; ++i)
            log_p[i] = std::log(std::max(p[i], 1e-8f));
    }
}
