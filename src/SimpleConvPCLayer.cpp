#include <deepity/layers/SimpleConvPCLayer.h>
#ifdef DEEPITY_USE_MKL
#include <mkl_cblas.h>
#else
#include <cblas.h>
#endif
#include <omp.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <cstddef>

namespace Deep
{
    SimpleConvPCLayer::SimpleConvPCLayer(int inChannels, int outChannels,
                                         int inHeight, int inWidth,
                                         int kernelH, int kernelW,
                                         int strideH, int strideW,
                                         int padH, int padW,
                                         int batchSize,
                                         float learningRate, float inferenceRate,
                                         float lmbda,
                                         ActivationType aType, ActivationType dType)
        : inChannels(inChannels), outChannels(outChannels),
          inHeight(inHeight), inWidth(inWidth),
          kernelH(kernelH), kernelW(kernelW),
          strideH(strideH), strideW(strideW),
          padH(padH), padW(padW),
          batchSize(batchSize),
          lr(learningRate), ir(inferenceRate), lmbda(lmbda),
          layerAbove(nullptr), layerBelow(nullptr), activationType(aType)
    {
        outHeight = (outChannels > 0) ? ConvOutDim(inHeight, kernelH, strideH, padH) : 0;
        outWidth = (outChannels > 0) ? ConvOutDim(inWidth, kernelW, strideW, padW) : 0;

        this->activation = To_Fn(aType);
        this->activationDerivative = To_dFn(dType);

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    size_t SimpleConvPCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        total += pad16(ownStateSize) * 3; // z, e, dz_dt -- NO p/log_p at all

        if (outChannels > 0)
        {
            size_t outSize = (size_t)outChannels * outHeight * outWidth;
            size_t outStateSize = (size_t)batchSize * outSize;
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;
            size_t colSize = colRows * colCols;
            size_t Wsize = (size_t)outChannels * colRows;

            total += pad16(Wsize);                           // W
            total += pad16((size_t)outChannels);             // b
            total += pad16(outStateSize);                    // mu
            total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
            total += pad16(outStateSize);                    // bottom_up_cols
            total += pad16((size_t)batchSize * colSize);     // colsRepacked
            total += pad16(outStateSize);                    // lgRepacked
            total += pad16(outStateSize);                    // muRepacked

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(Wsize) * 3;               // grad_W, m_W, v_W
                total += pad16((size_t)outChannels) * 3; // grad_b, m_b, v_b
            }
        }

        return total;
    }

    void SimpleConvPCLayer::BindMemory(MemoryArena &arena)
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        z = arena.AllocateFloats(ownStateSize);
        e = arena.AllocateFloats(ownStateSize);
        dz_dt = arena.AllocateFloats(ownStateSize);

        std::memset(z, 0, ownStateSize * sizeof(float));
        std::memset(e, 0, ownStateSize * sizeof(float));
        std::memset(dz_dt, 0, ownStateSize * sizeof(float));

        if (outChannels > 0)
        {
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;
            size_t outStateSize = (size_t)batchSize * outChannels * colCols;
            size_t colSize = (size_t)batchSize * colRows * colCols;
            size_t Wsize = (size_t)outChannels * colRows;

            W = arena.AllocateFloats(Wsize);
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

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                grad_W = arena.AllocateFloats(Wsize);
                m_W = arena.AllocateFloats(Wsize);
                v_W = arena.AllocateFloats(Wsize);
                grad_b = arena.AllocateFloats(outChannels);
                m_b = arena.AllocateFloats(outChannels);
                v_b = arena.AllocateFloats(outChannels);

                std::memset(m_W, 0, Wsize * sizeof(float));
                std::memset(v_W, 0, Wsize * sizeof(float));
                std::memset(m_b, 0, outChannels * sizeof(float));
                std::memset(v_b, 0, outChannels * sizeof(float));
                std::memset(grad_W, 0, Wsize * sizeof(float));
                std::memset(grad_b, 0, outChannels * sizeof(float));
            }
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

    void SimpleConvPCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        if (outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t Wsz = (size_t)outChannels * colRows;
        float limit = std::sqrt(2.0f / (float)colRows);

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

    float SimpleConvPCLayer::CalculateState() noexcept
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

        // No precision weighting: E = 0.5*sum(e^2), plain SSE.
        float totalEnergy = 0.0f;
#pragma omp parallel for schedule(static) reduction(+ : totalEnergy) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            for (size_t i = 0; i < ownSize; ++i)
            {
                float err = e[(size_t)batch * ownSize + i];
                totalEnergy += 0.5f * err * err;
            }
        }

        if (outChannels > 0)
        {
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;

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
                    1.0f, W, (int)colRows, cols_item, (int)colCols,
                    0.0f, mu_item, (int)colCols);

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

    void SimpleConvPCLayer::UpdateState() noexcept
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

        std::memset(dz_dt, 0, ownStateSize * sizeof(float));

        // Top-down feedback -- NO p_above multiply (was e_above*p_above*mu;
        // p_above=1 always made that a no-op).
        if (layerAbove != nullptr && outChannels > 0)
        {
            const float *e_above = layerAbove->GetErrors();
            size_t outSize = (size_t)outChannels * colCols;

#pragma omp parallel for schedule(static) collapse(2)
            for (int batch = 0; batch < batchSize; ++batch)
            {
                for (size_t f = 0; f < outSize; ++f)
                {
                    size_t idx = (size_t)batch * outSize + f;
                    bottom_up_cols[idx] = e_above[idx] * mu[idx];
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
                    1.0f, W, (int)colRows, lg_item, (int)colCols,
                    0.0f, scratch_item, (int)colCols);

                float *dz_item = dz_dt + (size_t)batch * ownSize;
                Col2Im(scratch_item, inChannels, inHeight, inWidth,
                       kernelH, kernelW, strideH, strideW, padH, padW,
                       dz_item);
            }
        }

        // Own term: dz_dt -= e (was -p*e; p=1 always made this a no-op multiply).
#pragma omp parallel for schedule(static) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            for (size_t i = 0; i < ownSize; ++i)
            {
                size_t idx = (size_t)batch * ownSize + i;
                dz_dt[idx] -= e[idx];
            }
        }

        cblas_saxpy((int)ownStateSize, ir, dz_dt, 1, z, 1);
    }

    void SimpleConvPCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t colCols = (size_t)outHeight * outWidth;
        size_t outSize = (size_t)outChannels * colCols;
        size_t Wsize = (size_t)outChannels * colRows;

        const float *e_above = layerAbove->GetErrors();

        // local_grad = e_above * mu(f') -- NO p_above multiply.
#pragma omp parallel for schedule(static) collapse(2)
        for (int batch = 0; batch < batchSize; ++batch)
        {
            for (size_t f = 0; f < outSize; ++f)
            {
                size_t idx = (size_t)batch * outSize + f;
                bottom_up_cols[idx] = e_above[idx] * mu[idx];
            }
        }

        // Repack (shared by both optimizer paths -- identical to ConvPCLayer's).
        const int maxRow = static_cast<int>(colRows);
        const int maxBatch = static_cast<int>(batchSize);
        const int maxOc = static_cast<int>(outChannels);

#pragma omp parallel for schedule(static) collapse(2)
        for (int row = 0; row < maxRow; ++row)
        {
            for (int batch = 0; batch < maxBatch; ++batch)
            {
                size_t u_row = static_cast<size_t>(row);
                size_t u_batch = static_cast<size_t>(batch);

                const float *src = colBuffer + u_batch * colRows * colCols + u_row * colCols;
                float *dst = colsRepacked + u_row * batchSize * colCols + u_batch * colCols;
                std::memcpy(dst, src, colCols * sizeof(float));
            }
        }

#pragma omp parallel for schedule(static) collapse(2)
        for (int oc = 0; oc < maxOc; ++oc)
        {
            for (int batch = 0; batch < maxBatch; ++batch)
            {
                size_t u_oc = static_cast<size_t>(oc);
                size_t u_batch = static_cast<size_t>(batch);

                const float *src = bottom_up_cols + u_batch * outSize + u_oc * colCols;
                float *dst = lgRepacked + u_oc * batchSize * colCols + u_batch * colCols;
                std::memcpy(dst, src, colCols * sizeof(float));
            }
        }

        size_t M = (size_t)batchSize * colCols;

        switch (opt)
        {
        case OptimizerType::SGD:
        {
            if (lmbda > 0.0f)
                cblas_sscal((int)Wsize, 1.0f - lmbda, W, 1);

            float lr_batch = lr / batchSize;

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                outChannels, (int)colRows, (int)M,
                lr_batch, lgRepacked, (int)M, colsRepacked, (int)M,
                1.0f, W, (int)colRows);

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
            break;
        }
        case OptimizerType::ADAM:
        case OptimizerType::ADAMW:
        {
            t++;

            // NEGATIVE scale, matching SimplePCLayer's confirmed sign fix:
            // this GEMM's un-negated form was verified (via the SGD path
            // above, and ConvPCLayer's own gradient check) to compute a
            // quantity that's already the NEGATIVE of the true dE/dW --
            // AdamWUpdate/AdamUpdate expect the TRUE (positive) gradient,
            // so this needs the explicit sign flip SimplePCLayer's AdamW
            // integration needed. NOT yet independently re-verified for
            // conv specifically -- see file-level warning.
            float grad_scale = -1.0f / batchSize;

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                outChannels, (int)colRows, (int)M,
                grad_scale, lgRepacked, (int)M, colsRepacked, (int)M,
                0.0f, grad_W, (int)colRows);

            std::memset(grad_b, 0, outChannels * sizeof(float));
            for (int batch = 0; batch < batchSize; ++batch)
            {
                const float *lg_item = bottom_up_cols + (size_t)batch * outSize;
                for (int oc = 0; oc < outChannels; ++oc)
                {
                    const float *row = lg_item + (size_t)oc * colCols;
                    float sum = 0.0f;
                    for (size_t j = 0; j < colCols; ++j)
                        sum += row[j];
                    grad_b[oc] += grad_scale * sum;
                }
            }

            if (opt == OptimizerType::ADAMW)
                Deep::AdamWUpdate(W, grad_W, m_W, v_W, Wsize, t, lr, lmbda);
            else
                Deep::AdamUpdate(W, grad_W, m_W, v_W, Wsize, t, lr);

            Deep::AdamUpdate(b, grad_b, m_b, v_b, outChannels, t, lr);
            break;
        }
        }
    }

    void SimpleConvPCLayer::ResetState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        std::memset(z, 0, ownStateSize * sizeof(float));
    }

    void SimpleConvPCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        size_t copySize = std::min(inputData.size(), ownStateSize) * sizeof(float);
        std::memcpy(z, inputData.data(), copySize);
        isClamped = true;
    }

    void SimpleConvPCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }
}
