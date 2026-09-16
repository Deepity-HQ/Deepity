#include <deepity/layers/SimpleConvPCLayer.h>
#include <deepity/backend/CPUBackend.h>
#include <cmath>
#include <type_traits>

namespace Deep
{
    namespace
    {
        void DeleteBackend(IComputeBackend *p) { delete p; }
        void NoOpDeleter(IComputeBackend *) {}
    }

    SimpleConvPCLayer::SimpleConvPCLayer(int inChannels, int outChannels,
                                         int inHeight, int inWidth,
                                         int kernelH, int kernelW,
                                         int strideH, int strideW,
                                         int padH, int padW,
                                         int batchSize,
                                         float learningRate, float inferenceRate,
                                         float lmbda,
                                         ActivationType aType, ActivationType dType,
                                         IComputeBackend *backend)
        : inChannels(inChannels), outChannels(outChannels),
          inHeight(inHeight), inWidth(inWidth),
          kernelH(kernelH), kernelW(kernelW),
          strideH(strideH), strideW(strideW),
          padH(padH), padW(padW),
          batchSize(batchSize),
          lr(learningRate), ir(inferenceRate), lmbda(lmbda),
          layerAbove(nullptr), layerBelow(nullptr),
          activationType(aType), derivativeType(dType),
          backend(backend ? backend : new CPUBackend(),
                  backend ? NoOpDeleter : DeleteBackend)
    {
        outHeight = (outChannels > 0) ? ConvOutDim(inHeight, kernelH, strideH, padH) : 0;
        outWidth = (outChannels > 0) ? ConvOutDim(inWidth, kernelW, strideW, padW) : 0;

        localArena = std::make_unique<MemoryArena>(GetRequiredFloats());
        BindMemory(*localArena);
    }

    void SimpleConvPCLayer::SetLearningRate(float learningRate) noexcept
    {
        lr = learningRate;
        if (lr_device)
            backend->CopyFromHost(lr_device, &lr, 1);
    }

    size_t SimpleConvPCLayer::GetRequiredFloats() const noexcept
    {
        auto pad16 = [](size_t n)
        { return (n + 15) & ~(size_t)15; };

        size_t total = 0;
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        total += pad16(ownStateSize) * 3; // z, e, dz_dt

        if (outChannels > 0)
        {
            size_t outSize = (size_t)outChannels * outHeight * outWidth;
            size_t outStateSize = (size_t)batchSize * outSize;
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;
            size_t colSize = colRows * colCols;
            size_t Wsize = (size_t)outChannels * colRows;
            size_t M = (size_t)batchSize * colCols;

            total += pad16(Wsize);                           // W
            total += pad16((size_t)outChannels);             // b
            total += pad16(outStateSize) * 2;                // mu, cachedMu
            total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
            total += pad16(outStateSize);                    // bottom_up_cols
            total += pad16((size_t)batchSize * colSize);     // colsRepacked
            total += pad16(outStateSize);                    // lgRepacked
            total += pad16(outStateSize);                    // muRepacked
            total += pad16(M);                               // onesVector

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                total += pad16(Wsize) * 3;               // grad_W, m_W, v_W
                total += pad16((size_t)outChannels) * 3; // grad_b, m_b, v_b
                total += pad16(1) * 2;                   // t_device, lr_device
            }
        }

        return total;
    }

    template <typename ArenaT>
    void SimpleConvPCLayer::BindMemory(ArenaT &arena)
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        z = arena.AllocateFloats(ownStateSize);
        e = arena.AllocateFloats(ownStateSize);
        dz_dt = arena.AllocateFloats(ownStateSize);

        backend->Zero(z, ownStateSize);
        backend->Zero(e, ownStateSize);
        backend->Zero(dz_dt, ownStateSize);

        if (outChannels > 0)
        {
            size_t colRows = (size_t)inChannels * kernelH * kernelW;
            size_t colCols = (size_t)outHeight * outWidth;
            size_t outStateSize = (size_t)batchSize * outChannels * colCols;
            size_t colSize = (size_t)batchSize * colRows * colCols;
            size_t Wsize = (size_t)outChannels * colRows;
            size_t M = (size_t)batchSize * colCols;

            W = arena.AllocateFloats(Wsize);
            b = arena.AllocateFloats(outChannels);
            mu = arena.AllocateFloats(outStateSize);
            cachedMu = arena.AllocateFloats(outStateSize);
            colBuffer = arena.AllocateFloats(colSize);
            feedbackScratch = arena.AllocateFloats(colSize);
            bottom_up_cols = arena.AllocateFloats(outStateSize);
            colsRepacked = arena.AllocateFloats(colSize);
            lgRepacked = arena.AllocateFloats(outStateSize);
            muRepacked = arena.AllocateFloats(outStateSize);
            onesVector = arena.AllocateFloats(M);

            backend->Zero(b, outChannels);
            backend->Zero(mu, outStateSize);
            backend->Zero(cachedMu, outStateSize);
            backend->Zero(colBuffer, colSize);
            backend->Zero(feedbackScratch, colSize);
            backend->Zero(bottom_up_cols, outStateSize);
            backend->Zero(colsRepacked, colSize);
            backend->Zero(lgRepacked, outStateSize);
            backend->Zero(muRepacked, outStateSize);
            backend->Fill(onesVector, M, 1.0f);

            if (opt == OptimizerType::ADAM || opt == OptimizerType::ADAMW)
            {
                grad_W = arena.AllocateFloats(Wsize);
                m_W = arena.AllocateFloats(Wsize);
                v_W = arena.AllocateFloats(Wsize);
                grad_b = arena.AllocateFloats(outChannels);
                m_b = arena.AllocateFloats(outChannels);
                v_b = arena.AllocateFloats(outChannels);

                backend->Zero(m_W, Wsize);
                backend->Zero(v_W, Wsize);
                backend->Zero(m_b, outChannels);
                backend->Zero(v_b, outChannels);
                backend->Zero(grad_W, Wsize);
                backend->Zero(grad_b, outChannels);

                t_device = reinterpret_cast<int *>(arena.AllocateFloats(1));
                lr_device = arena.AllocateFloats(1);
                int zero = 0;
                backend->CopyFromHost(reinterpret_cast<float *>(t_device), reinterpret_cast<float *>(&zero), 1);
                backend->CopyFromHost(lr_device, &lr, 1);
            }
        }
        else
        {
            W = nullptr;
            b = nullptr;
            mu = nullptr;
            cachedMu = nullptr;
            colBuffer = nullptr;
            feedbackScratch = nullptr;
            bottom_up_cols = nullptr;
            colsRepacked = nullptr;
            lgRepacked = nullptr;
            muRepacked = nullptr;
            onesVector = nullptr;
        }

        if constexpr (std::is_same_v<ArenaT, MemoryArena>)
        {
            if (localArena && localArena.get() != &arena)
                localArena.reset();
        }
        else
        {
            localArena.reset();
        }
    }

    void SimpleConvPCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        if (outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t Wsz = (size_t)outChannels * colRows;
        float limit = std::sqrt(2.0f / (float)colRows);

        std::uniform_int_distribution<uint32_t> seedDist;
        uint32_t seed = seedDist(seedGenerator);

        backend->RandomizeNormal(W, Wsz, 0.0f, limit, seed);
    }

    float SimpleConvPCLayer::CalculateState() noexcept
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        if (layerBelow == nullptr)
        {
            backend->Zero(e, ownStateSize);
            if (outChannels > 0)
                ComputeMuOnly();
            return 0.0f;
        }

        float totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->mu, ownStateSize);

        if (outChannels > 0)
            ComputeMuOnly();

        return totalEnergy;
    }

    void SimpleConvPCLayer::ComputeMuOnly() noexcept
    {
        if (outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t colCols = (size_t)outHeight * outWidth;
        size_t Nout = (size_t)batchSize * outChannels * colCols;
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;

        if (isClamped && muCacheValid)
        {
            backend->Copy(mu, cachedMu, Nout);
            return;
        }

        for (int batch = 0; batch < batchSize; ++batch)
        {
            const float *z_item = z + (size_t)batch * ownSize;
            float *cols_item = colBuffer + (size_t)batch * colRows * colCols;

            backend->Im2Col(z_item, inChannels, inHeight, inWidth,
                            kernelH, kernelW, strideH, strideW, padH, padW,
                            cols_item);

            float *mu_item = mu + (size_t)batch * outChannels * colCols;

            backend->MatMul(
                /*transA=*/false, /*transB=*/false,
                outChannels, (int)colCols, (int)colRows,
                1.0f, W, (int)colRows, cols_item, (int)colCols,
                0.0f, mu_item, (int)colCols);

            backend->AddBiasPerChannel(mu_item, b, outChannels, colCols);
        }

        backend->Activation(activationType, mu, Nout);

        if (isClamped)
        {
            backend->Copy(cachedMu, mu, Nout);
            muCacheValid = true;
        }
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
            backend->ActivationDerivative(derivativeType, mu, outTotal, true);
        }

        if (isClamped)
            return;

        backend->Zero(dz_dt, ownStateSize);

        if (layerAbove != nullptr && outChannels > 0)
        {
            const float *e_above = layerAbove->GetErrors();
            size_t outSize = (size_t)outChannels * colCols;
            size_t outTotal = (size_t)batchSize * outSize;

            backend->MultiplyInto(bottom_up_cols, e_above, mu, outTotal);

            for (int batch = 0; batch < batchSize; ++batch)
            {
                const float *lg_item = bottom_up_cols + (size_t)batch * outChannels * colCols;
                float *scratch_item = feedbackScratch + (size_t)batch * colRows * colCols;

                backend->MatMul(
                    /*transA=*/true, /*transB=*/false,
                    (int)colRows, (int)colCols, outChannels,
                    1.0f, W, (int)colRows, lg_item, (int)colCols,
                    0.0f, scratch_item, (int)colCols);

                float *dz_item = dz_dt + (size_t)batch * ownSize;
                backend->Col2Im(scratch_item, inChannels, inHeight, inWidth,
                                kernelH, kernelW, strideH, strideW, padH, padW,
                                dz_item);
            }
        }

        backend->AxpyInto(dz_dt, e, ownStateSize, -1.0f);
        backend->AxpyInto(z, dz_dt, ownStateSize, ir);
    }

    void SimpleConvPCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t colCols = (size_t)outHeight * outWidth;
        size_t outSize = (size_t)outChannels * colCols;
        size_t Wsize = (size_t)outChannels * colRows;
        size_t outTotal = (size_t)batchSize * outSize;

        const float *e_above = layerAbove->GetErrors();

        backend->MultiplyInto(bottom_up_cols, e_above, mu, outTotal);

        backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
        backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);

        size_t M = (size_t)batchSize * colCols;

        switch (opt)
        {
        case OptimizerType::SGD:
        {
            if (lmbda > 0.0f)
                backend->Scale(W, Wsize, 1.0f - lmbda);

            float lr_batch = lr / batchSize;

            backend->MatMul(
                /*transA=*/false, /*transB=*/true,
                outChannels, (int)colRows, (int)M,
                lr_batch, lgRepacked, (int)M, colsRepacked, (int)M,
                1.0f, W, (int)colRows);

            // Bias gradient: db[oc] = lr_batch * sum over lgRepacked's
            // oc-th row (M contiguous values -- both batch AND spatial
            // combined). NOT the same reduction shape as SumRows (which
            // reduces across batch only) -- expressed instead as a GEMM
            // against an all-ones vector: db = lgRepacked[outChannels,M]
            // @ ones[M,1]. beta=1.0f accumulates onto b directly,
            // matching the original loop's `b[oc] += ...`.
            backend->MatMul(
                /*transA=*/false, /*transB=*/false,
                outChannels, 1, (int)M,
                lr_batch, lgRepacked, (int)M, onesVector, 1,
                1.0f, b, 1);
            break;
        }
        case OptimizerType::ADAM:
        case OptimizerType::ADAMW:
        {
            backend->IncrementCounter(t_device);

            float grad_scale = -1.0f / batchSize;

            backend->MatMul(
                /*transA=*/false, /*transB=*/true,
                outChannels, (int)colRows, (int)M,
                grad_scale, lgRepacked, (int)M, colsRepacked, (int)M,
                0.0f, grad_W, (int)colRows);

            // grad_b = grad_scale * (lgRepacked @ ones) -- beta=0.0f
            // overwrites, matching the original's memset(grad_b,0,...)
            // followed by accumulation (equivalent since nothing else
            // writes grad_b between the memset and this sum).
            backend->MatMul(
                /*transA=*/false, /*transB=*/false,
                outChannels, 1, (int)M,
                grad_scale, lgRepacked, (int)M, onesVector, 1,
                0.0f, grad_b, 1);

            if (opt == OptimizerType::ADAMW)
                backend->AdamWStep(W, grad_W, m_W, v_W, Wsize, t_device, lr_device, lmbda);
            else
                backend->AdamStep(W, grad_W, m_W, v_W, Wsize, t_device, lr_device);

            backend->AdamStep(b, grad_b, m_b, v_b, outChannels, t_device, lr_device);
            break;
        }
        }
    }

    void SimpleConvPCLayer::ResetState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        backend->Zero(z, ownStateSize);
    }

    void SimpleConvPCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        size_t copyFloats = (std::min)(inputData.size(), ownStateSize);
        backend->CopyFromHost(z, inputData.data(), copyFloats);
        isClamped = true;
        muCacheValid = false;
    }

    void SimpleConvPCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    template void SimpleConvPCLayer::BindMemory<MemoryArena>(MemoryArena &arena);
#if defined(DEEPITY_USE_CUDA)
    template void SimpleConvPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena &arena);
#endif
}