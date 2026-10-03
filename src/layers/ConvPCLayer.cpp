#include <deepity/layers/ConvPCLayer.h>
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

    ConvPCLayer::ConvPCLayer(int inChannels, int outChannels,
                             int inHeight, int inWidth,
                             int kernelH, int kernelW,
                             int strideH, int strideW,
                             int padH, int padW,
                             int batchSize,
                             float learningRate, float inferenceRate,
                             float precisionRate, float lmbda,
                             ActivationType aType, ActivationType dType,
                             IComputeBackend *backend)
        : inChannels(inChannels), outChannels(outChannels),
          inHeight(inHeight), inWidth(inWidth),
          kernelH(kernelH), kernelW(kernelW),
          strideH(strideH), strideW(strideW),
          padH(padH), padW(padW),
          batchSize(batchSize),
          lr(learningRate), ir(inferenceRate), pr(precisionRate), lmbda(lmbda),
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

    void ConvPCLayer::SetLearningRate(float learningRate) noexcept
    {
        lr = learningRate;
        if (lr_device)
            backend->CopyFromHost(lr_device, &lr, 1);
    }

    size_t ConvPCLayer::GetRequiredFloats() const noexcept
    {
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
            size_t Wsize = (size_t)outChannels * colRows;
            size_t M = (size_t)batchSize * colCols;

            total += pad16(Wsize);                           // W
            total += pad16((size_t)outChannels);             // b
            total += pad16(outStateSize) * 2;                // mu, cachedMu
            total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
            total += pad16(outStateSize);                    // bottom_up_cols
            total += pad16((size_t)batchSize * colSize);     // colsRepacked
            total += pad16(outStateSize);                    // lgRepacked
            total += pad16(M);                               // onesVector

            // Always allocated, regardless of optimizer, so switching to Adam/AdamW after Compile() stays safe.
            total += pad16(Wsize) * 3;               // grad_W, m_W, v_W
            total += pad16((size_t)outChannels) * 3; // grad_b, m_b, v_b
            total += pad16(1) * 2;                   // t_device, lr_device
        }

        return total;
    }

    std::map<std::string, TensorDescriptor> ConvPCLayer::GetStateDict() const
    {
        return {
            {"W", {W, {(size_t)outChannels, (size_t)(inChannels * kernelH * kernelW)}}},
            {"b", {b, {(size_t)outChannels}}},
            {"p", {p, {(size_t)inChannels * inHeight * inWidth}}}};
    }

    template <typename ArenaT>
    void ConvPCLayer::BindMemory(ArenaT &arena)
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        z = arena.AllocateFloats(ownStateSize);
        e = arena.AllocateFloats(ownStateSize);
        dz_dt = arena.AllocateFloats(ownStateSize);
        p = arena.AllocateFloats(ownSize);
        log_p = arena.AllocateFloats(ownSize);

        backend->Zero(z, ownStateSize);
        backend->Zero(e, ownStateSize);
        backend->Zero(dz_dt, ownStateSize);
        backend->Fill(p, ownSize, 1.0f);
        backend->Zero(log_p, ownSize);

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
            onesVector = arena.AllocateFloats(M);

            backend->Zero(b, outChannels);
            backend->Zero(mu, outStateSize);
            backend->Zero(cachedMu, outStateSize);
            backend->Zero(colBuffer, colSize);
            backend->Zero(feedbackScratch, colSize);
            backend->Zero(bottom_up_cols, outStateSize);
            backend->Zero(colsRepacked, colSize);
            backend->Zero(lgRepacked, outStateSize);
            backend->Fill(onesVector, M, 1.0f);

            // Always allocated, see GetRequiredFloats().
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

    void ConvPCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
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

    float ConvPCLayer::CalculateState(bool needEnergy) noexcept
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;

        // Zero-Energy bypass, matches the original's -0.5*sum(log_p)
        // formula for the input layer (no e term at all, since there's
        // no layerBelow to compare against). backend->Sum() is a
        // synchronous GPU readback (same constraint as
        // ComputePrecisionWeightedErrorAndEnergy below), so it's only
        // computed when actually needed -- also fixes a real bug this
        // branch had before any needEnergy existed: indexing log_p[i]
        // directly from host code is illegal once log_p is a GPU device
        // pointer (DeviceMemoryArena), regardless of graph capture.
        if (layerBelow == nullptr)
        {
            backend->Zero(e, ownStateSize);
            float totalEnergy = 0.0f;
            if (needEnergy)
                totalEnergy = -0.5f * (float)batchSize * backend->Sum(log_p, ownSize);
            if (outChannels > 0)
                ComputeMuOnly();
            return totalEnergy;
        }

        float totalEnergy = 0.0f;
        if (needEnergy)
            totalEnergy = backend->ComputePrecisionWeightedErrorAndEnergy(
                e, z, layerBelow->mu, p, batchSize, ownSize);
        else
            backend->ComputePrecisionWeightedError(e, z, layerBelow->mu, p, batchSize, ownSize);

        if (outChannels > 0)
            ComputeMuOnly();

        return totalEnergy;
    }

    void ConvPCLayer::ComputeMuOnly() noexcept
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

        // Im2Col is an inherently per-image gather (no batched primitive);
        // everything after it is batched into one big GEMM instead of
        // batchSize small ones, reusing colsRepacked/lgRepacked as pure
        // scratch exactly as UpdateWeights() already does for its own GEMM
        // (see FullConvPCLayer::ComputeMuOnly() for the identical trick).
        for (int batch = 0; batch < batchSize; ++batch)
        {
            const float *z_item = z + (size_t)batch * ownSize;
            float *cols_item = colBuffer + (size_t)batch * colRows * colCols;

            backend->Im2Col(z_item, inChannels, inHeight, inWidth,
                            kernelH, kernelW, strideH, strideW, padH, padW,
                            cols_item);
        }

        backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
        backend->MatMul(
            /*transA=*/false, /*transB=*/false,
            outChannels, (int)(batchSize * colCols), (int)colRows,
            1.0f, W, (int)colRows, colsRepacked, (int)(batchSize * colCols),
            0.0f, lgRepacked, (int)(batchSize * colCols));
        backend->AddBiasPerChannel(lgRepacked, b, outChannels, batchSize * colCols);
        backend->RepackForBatchedGemm(mu, lgRepacked, outChannels, batchSize, colCols);

        backend->Activation(activationType, mu, Nout);

        if (isClamped)
        {
            backend->Copy(cachedMu, mu, Nout);
            muCacheValid = true;
        }
    }

    void ConvPCLayer::UpdateState() noexcept
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;
        size_t colCols = (outChannels > 0) ? (size_t)outHeight * outWidth : 0;
        size_t colRows = (outChannels > 0) ? (size_t)inChannels * kernelH * kernelW : 0;

        // mu must end up holding the derivative whenever outChannels > 0,
        // regardless of isClamped/layerAbove, UpdateWeights() reads it
        // unconditionally later.
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
            const float *p_above = layerAbove->GetPrecisions();
            size_t outSize = (size_t)outChannels * colCols;

            // bottom_up_cols[idx] = e_above[idx] * p_above[i] * mu[idx]
            // (mu already holds the derivative from the call above).
            backend->MultiplyBroadcastInto(bottom_up_cols, e_above, p_above, mu, batchSize, outSize);

            // Same batched-GEMM trick as ComputeMuOnly(): one big transA
            // GEMM instead of batchSize small ones, un-repacking the
            // result back to per-batch-contiguous layout only where
            // Col2Im (a genuine per-image scatter) actually needs it.
            backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);
            backend->MatMul(
                /*transA=*/true, /*transB=*/false,
                (int)colRows, (int)(batchSize * colCols), outChannels,
                1.0f, W, (int)colRows, lgRepacked, (int)(batchSize * colCols),
                0.0f, colsRepacked, (int)(batchSize * colCols));
            backend->RepackForBatchedGemm(feedbackScratch, colsRepacked, colRows, batchSize, colCols);

            for (int batch = 0; batch < batchSize; ++batch)
            {
                float *scratch_item = feedbackScratch + (size_t)batch * colRows * colCols;
                float *dz_item = dz_dt + (size_t)batch * ownSize;
                backend->Col2Im(scratch_item, inChannels, inHeight, inWidth,
                                kernelH, kernelW, strideH, strideW, padH, padW,
                                dz_item);
            }
        }

        // dz_dt[idx] -= p[i] * e[idx] (own term), then z += ir * dz_dt.
        backend->AxpyBroadcastInto(dz_dt, e, p, batchSize, ownSize, -1.0f);
        backend->AxpyInto(z, dz_dt, ownStateSize, ir);
    }

    void ConvPCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || outChannels == 0)
            return;

        size_t colRows = (size_t)inChannels * kernelH * kernelW;
        size_t colCols = (size_t)outHeight * outWidth;
        size_t outSize = (size_t)outChannels * colCols;
        size_t Wsize = (size_t)outChannels * colRows;
        size_t outTotal = (size_t)batchSize * outSize;

        const float *e_above = layerAbove->GetErrors();
        const float *p_above = layerAbove->GetPrecisions();

        // mu still holds the derivative left over from the last
        // UpdateState() call.
        backend->MultiplyBroadcastInto(bottom_up_cols, e_above, p_above, mu, batchSize, outSize);

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

            // Bias gradient via GEMM against an all-ones vector.
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

    void ConvPCLayer::UpdatePrecision() noexcept
    {
        if (layerBelow == nullptr)
            return;

        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        backend->UpdatePrecisionFromError(p, log_p, e, batchSize, ownSize, pr);
    }

    void ConvPCLayer::ResetState() noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        backend->Zero(z, ownStateSize);
    }

    void ConvPCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        size_t ownStateSize = (size_t)batchSize * inChannels * inHeight * inWidth;
        size_t copyFloats = (std::min)(inputData.size(), ownStateSize);
        backend->CopyFromHost(z, inputData.data(), copyFloats);
        isClamped = true;
        muCacheValid = false;
    }

    void ConvPCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }

    void ConvPCLayer::ResyncLogPrecision() noexcept
    {
        // Cold path (checkpoint load only), a host round-trip keeps
        // this correct regardless of backend/device rather than adding a
        // dedicated primitive for a log-transform that only ever runs
        // once per load.
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        std::vector<float> hostP(ownSize);
        backend->CopyToHost(hostP.data(), p, ownSize);

        std::vector<float> hostLogP(ownSize);
        for (size_t i = 0; i < ownSize; ++i)
            hostLogP[i] = std::log(std::max(hostP[i], 1e-8f));

        backend->CopyFromHost(log_p, hostLogP.data(), ownSize);
    }

    template void ConvPCLayer::BindMemory<MemoryArena>(MemoryArena &arena);
#if defined(DEEPITY_USE_CUDA)
    template void ConvPCLayer::BindMemory<DeviceMemoryArena>(DeviceMemoryArena &arena);
#endif
}
