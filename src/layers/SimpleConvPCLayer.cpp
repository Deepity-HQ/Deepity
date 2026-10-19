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
            total += pad16(outStateSize) * 4;                // mu, cachedMu, preact, cachedPreact
            total += pad16((size_t)batchSize * colSize) * 2; // colBuffer, feedbackScratch
            total += pad16(outStateSize);                    // bottom_up_cols
            total += pad16((size_t)batchSize * colSize);     // colsRepacked
            total += pad16(outStateSize);                    // lgRepacked
            total += pad16(outStateSize);                    // muRepacked
            total += pad16(M);                               // onesVector

            // Always allocated, regardless of optimizer, so switching to Adam/AdamW after Compile() stays safe.
            total += pad16(Wsize) * 3;               // grad_W, m_W, v_W
            total += pad16((size_t)outChannels) * 3; // grad_b, m_b, v_b
            total += pad16(1) * 2;                   // t_device, lr_device
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
            preact = arena.AllocateFloats(outStateSize);
            cachedPreact = arena.AllocateFloats(outStateSize);
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
            backend->Zero(preact, outStateSize);
            backend->Zero(cachedPreact, outStateSize);
            backend->Zero(colBuffer, colSize);
            backend->Zero(feedbackScratch, colSize);
            backend->Zero(bottom_up_cols, outStateSize);
            backend->Zero(colsRepacked, colSize);
            backend->Zero(lgRepacked, outStateSize);
            backend->Zero(muRepacked, outStateSize);
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
            preact = nullptr;
            cachedPreact = nullptr;
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

    float SimpleConvPCLayer::CalculateState(bool needEnergy) noexcept
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

        float totalEnergy = 0.0f;
        if (needEnergy)
            totalEnergy = backend->ComputeErrorAndEnergy(e, z, layerBelow->mu, ownStateSize);
        else
            backend->ComputeError(e, z, layerBelow->mu, ownStateSize);

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

        if (isClamped && muCacheValid)
        {
            backend->Copy(mu, cachedMu, Nout);
            backend->Copy(preact, cachedPreact, Nout);
            return;
        }

        // Tries a single fused cuDNN convolution straight into mu first,
        // falling back to the original Im2Col()+RepackForBatchedGemm()+
        // MatMul() composition otherwise -- same pattern as
        // FullConvPCLayer::ComputeMuOnly(), see its own comment for why.
        if (!backend->TryFusedConvForward(z, W, mu, 1.0f, batchSize, inChannels, inHeight, inWidth,
                                          outChannels, kernelH, kernelW, strideH, strideW, padH, padW))
        {
            backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth,
                            kernelH, kernelW, strideH, strideW, padH, padW,
                            colBuffer);

            backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
            backend->MatMul(
                /*transA=*/false, /*transB=*/false,
                outChannels, (int)(batchSize * colCols), (int)colRows,
                1.0f, W, (int)colRows, colsRepacked, (int)(batchSize * colCols),
                0.0f, lgRepacked, (int)(batchSize * colCols));
            backend->RepackForBatchedGemm(mu, lgRepacked, outChannels, batchSize, colCols);
        }

        // mu is [batchSize, outChannels, colCols] (batch outermost) --
        // AddBiasPerChannel() needs channel outermost instead (wrong
        // here for any batchSize > 1, see its own doc comment), so this
        // needs AddBiasPerChannelNCHW() specifically.
        backend->AddBiasPerChannelNCHW(mu, b, batchSize, outChannels, colCols);

        // Snapshotted BEFORE Activation() overwrites mu in place: GELU's
        // derivative has no closed form from the activated value alone
        // (see preact's own doc comment), so every later derivative
        // computation reads this instead of mu.
        backend->Copy(preact, mu, Nout);
        backend->Activation(activationType, mu, Nout);

        if (isClamped)
        {
            backend->Copy(cachedMu, mu, Nout);
            backend->Copy(cachedPreact, preact, Nout);
            muCacheValid = true;
        }
    }

    void SimpleConvPCLayer::UpdateState() noexcept
    {
        size_t ownSize = (size_t)inChannels * inHeight * inWidth;
        size_t ownStateSize = (size_t)batchSize * ownSize;
        size_t colCols = (outChannels > 0) ? (size_t)outHeight * outWidth : 0;
        size_t colRows = (outChannels > 0) ? (size_t)inChannels * kernelH * kernelW : 0;

        // mu must end up holding the derivative whenever outChannels > 0,
        // regardless of isClamped or layerAbove, UpdateWeights() reads it
        // unconditionally later with no isClamped guard of its own. Fuse
        // the derivative with the immediately-following multiply only in
        // the case that actually reaches that multiply (connected,
        // unclamped); fall back to the plain derivative-only call
        // otherwise so the invariant still holds.
        bool canFuse = (layerAbove != nullptr && outChannels > 0 && !isClamped);

        if (outChannels > 0 && !canFuse)
        {
            size_t outTotal = (size_t)batchSize * outChannels * colCols;
            backend->ActivationDerivativeInto(derivativeType, mu, preact, outTotal);
        }

        if (isClamped)
            return;

        backend->Zero(dz_dt, ownStateSize);

        if (layerAbove != nullptr && outChannels > 0)
        {
            const float *e_above = layerAbove->GetErrors();
            size_t outSize = (size_t)outChannels * colCols;
            size_t outTotal = (size_t)batchSize * outSize;

            // Un-fused (was FusedActivationDerivativeMultiply(..., mu,
            // ...)): that single-call fusion assumed mu already held the
            // ACTIVATED value it could derive from in place, which GELU
            // has no closed form for (see preact's own doc comment) -- so
            // the derivative now reads preact explicitly, in its own step.
            backend->ActivationDerivativeInto(derivativeType, mu, preact, outTotal);
            backend->MultiplyInto(bottom_up_cols, e_above, mu, outTotal);

            // Tries a single fused cuDNN backward-data convolution
            // straight into dz_dt (already Zero()d above, so this
            // overwrite contract matches Col2Im()'s accumulate one
            // here); same pattern as FullConvPCLayer::UpdateState().
            if (!backend->TryFusedConvBackwardData(bottom_up_cols, W, dz_dt, 1.0f, batchSize,
                                                   inChannels, inHeight, inWidth, outChannels,
                                                   kernelH, kernelW, strideH, strideW, padH, padW))
            {
                // Same batched-GEMM trick as ComputeMuOnly(): one big transA
                // GEMM instead of batchSize small ones, un-repacking the
                // result back to per-batch-contiguous layout Col2Im() expects.
                backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);
                backend->MatMul(
                    /*transA=*/true, /*transB=*/false,
                    (int)colRows, (int)(batchSize * colCols), outChannels,
                    1.0f, W, (int)colRows, lgRepacked, (int)(batchSize * colCols),
                    0.0f, colsRepacked, (int)(batchSize * colCols));
                backend->RepackForBatchedGemm(feedbackScratch, colsRepacked, colRows, batchSize, colCols);

                backend->Col2Im(feedbackScratch, batchSize, inChannels, inHeight, inWidth,
                                kernelH, kernelW, strideH, strideW, padH, padW,
                                dz_dt);
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

        // lgRepacked feeds the bias-gradient GEMM below regardless of
        // which weight-gradient path runs; colBuffer/colsRepacked are
        // only needed by the im2col+GEMM FALLBACK path inside each
        // branch below, not by TryFusedConvBackwardFilter() (which reads
        // z directly) -- same reasoning as FullConvPCLayer::UpdateWeights().
        backend->RepackForBatchedGemm(lgRepacked, bottom_up_cols, batchSize, outChannels, colCols);

        size_t M = (size_t)batchSize * colCols;

        switch (opt)
        {
        case OptimizerType::SGD:
        {
            if (lmbda > 0.0f)
                backend->Scale(W, Wsize, 1.0f - lmbda);

            float lr_batch = lr / batchSize;

            // Fuses the weight-gradient GEMM AND its lr_batch-scaled
            // accumulate into W (beta=1.0) in one cuDNN call, straight
            // from z -- no im2col needed. Falls back to the original
            // Im2Col()+RepackForBatchedGemm()+MatMul() composition
            // otherwise.
            if (!backend->TryFusedConvBackwardFilter(z, bottom_up_cols, W, lr_batch, 1.0f,
                                                     batchSize, inChannels, inHeight, inWidth,
                                                     outChannels, kernelH, kernelW, strideH,
                                                     strideW, padH, padW))
            {
                backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth,
                                kernelH, kernelW, strideH, strideW, padH, padW,
                                colBuffer);
                backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
                backend->MatMul(
                    /*transA=*/false, /*transB=*/true,
                    outChannels, (int)colRows, (int)M,
                    lr_batch, lgRepacked, (int)M, colsRepacked, (int)M,
                    1.0f, W, (int)colRows);
            }

            // Bias gradient: db[oc] = lr_batch * sum over lgRepacked's
            // oc-th row (M contiguous values, both batch AND spatial
            // combined). NOT the same reduction shape as SumRows (which
            // reduces across batch only), expressed instead as a GEMM
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

            // Same fused-first, im2col-fallback pattern as the SGD
            // branch above, writing a fresh gradient (beta=0.0) into
            // grad_W instead of accumulating into W directly.
            if (!backend->TryFusedConvBackwardFilter(z, bottom_up_cols, grad_W, grad_scale, 0.0f,
                                                     batchSize, inChannels, inHeight, inWidth,
                                                     outChannels, kernelH, kernelW, strideH,
                                                     strideW, padH, padW))
            {
                backend->Im2Col(z, batchSize, inChannels, inHeight, inWidth,
                                kernelH, kernelW, strideH, strideW, padH, padW,
                                colBuffer);
                backend->RepackForBatchedGemm(colsRepacked, colBuffer, batchSize, colRows, colCols);
                backend->MatMul(
                    /*transA=*/false, /*transB=*/true,
                    outChannels, (int)colRows, (int)M,
                    grad_scale, lgRepacked, (int)M, colsRepacked, (int)M,
                    0.0f, grad_W, (int)colRows);
            }

            // grad_b = grad_scale * (lgRepacked @ ones), beta=0.0f
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

    void SimpleConvPCLayer::PrepareConvWorkspace() noexcept
    {
        if (outChannels == 0)
            return;
        backend->PrepareConvWorkspace(batchSize, inChannels, inHeight, inWidth, outChannels,
                                      kernelH, kernelW, strideH, strideW, padH, padW);
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