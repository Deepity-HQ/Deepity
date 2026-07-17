#include "DiscriminativePCLayer.h"
#include <iostream>
#include <chrono>
#include <cblas.h>
#include <omp.h>
#include <algorithm>
#include <cstring>

#define ALIGN64(n) (((n) + 63) & ~63)

namespace Deep
{
    DiscriminativePCLayer::DiscriminativePCLayer(int size, int nextSize, int batchSize,
                                                 float learningRate, float inferenceRate,
                                                 void (*act)(float *, size_t),
                                                 void (*dAct)(float *, size_t))
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct)
    {
        this->size = size;
        this->nextSize = nextSize;

        size_t allocOwn = (size_t)batchSize * size * sizeof(float);     // z, e, dz_dt
        size_t allocOut = (size_t)batchSize * nextSize * sizeof(float); // mu, sigma_prime, scratch

        z = (float *)std::aligned_alloc(64, ALIGN64(allocOwn));
        e = (float *)std::aligned_alloc(64, ALIGN64(allocOwn));
        dz_dt = (float *)std::aligned_alloc(64, ALIGN64(allocOwn));
        std::memset(z, 0, ALIGN64(allocOwn));
        std::memset(e, 0, ALIGN64(allocOwn));
        std::memset(dz_dt, 0, ALIGN64(allocOwn));

        if (nextSize > 0)
        {
            // W stored (nextSize, size) row-major: maps FROM this layer's
            // own activity TO a prediction of the layer above (standard
            // feedforward/discriminative direction).
            W = (float *)std::aligned_alloc(64, ALIGN64((size_t)nextSize * size * sizeof(float)));
            mu = (float *)std::aligned_alloc(64, ALIGN64(allocOut));
            sigma_prime = (float *)std::aligned_alloc(64, ALIGN64(allocOut));
            // NOTE: 'bottom_up' is reused as generic scratch here (holds
            // e_above * sigma_prime_own) -- no longer literally a
            // "bottom-up" signal, kept only to avoid a header rename.
            bottom_up = (float *)std::aligned_alloc(64, ALIGN64(allocOut));

            std::memset(mu, 0, ALIGN64(allocOut));
            std::memset(sigma_prime, 0, ALIGN64(allocOut));
            std::memset(bottom_up, 0, ALIGN64(allocOut));
            // W left uninitialized -- RandomizeWeights() must run first.
        }
        else
        {
            W = nullptr;
            mu = nullptr;
            sigma_prime = nullptr;
            bottom_up = nullptr;
        }
    }

    DiscriminativePCLayer::~DiscriminativePCLayer()
    {
        std::free(z);
        std::free(e);
        std::free(dz_dt);
        if (nextSize > 0)
        {
            std::free(W);
            std::free(mu);
            std::free(sigma_prime);
            std::free(bottom_up);
        }
    }

    // Copy constructor
    DiscriminativePCLayer::DiscriminativePCLayer(const DiscriminativePCLayer &other)
        : batchSize(other.batchSize), lr(other.lr), ir(other.ir), isClamped(other.isClamped),
          layerAbove(nullptr), layerBelow(nullptr),
          activation(other.activation), activationDerivative(other.activationDerivative)
    {
        this->size = other.size;
        this->nextSize = other.nextSize;

        size_t allocOwn = ALIGN64((size_t)batchSize * size * sizeof(float));
        z = (float *)std::aligned_alloc(64, allocOwn);
        e = (float *)std::aligned_alloc(64, allocOwn);
        dz_dt = (float *)std::aligned_alloc(64, allocOwn);
        memcpy(z, other.z, allocOwn);
        memcpy(e, other.e, allocOwn);
        memcpy(dz_dt, other.dz_dt, allocOwn);

        if (nextSize > 0)
        {
            size_t wSize = ALIGN64((size_t)nextSize * size * sizeof(float));
            size_t allocOut = ALIGN64((size_t)batchSize * nextSize * sizeof(float));

            W = (float *)std::aligned_alloc(64, wSize);
            mu = (float *)std::aligned_alloc(64, allocOut);
            sigma_prime = (float *)std::aligned_alloc(64, allocOut);
            bottom_up = (float *)std::aligned_alloc(64, allocOut);

            memcpy(W, other.W, wSize);
            memcpy(mu, other.mu, allocOut);
            memcpy(sigma_prime, other.sigma_prime, allocOut);
            memcpy(bottom_up, other.bottom_up, allocOut);
        }
        else
        {
            W = nullptr;
            mu = nullptr;
            sigma_prime = nullptr;
            bottom_up = nullptr;
        }
    }

    // Copy assignment
    DiscriminativePCLayer &DiscriminativePCLayer::operator=(const DiscriminativePCLayer &other)
    {
        if (this == &other)
            return *this;

        std::free(z);
        std::free(e);
        std::free(dz_dt);
        if (nextSize > 0)
        {
            std::free(W);
            std::free(mu);
            std::free(sigma_prime);
            std::free(bottom_up);
        }

        lr = other.lr;
        ir = other.ir;
        isClamped = other.isClamped;
        size = other.size;
        nextSize = other.nextSize;
        batchSize = other.batchSize;
        activation = other.activation;
        activationDerivative = other.activationDerivative;
        layerAbove = nullptr;
        layerBelow = nullptr;

        size_t allocOwn = ALIGN64((size_t)batchSize * size * sizeof(float));
        z = (float *)std::aligned_alloc(64, allocOwn);
        e = (float *)std::aligned_alloc(64, allocOwn);
        dz_dt = (float *)std::aligned_alloc(64, allocOwn);
        memcpy(z, other.z, allocOwn);
        memcpy(e, other.e, allocOwn);
        memcpy(dz_dt, other.dz_dt, allocOwn);

        if (nextSize > 0)
        {
            size_t wSize = ALIGN64((size_t)nextSize * size * sizeof(float));
            size_t allocOut = ALIGN64((size_t)batchSize * nextSize * sizeof(float));

            W = (float *)std::aligned_alloc(64, wSize);
            mu = (float *)std::aligned_alloc(64, allocOut);
            sigma_prime = (float *)std::aligned_alloc(64, allocOut);
            bottom_up = (float *)std::aligned_alloc(64, allocOut);

            memcpy(W, other.W, wSize);
            memcpy(mu, other.mu, allocOut);
            memcpy(sigma_prime, other.sigma_prime, allocOut);
            memcpy(bottom_up, other.bottom_up, allocOut);
        }
        else
        {
            W = nullptr;
            mu = nullptr;
            sigma_prime = nullptr;
            bottom_up = nullptr;
        }

        return *this;
    }

    DiscriminativePCLayer::DiscriminativePCLayer(DiscriminativePCLayer &&other) noexcept
        : W(other.W),
          e(other.e),
          z(other.z),
          batchSize(other.batchSize),
          mu(other.mu),
          sigma_prime(other.sigma_prime),
          dz_dt(other.dz_dt),
          bottom_up(other.bottom_up),
          lr(other.lr),
          ir(other.ir),
          isClamped(other.isClamped),
          layerAbove(nullptr),
          layerBelow(nullptr),
          activation(other.activation),
          activationDerivative(other.activationDerivative)
    {
        this->size = other.size;
        this->nextSize = other.nextSize;

        other.W = nullptr;
        other.e = nullptr;
        other.z = nullptr;
        other.mu = nullptr;
        other.sigma_prime = nullptr;
        other.dz_dt = nullptr;
        other.bottom_up = nullptr;
    }

    // Move assignment (pure pointer transfer -- unaffected by the direction change)
    DiscriminativePCLayer &DiscriminativePCLayer::operator=(DiscriminativePCLayer &&other)
    {
        if (this == &other)
            return *this;

        std::free(z);
        std::free(e);
        std::free(dz_dt);
        if (nextSize > 0)
        {
            std::free(W);
            std::free(mu);
            std::free(sigma_prime);
            std::free(bottom_up);
        }

        lr = other.lr;
        ir = other.ir;
        isClamped = other.isClamped;
        size = other.size;
        nextSize = other.nextSize;
        batchSize = other.batchSize;
        activation = other.activation;
        activationDerivative = other.activationDerivative;
        layerAbove = nullptr;
        layerBelow = nullptr;

        z = other.z;
        e = other.e;
        W = other.W;
        mu = other.mu;
        sigma_prime = other.sigma_prime;
        dz_dt = other.dz_dt;
        bottom_up = other.bottom_up;

        other.z = other.e = other.W = other.mu = nullptr;
        other.sigma_prime = other.dz_dt = other.bottom_up = nullptr;

        return *this;
    }

    void DiscriminativePCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        // Total element count (size*nextSize) is unchanged by the storage
        // orientation flip; Xavier limit is symmetric in size/nextSize.
        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = (size_t)size * nextSize;
        float limit = std::sqrt(6.0f / (size + nextSize));

        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(seedGenerator);

#pragma omp parallel
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
            std::uniform_real_distribution<float> dist(-limit, limit);

#pragma omp for
            for (size_t i = 0; i < Wsz; ++i)
                W[i] = dist(rng);
        }
    }

    float DiscriminativePCLayer::CalculateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

        // Step 1: this layer's OWN error, from the prediction supplied by
        // the layer BELOW (feedforward direction: below predicts above).
        if (layerBelow == nullptr)
        {
            // Input layer: nothing predicts it. Trivial boundary; in normal
            // usage this layer is always clamped, so this rarely matters.
            cblas_scopy(N, z, 1, e, 1);
        }
        else
        {
            // layerBelow already computed its own outgoing prediction (mu,
            // sized batch x size, since layerBelow->nextSize == this->size)
            // earlier THIS SAME sweep -- PCNetwork::CalculateState() walks
            // layers front-to-back (input first), so layerBelow is always
            // processed before this layer.
            cblas_scopy(N, z, 1, e, 1);
            cblas_saxpy(N, -1.0f, layerBelow->mu, 1, e, 1); // e = z - mu_incoming
        }

        float totalEnergy = 0.5f * cblas_sdot(N, e, 1, e, 1);

        // Step 2: this layer's OWN outgoing prediction, targeting the layer
        // above -- only if this layer owns an outgoing weight.
        if (nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;

            // pre(batch,nextSize) = z(batch,size) @ W^T(size,nextSize)
            // W stored (nextSize, size) row-major.
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                batchSize, nextSize, size,
                1.0f,
                z, size,
                W, size,
                0.0f, mu, nextSize);

            cblas_scopy(Nout, mu, 1, sigma_prime, 1); // stash RAW pre-activation
            activation(mu, Nout);
        }

        return totalEnergy;
    }

    void DiscriminativePCLayer::UpdateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

        if (nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;
            activationDerivative(sigma_prime, Nout); // raw pre-activation -> f'(pre), in place
        }

        if (isClamped)
            return;

        // Pull toward matching the incoming prediction from below.
        std::memset(dz_dt, 0, N * sizeof(float));
        cblas_saxpy(N, -1.0f, e, 1, dz_dt, 1);

        // Feedback from layerAbove -- uses ONLY this layer's own W and
        // sigma_prime, plus layerAbove's already-finalized public error.
        // No cross-layer scratch-buffer dependency needed anymore.
        if (layerAbove != nullptr && nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;
            const float *e_above = layerAbove->GetErrors();

#pragma omp simd
            for (size_t i = 0; i < Nout; i++)
                bottom_up[i] = e_above[i] * sigma_prime[i]; // local_grad, using scratch buffer

            // dz_dt(batch,size) += local_grad(batch,nextSize) @ W(nextSize,size)
            // W used directly, NO transpose (unlike step 2 in CalculateState).
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, nextSize,
                1.0f,
                bottom_up, nextSize,
                W, size,
                1.0f, dz_dt, size); // beta=1: accumulate directly
        }

        cblas_saxpy(N, ir, dz_dt, 1, z, 1);
    }

    void DiscriminativePCLayer::UpdateWeights() noexcept
    {
        if (layerAbove == nullptr || nextSize == 0)
            return;

        size_t Nout = (size_t)batchSize * nextSize;
        const float *e_above = layerAbove->GetErrors();
        float *local_grad = bottom_up; // scratch, same buffer as UpdateState's feedback term

#pragma omp simd
        for (size_t i = 0; i < Nout; i++)
            local_grad[i] = e_above[i] * sigma_prime[i];

        // W(nextSize,size) += (lr/batch) * local_grad^T(nextSize,batch) @ z(batch,size)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            nextSize, size, batchSize,
            lr / batchSize,
            local_grad, nextSize,
            z, size,
            1.0f, W, size);
    }

    void DiscriminativePCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        // Unchanged -- still tiles a single vector across all batch slots
        // (a known, separate pre-existing issue; out of scope here).
        for (int b = 0; b < batchSize; b++)
            memcpy(z + b * size, inputData.data(),
                   std::min(inputData.size(), (size_t)size) * sizeof(float));
        isClamped = true;
    }

    void DiscriminativePCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }
}