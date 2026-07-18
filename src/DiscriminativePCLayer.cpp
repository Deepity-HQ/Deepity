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
                                                 float learningRate, float inferenceRate, float lmbda,
                                                 void (*act)(float *, size_t),
                                                 void (*dAct)(float *, size_t))
        : batchSize(batchSize), lr(learningRate), ir(inferenceRate), lmbda(lmbda), isClamped(false),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct)
    {
        this->size = size;
        this->nextSize = nextSize;

        size_t allocOwn = (size_t)batchSize * size * sizeof(float);     // z, e, dz_dt
        size_t allocOut = (size_t)batchSize * nextSize * sizeof(float); // mu, sigma_prime, scratch
        size_t allocBias = nextSize * sizeof(float);                    // b

        z = (float *)std::aligned_alloc(64, ALIGN64(allocOwn));
        e = (float *)std::aligned_alloc(64, ALIGN64(allocOwn));
        dz_dt = (float *)std::aligned_alloc(64, ALIGN64(allocOwn));
        std::memset(z, 0, ALIGN64(allocOwn));
        std::memset(e, 0, ALIGN64(allocOwn));
        std::memset(dz_dt, 0, ALIGN64(allocOwn));

        if (nextSize > 0)
        {
            W = (float *)std::aligned_alloc(64, ALIGN64((size_t)nextSize * size * sizeof(float)));
            b = (float *)std::aligned_alloc(64, ALIGN64(allocBias));
            mu = (float *)std::aligned_alloc(64, ALIGN64(allocOut));
            sigma_prime = (float *)std::aligned_alloc(64, ALIGN64(allocOut));
            bottom_up = (float *)std::aligned_alloc(64, ALIGN64(allocOut));

            std::memset(b, 0, ALIGN64(allocBias));
            std::memset(mu, 0, ALIGN64(allocOut));
            std::memset(sigma_prime, 0, ALIGN64(allocOut));
            std::memset(bottom_up, 0, ALIGN64(allocOut));
            // W left uninitialized -- RandomizeWeights() must run first.
        }
        else
        {
            W = nullptr;
            b = nullptr;
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
            std::free(b);
            std::free(mu);
            std::free(sigma_prime);
            std::free(bottom_up);
        }
    }

    // Copy constructor
    DiscriminativePCLayer::DiscriminativePCLayer(const DiscriminativePCLayer &other)
        : batchSize(other.batchSize), lr(other.lr), ir(other.ir), lmbda(other.lmbda), isClamped(other.isClamped),
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
            size_t bSize = ALIGN64(nextSize * sizeof(float));
            size_t allocOut = ALIGN64((size_t)batchSize * nextSize * sizeof(float));

            W = (float *)std::aligned_alloc(64, wSize);
            b = (float *)std::aligned_alloc(64, bSize);
            mu = (float *)std::aligned_alloc(64, allocOut);
            sigma_prime = (float *)std::aligned_alloc(64, allocOut);
            bottom_up = (float *)std::aligned_alloc(64, allocOut);

            memcpy(W, other.W, wSize);
            memcpy(b, other.b, bSize);
            memcpy(mu, other.mu, allocOut);
            memcpy(sigma_prime, other.sigma_prime, allocOut);
            memcpy(bottom_up, other.bottom_up, allocOut);
        }
        else
        {
            W = nullptr;
            b = nullptr;
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
            std::free(b);
            std::free(mu);
            std::free(sigma_prime);
            std::free(bottom_up);
        }

        lr = other.lr;
        ir = other.ir;
        lmbda = other.lmbda;
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
            size_t bSize = ALIGN64(nextSize * sizeof(float));
            size_t allocOut = ALIGN64((size_t)batchSize * nextSize * sizeof(float));

            W = (float *)std::aligned_alloc(64, wSize);
            b = (float *)std::aligned_alloc(64, bSize);
            mu = (float *)std::aligned_alloc(64, allocOut);
            sigma_prime = (float *)std::aligned_alloc(64, allocOut);
            bottom_up = (float *)std::aligned_alloc(64, allocOut);

            memcpy(W, other.W, wSize);
            memcpy(b, other.b, bSize);
            memcpy(mu, other.mu, allocOut);
            memcpy(sigma_prime, other.sigma_prime, allocOut);
            memcpy(bottom_up, other.bottom_up, allocOut);
        }
        else
        {
            W = nullptr;
            b = nullptr;
            mu = nullptr;
            sigma_prime = nullptr;
            bottom_up = nullptr;
        }

        return *this;
    }

    DiscriminativePCLayer::DiscriminativePCLayer(DiscriminativePCLayer &&other) noexcept
        : W(other.W),
          b(other.b),
          e(other.e),
          z(other.z),
          batchSize(other.batchSize),
          mu(other.mu),
          sigma_prime(other.sigma_prime),
          dz_dt(other.dz_dt),
          bottom_up(other.bottom_up),
          lr(other.lr),
          ir(other.ir),
          lmbda(other.lmbda),
          isClamped(other.isClamped),
          layerAbove(nullptr),
          layerBelow(nullptr),
          activation(other.activation),
          activationDerivative(other.activationDerivative)
    {
        this->size = other.size;
        this->nextSize = other.nextSize;

        other.W = nullptr;
        other.b = nullptr;
        other.e = nullptr;
        other.z = nullptr;
        other.mu = nullptr;
        other.sigma_prime = nullptr;
        other.dz_dt = nullptr;
        other.bottom_up = nullptr;
    }

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
            std::free(b);
            std::free(mu);
            std::free(sigma_prime);
            std::free(bottom_up);
        }

        lr = other.lr;
        ir = other.ir;
        lmbda = other.lmbda;
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
        b = other.b;
        mu = other.mu;
        sigma_prime = other.sigma_prime;
        dz_dt = other.dz_dt;
        bottom_up = other.bottom_up;

        other.z = other.e = other.W = other.b = other.mu = nullptr;
        other.sigma_prime = other.dz_dt = other.bottom_up = nullptr;

        return *this;
    }

    void DiscriminativePCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = (size_t)size * nextSize;
        float limit = std::sqrt(6.0f / (size + nextSize));

        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(seedGenerator);

#pragma omp parallel
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
            std::normal_distribution<float> dist(0.0f, 1.0f);

#pragma omp for
            for (size_t i = 0; i < Wsz; ++i)
                W[i] = dist(rng);
        }

        // Biases are kept at zero initially, so we don't randomize them here.
    }

    float DiscriminativePCLayer::CalculateState() noexcept
    {
        size_t N = (size_t)batchSize * size;

        if (layerBelow == nullptr)
        {
            std::memset(e, 0, N * sizeof(float));
        }
        else
        {
            cblas_scopy(N, z, 1, e, 1);
            cblas_saxpy(N, -1.0f, layerBelow->mu, 1, e, 1);
        }

        float totalEnergy = 0.5f * cblas_sdot(N, e, 1, e, 1);

        if (nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;

            // pre(batch,nextSize) = z(batch,size) @ W^T(size,nextSize)
            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                batchSize, nextSize, size,
                1.0f,
                z, size,
                W, size,
                0.0f, mu, nextSize);

            // Add the bias vector (b) to every row of the batch
            for (int batch = 0; batch < batchSize; batch++)
            {
                cblas_saxpy(nextSize, 1.0f, b, 1, mu + batch * nextSize, 1);
            }

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

        std::memset(dz_dt, 0, N * sizeof(float));
        cblas_saxpy(N, -1.0f, e, 1, dz_dt, 1);

        if (layerAbove != nullptr && nextSize > 0)
        {
            size_t Nout = (size_t)batchSize * nextSize;
            const float *e_above = layerAbove->GetErrors();

#pragma omp simd
            for (size_t i = 0; i < Nout; i++)
                bottom_up[i] = e_above[i] * sigma_prime[i];

            // dz_dt(batch,size) += local_grad(batch,nextSize) @ W(nextSize,size)
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

        // L2 weight decay: W *= (1 - lmbda)
        if (lmbda > 0.0f)
            cblas_sscal((size_t)nextSize * size, 1.0f - lmbda, W, 1);

        // W(nextSize,size) += (lr/batch) * local_grad^T(nextSize,batch) @ z(batch,size)
        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            nextSize, size, batchSize,
            lr / batchSize,
            local_grad, nextSize,
            z, size,
            1.0f, W, size);

        float lr_batch = lr / batchSize;
        for (int batch = 0; batch < batchSize; batch++)
        {
            cblas_saxpy(nextSize, lr_batch, local_grad + batch * nextSize, 1, b, 1);
        }
    }

    void DiscriminativePCLayer::ResetState() noexcept
    {
        size_t N = (size_t)batchSize * size;
        std::memset(z, 0, N * sizeof(float));
    }

    void DiscriminativePCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        for (int b_idx = 0; b_idx < batchSize; b_idx++)
            memcpy(z + b_idx * size, inputData.data(),
                   std::min(inputData.size(), (size_t)size) * sizeof(float));
        isClamped = true;
    }

    void DiscriminativePCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }
}