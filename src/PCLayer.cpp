#include "PCLayer.h"
#include <iostream>
#include <chrono>
#include <cblas.h>
#include <omp.h>
#include <cstring>

#define ALIGN64(n) (((n) + 63) & ~63)

namespace Deep
{

    PCLayer::PCLayer(int size, int nextSize, int batchSize, float learningRate,
                 void (*act)(float *, size_t),
                 void (*dAct)(float *, size_t))
        : lr(learningRate), isClamped(false), batchSize(batchSize),
          layerAbove(nullptr), layerBelow(nullptr), activation(act), activationDerivative(dAct)
    {
        this->size = size;
        this->nextSize = nextSize;
        size_t allocSize = (size_t)(batchSize * size * sizeof(float));
        z = (float *)(std::aligned_alloc(64, ALIGN64(allocSize)));
        e = (float *)(std::aligned_alloc(64, ALIGN64(allocSize)));
        W = (nextSize > 0) ? (float *)(std::aligned_alloc(64, ALIGN64(size * nextSize * sizeof(float)))) : nullptr;

        mu = (float *)(std::aligned_alloc(64, ALIGN64(allocSize)));
        sigma_prime = (float *)(std::aligned_alloc(64, ALIGN64(allocSize)));
        dz_dt = (float *)(std::aligned_alloc(64, ALIGN64(allocSize)));
        bottom_up = (float *)(std::aligned_alloc(64, ALIGN64(allocSize)));
    }

    PCLayer::~PCLayer()
    {
        std::free(z);
        std::free(e);
        if (nextSize > 0)
            std::free(W);

        std::free(mu);
        std::free(sigma_prime);
        std::free(dz_dt);
        std::free(bottom_up);
    }

    // Copy constructor
    PCLayer::PCLayer(const PCLayer &other)
        : lr(other.lr), isClamped(other.isClamped), batchSize(other.batchSize),
          layerAbove(nullptr), layerBelow(nullptr),
          activation(other.activation), activationDerivative(other.activationDerivative)
    {
        this->size = size;
        this->nextSize = nextSize;
        size_t allocSize = ALIGN64(batchSize * size * sizeof(float));
        z = (float *)std::aligned_alloc(64, allocSize);
        e = (float *)std::aligned_alloc(64, allocSize);
        mu = (float *)std::aligned_alloc(64, allocSize);
        sigma_prime = (float *)std::aligned_alloc(64, allocSize);
        dz_dt = (float *)std::aligned_alloc(64, allocSize);
        bottom_up = (float *)std::aligned_alloc(64, allocSize);

        memcpy(z, other.z, allocSize);
        memcpy(e, other.e, allocSize);
        memcpy(mu, other.mu, allocSize);
        memcpy(sigma_prime, other.sigma_prime, allocSize);
        memcpy(dz_dt, other.dz_dt, allocSize);
        memcpy(bottom_up, other.bottom_up, allocSize);

        if (nextSize > 0)
        {
            size_t wSize = ALIGN64(size * nextSize * sizeof(float));
            W = (float *)std::aligned_alloc(64, wSize);
            memcpy(W, other.W, wSize);
        }
        else
        {
            W = nullptr;
        }
    }

    // Copy assignment
    PCLayer &PCLayer::operator=(const PCLayer &other)
    {
        if (this == &other)
            return *this;

        // Free existing
        std::free(z);
        std::free(e);
        std::free(mu);
        std::free(sigma_prime);
        std::free(dz_dt);
        std::free(bottom_up);
        if (nextSize > 0)
            std::free(W);

        lr = other.lr;
        isClamped = other.isClamped;
        size = other.size;
        nextSize = other.nextSize;
        batchSize = other.batchSize;
        activation = other.activation;
        activationDerivative = other.activationDerivative;
        layerAbove = nullptr;
        layerBelow = nullptr;

        size_t allocSize = ALIGN64(batchSize * size * sizeof(float));
        z = (float *)std::aligned_alloc(64, allocSize);
        e = (float *)std::aligned_alloc(64, allocSize);
        mu = (float *)std::aligned_alloc(64, allocSize);
        sigma_prime = (float *)std::aligned_alloc(64, allocSize);
        dz_dt = (float *)std::aligned_alloc(64, allocSize);
        bottom_up = (float *)std::aligned_alloc(64, allocSize);

        memcpy(z, other.z, allocSize);
        memcpy(e, other.e, allocSize);
        memcpy(mu, other.mu, allocSize);
        memcpy(sigma_prime, other.sigma_prime, allocSize);
        memcpy(dz_dt, other.dz_dt, allocSize);
        memcpy(bottom_up, other.bottom_up, allocSize);

        if (nextSize > 0)
        {
            size_t wSize = ALIGN64(size * nextSize * sizeof(float));
            W = (float *)std::aligned_alloc(64, wSize);
            memcpy(W, other.W, wSize);
        }
        else
        {
            W = nullptr;
        }

        return *this;
    }

    // Move constructor
    PCLayer::PCLayer(PCLayer &&other)
        : lr(other.lr), isClamped(other.isClamped), batchSize(other.batchSize),
          layerAbove(nullptr), layerBelow(nullptr),
          activation(other.activation), activationDerivative(other.activationDerivative),
          z(other.z), e(other.e), W(other.W), mu(other.mu),
          sigma_prime(other.sigma_prime), dz_dt(other.dz_dt), bottom_up(other.bottom_up)
    {
        this->size = other.size;
        this->nextSize = other.nextSize;
        other.z = other.e = other.W = other.mu = nullptr;
        other.sigma_prime = other.dz_dt = other.bottom_up = nullptr;
    }

    // Move assignment
    PCLayer &PCLayer::operator=(PCLayer &&other)
    {
        if (this == &other)
            return *this;

        std::free(z);
        std::free(e);
        std::free(mu);
        std::free(sigma_prime);
        std::free(dz_dt);
        std::free(bottom_up);
        if (nextSize > 0)
            std::free(W);

        lr = other.lr;
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

    void PCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = size * nextSize;

        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(seedGenerator);

#pragma omp parallel
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
            std::uniform_real_distribution<float> dist(0.0f, 0.1f);

#pragma omp for
            for (size_t i = 0; i < Wsz; ++i)
            {
                W[i] = dist(rng);
            }
        }
    }

    float PCLayer::CalculateState() noexcept
    { // E = \sum_l 1/2 ||z^{(l)} - \mu^{(l)}||^2

        float totalEnergy = 0.0f;
        size_t N = batchSize * size;
        if (layerAbove == nullptr || nextSize == 0)
        { // No input
            cblas_scopy(N,
                        z, 1,
                        e, 1);

            totalEnergy = 0.5f * cblas_sdot(N, e, 1, e, 1);
            return totalEnergy;
        }

        const float *z_above = layerAbove->GetBeliefs();

        cblas_sgemm(
            CblasRowMajor,
            CblasNoTrans, CblasTrans,
            batchSize, size, nextSize, 1.0f,
            z_above, nextSize,
            W, nextSize, 0.0f, mu, size);
        // for (size_t i = 0; i < size; i++)
        // {
        //     mu[i] = 0.0f;
        //     for (size_t j = 0; j < layerAbove->size; j++)
        //     {
        //         mu[i] += W[i * nextSize + j] * z_above[j];
        //     }
        // }
        activation(mu, N);

        // Calculate Error and Energy
        cblas_scopy(N, z, 1, e, 1);
        cblas_saxpy(N, -1.0f, mu, 1, e, 1);
        totalEnergy = 0.5f * cblas_sdot(N, e, 1, e, 1);
        // for (size_t i = 0; i < size; i++)
        // {
        //     e[i] = z[i] - mu[i];
        //     totalEnergy += 0.5f * e[i] * e[i];
        // }
        return totalEnergy;
    }

    void PCLayer::UpdateState() noexcept
    {
        // If clamped, the state is fixed to the input data; do not update.
        if (isClamped)
            return;

        size_t N = size * batchSize;

        // Compute the Pre-Activation Projection (mu = W * z_above)
        // Use to compute the derivative of the activation function.
        if (layerAbove != nullptr && nextSize > 0)
        {
            const float *z_above = layerAbove->GetBeliefs();

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                batchSize, size, nextSize,
                1.0f, z_above, nextSize,
                W, nextSize,
                0.0f, mu, size);
            // for (size_t i = 0; i < size; ++i)
            // {
            //     for (size_t j = 0; j < layerAbove->size; ++j)
            //     {
            //         mu[i] += W[i * nextSize +j] * z_above[j];
            //     }
            // }
        }

        cblas_scopy(N, mu, 1, sigma_prime, 1); // <- Derivative
        activationDerivative(sigma_prime, N);

        // Initialize dz_dt with top-down force (-e)
        memset(dz_dt, 0, N * sizeof(float));
        cblas_saxpy(N, -1.0f, e, 1, dz_dt, 1);
        // for (size_t i = 0; i < size; ++i)
        // {
        //     dz_dt[i] = -e[i];
        // }

        // Add Bottom-Up force: (W^(l-1))^T e^(l-1) * sigma'(mu)
        if (layerBelow != nullptr && layerBelow->nextSize > 0)
        {
            const float *W_below = layerBelow->GetWeights();
            const float *e_below = layerBelow->GetErrors();

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasTrans,
                batchSize, size, layerBelow->nextSize,
                1.0f, e_below, layerBelow->nextSize,
                W_below, layerBelow->nextSize,
                0.0f, bottom_up, size);

            for (size_t i = 0; i < N; i++)
            {
                dz_dt[i] += bottom_up[i] * sigma_prime[i];
            }
        }

        // Update latent state
        cblas_saxpy(N, lr, dz_dt, 1, z, 1);
        // for (size_t i = 0; i < size; ++i)
        // {
        //     z[i] += lr * dz_dt[i];
        // }
    }

    void PCLayer::UpdateWeights() noexcept
    { // \Delta W^{(l)} = -\eta e^{(l)} (z^{(l+1)})^T
        if (layerAbove == nullptr || nextSize == 0)
            return;

        const float *z_above = layerAbove->GetBeliefs();

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            size, nextSize, batchSize,
            -lr,
            e, size,
            z_above, nextSize,
            1.0f, W, nextSize);

        // for (size_t i = 0; i < size; i++)
        // {
        //     float scaled_err = -lr * e[i];

        //     for (size_t j = 0; j < layerAbove->size; j++)
        //     {
        //         W[i * nextSize + j] += scaled_err * z_above[j];
        //     }
        // }
    }

    void PCLayer::ClampState(const std::vector<float> &inputData) noexcept
    {
        // Tile the single input across all batch slots
        for (int b = 0; b < batchSize; b++)
            memcpy(z + b * size, inputData.data(),
                   std::min(inputData.size(), (size_t)size) * sizeof(float));
        isClamped = true;
    }

    void PCLayer::UnclampState() noexcept
    {
        isClamped = false;
    }
}