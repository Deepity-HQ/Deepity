#include "PCLayer.h"
#include <iostream>
#include <chrono>
#include <cblas.h>
#include <omp.h>
#include <algorithm>
#include <cstring>

#define ALIGN64(n) (((n) + 63) & ~63)

namespace Deep
{

    PCLayer::PCLayer(int size, int nextSize, int batchSize,
        float learningRate, float inferenceRate,
                     void (*act)(float *, size_t),
                     void (*dAct)(float *, size_t))
        : lr(learningRate), ir(inferenceRate), isClamped(false), batchSize(batchSize),
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

        std::memset(z, 0, ALIGN64(allocSize));
        std::memset(e, 0, ALIGN64(allocSize));
        std::memset(mu, 0, ALIGN64(allocSize));
        std::memset(sigma_prime, 0, ALIGN64(allocSize));
        std::memset(dz_dt, 0, ALIGN64(allocSize));
        std::memset(bottom_up, 0, ALIGN64(allocSize));
        // W is intentionally left uninitialized here — RandomizeWeights()
        // is expected to be called before first use.
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
        : lr(other.lr), ir(other.ir), isClamped(other.isClamped), batchSize(other.batchSize),
          layerAbove(nullptr), layerBelow(nullptr),
          activation(other.activation), activationDerivative(other.activationDerivative)
    {
        this->size = other.size;
        this->nextSize = other.nextSize;
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
        ir = other.ir;
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
        : lr(other.lr), ir(other.ir), isClamped(other.isClamped), batchSize(other.batchSize),
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

    void PCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        std::uniform_int_distribution<uint32_t> seedDist;
        size_t Wsz = size * nextSize;
        float limit = std::sqrt(6.0f / (size + nextSize)); // Xavier/Glorot Uniform

        std::vector<uint32_t> seeds(omp_get_max_threads());
        for (auto &s : seeds)
            s = seedDist(seedGenerator);
#pragma omp parallel
        {
            std::mt19937 rng(seeds[omp_get_thread_num()]);
std::uniform_real_distribution<float> dist(-limit, limit);

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

        cblas_scopy(N, mu, 1, sigma_prime, 1); // <- Derivative for update
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
        size_t N = size * batchSize;

        if (layerAbove != nullptr && nextSize > 0)
            activationDerivative(sigma_prime, N);
        else
            std::fill_n(sigma_prime, N, 1.0f);

        // If clamped, the state is fixed to the input data; do not update.
        if (isClamped)
            return;

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
            size_t N_below = layerBelow->GetInputSize() * batchSize;

            // layerBelow->sigma_prime already holds act'(pre_below) — layerBelow's
            // UpdateState() ran earlier this sweep (front-to-back iteration).
            float *e_below_scaled = layerBelow->bottom_up; // reuse as scratch

#pragma omp simd
            for (size_t j = 0; j < N_below; j++)
                e_below_scaled[j] = e_below[j] * layerBelow->sigma_prime[j];

            cblas_sgemm(
                CblasRowMajor, CblasNoTrans, CblasNoTrans,
                batchSize, size, layerBelow->GetInputSize(),
                1.0f,
                e_below_scaled, layerBelow->GetInputSize(),
                W_below, size,
                0.0f, bottom_up, size);

#pragma omp simd
            for (size_t i = 0; i < N; i++)
                dz_dt[i] += bottom_up[i]; // derivative already folded in above
        }
        
        // Update latent state
        cblas_saxpy(N, ir, dz_dt, 1, z, 1);
        // for (size_t i = 0; i < size; ++i)
        // {
        //     z[i] += lr * dz_dt[i];
        // }
    }

    void PCLayer::UpdateWeights() noexcept
    { // \Delta W^{(l)} = -\eta e^{(l)} (z^{(l+1)})^T
        if (layerAbove == nullptr || nextSize == 0)
            return;

        size_t N = size * batchSize;
        const float *z_above = layerAbove->GetBeliefs();
        float *e_scaled = bottom_up;

        #pragma omp simd
        for (size_t i=0; i < N; i++) {
            e_scaled[i] = e[i] * sigma_prime[i];
        }

        cblas_sgemm(
            CblasRowMajor, CblasTrans, CblasNoTrans,
            size, nextSize, batchSize,
            lr / batchSize,
            e_scaled, size,
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