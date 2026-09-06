#include <deepity/backend/CUDABackend.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <curand_kernel.h>

namespace Deep
{
    CUDABackend::CUDABackend()
    {
        cudaStreamCreate(&this->stream);
        cublasCreate(&this->handle);
        cublasSetStream(this->handle, this->stream);
    }

    CUDABackend::~CUDABackend()
    {
        cublasDestroy(this->handle);
        cudaStreamDestroy(this->stream);
    }

    void CUDABackend::BeginGraphCapture() noexcept
    {
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
    }

    void CUDABackend::EndGraphCapture() noexcept
    {
        cudaGraph_t newGraph;
        cudaError_t err = cudaStreamEndCapture(stream, &newGraph);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaStreamEndCapture failed: " << cudaGetErrorString(err) << "\n";
            return;
        }

        if (hasGraph)
        {
            // Re-capturing (e.g. batch size or settling-step count changed)
            // -- destroy the previous instantiated graph first.
            cudaGraphExecDestroy(graphExec);
            cudaGraphDestroy(graph);
        }

        graph = newGraph;
        err = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaGraphInstantiate failed: " << cudaGetErrorString(err) << "\n";
            hasGraph = false;
            return;
        }

        hasGraph = true;
    }

    void CUDABackend::ReplayGraph() noexcept
    {
        if (!hasGraph)
        {
            std::cerr << "ReplayGraph() called before any graph was captured.\n";
            return;
        }
        cudaGraphLaunch(graphExec, stream);
    }

    float *CUDABackend::Allocate(size_t numFloats)
    {
        float *ptr = nullptr;
        if (cudaMalloc(&ptr, numFloats * sizeof(float)) != cudaSuccess)
        {
            std::cerr << "Could not allocate memory to CUDA backend.\n";
            return nullptr;
        }
        return ptr;
    }

    void CUDABackend::Free(float *ptr) noexcept
    {
        if (cudaFree(ptr) != cudaSuccess)
            std::cerr << "Could not free memory from CUDA backend.\n";
    }

    void CUDABackend::Zero(float *ptr, size_t numFloats) noexcept
    {
        cudaMemset(ptr, 0, numFloats * sizeof(float));
    }

    void CUDABackend::Copy(float *dst, const float *src, size_t numFloats) noexcept
    {
        cudaMemcpy(dst, src, numFloats * sizeof(float), cudaMemcpyDefault);
    }

    void CUDABackend::CopyFromHost(float *deviceDst, const float *hostSrc, size_t numFloats) noexcept
    {
        cudaMemcpy(deviceDst, hostSrc, numFloats * sizeof(float), cudaMemcpyHostToDevice);
    }

    void CUDABackend::CopyToHost(float *hostDst, const float *deviceSrc, size_t numFloats) noexcept
    {
        cudaMemcpy(hostDst, deviceSrc, numFloats * sizeof(float), cudaMemcpyDeviceToHost);
    }

    __global__ void normal_generation(curandState *state, float *random_numbers,
                                      size_t n, float mean, float stddev, uint32_t seed)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            curand_init(seed, static_cast<unsigned long long>(i), 0, &state[i]);
            random_numbers[i] = mean + stddev * curand_normal(&state[i]);
        }
    }

    __global__ void uniform_generation(curandState *state, float *random_numbers, size_t n, float min, float range, uint32_t seed)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            curand_init(seed, static_cast<unsigned long long>(i), 0, &state[i]);
            random_numbers[i] = min + curand_uniform(&state[i]) * range;
        }
    }

    void CUDABackend::RandomizeNormal(float *buf, size_t n, float mean, float stddev, uint32_t seed) noexcept
    {
        if (n == 0)
            return;

        curandState *state = nullptr;
        if (cudaMalloc(&state, n * sizeof(curandState)) != cudaSuccess)
            return;

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        normal_generation<<<blocks, BLOCK_SIZE>>>(state, buf, n, mean, stddev, seed);
        cudaFree(state);
    }

    void CUDABackend::RandomizeUniform(float *buf, size_t n, float min, float max, uint32_t seed) noexcept
    {
        if (n == 0)
            return;

        curandState *state = nullptr;
        cudaError_t mallocErr = cudaMalloc(&state, n * sizeof(curandState));
        if (mallocErr != cudaSuccess)
        {
            std::cerr << "curandState cudaMalloc failed for n=" << n
                      << " (" << n * sizeof(curandState) << " bytes): "
                      << cudaGetErrorString(mallocErr) << "\n";
            return;
        }
        std::cerr << "curandState allocated OK: n=" << n << ", ptr=" << (void *)state << "\n";

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        uniform_generation<<<blocks, BLOCK_SIZE>>>(state, buf, n, min, max - min, seed);

        cudaError_t launchErr = cudaGetLastError();
        if (launchErr != cudaSuccess)
            std::cerr << "uniform_generation kernel launch failed: " << cudaGetErrorString(launchErr) << "\n";

        cudaDeviceSynchronize(); // force the kernel to finish and surface any async error
        cudaError_t syncErr = cudaGetLastError();
        if (syncErr != cudaSuccess)
            std::cerr << "uniform_generation kernel execution failed: " << cudaGetErrorString(syncErr) << "\n";

        cudaFree(state);
    }

    void CUDABackend::MatMul(bool transA, bool transB, int M, int N, int K,
                             float alpha, const float *A, int lda,
                             const float *B, int ldb,
                             float beta, float *C, int ldc) noexcept
    {
        // CUDA is *column-major*.
        cublasOperation_t cuTransA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t cuTransB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        if (cublasSgemm(handle, cuTransB, cuTransA,
                        N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc) != CUBLAS_STATUS_SUCCESS)
            std::cerr << "Failed to perform CUDA MatMul.\n";
    }

    void CUDABackend::Scale(float *buf, size_t n, float alpha) noexcept
    {
        if (cublasSscal(handle, n, &alpha, buf, 1) != CUBLAS_STATUS_SUCCESS)
            std::cerr << "Failed to perform CUDA Scale.\n";
    }

    void CUDABackend::AxpyInto(float *y, const float *x, size_t n, float alpha) noexcept
    {
        if (cublasSaxpy(handle, n, &alpha, x, 1, y, 1))
            std::cerr << "Failed to perform CUDA Axpy.\n";
    }

    __global__ void AddBiasBroadcastKernel(float *buf, const float *bias, size_t batchSize, size_t width)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < batchSize * width)
            buf[i] += bias[i % width];
    }

    void CUDABackend::AddBiasBroadcast(float *buf, const float *bias, size_t batchSize, size_t width) noexcept
    {
        constexpr int BLOCK_SIZE = 256;
        size_t total = batchSize * width;
        const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        AddBiasBroadcastKernel<<<blocks, BLOCK_SIZE>>>(buf, bias, batchSize, width);
    }

#pragma region ACTIVATIONS_AND_KERNELS

    __global__ void ReluKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = fmaxf(0.0f, src[i]);
    }

    __global__ void tanhKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float y;
            asm("tanh.approx.f32 %0, %1;" : "=f"(y) : "f"(src[i]));
            dst[i] = y;
        }
    }

    __global__ void sigmoidKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float t;
            asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * src[i]));
            dst[i] = fmaf(0.5f, t, 0.5f);
        }
    }

    __global__ void eSigmoidKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float s = src[i];
            dst[i] = 0.5f * (s / (1.0f + fabsf(s)) + 1.0f);
        }
    }

    __global__ void linearKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = src[i];
    }

    __global__ void dReluKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = (float)(src[i] > 0.0f);
    }

    __global__ void dTanhKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float t;
            asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(src[i]));
            dst[i] = fmaf(-t, t, 1.0f);
        }
    }

    __global__ void dSigmoidKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float t;
            asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(0.5f * src[i]));
            dst[i] = 0.25f * fmaf(-t, t, 1.0f);
        }
    }

    __global__ void dSigmoidActivatedKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float s = src[i];
            dst[i] = fmaf(-s, s, s);
        }
    }

    __global__ void d_eSigmoidKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float a = 1.0f + fabsf(src[i]);
            dst[i] = 0.5f / (a * a);
        }
    }

    __global__ void d_eSigmoidActivatedKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float s = src[i];
            dst[i] = 2.0f * fmaf(-s, s, s);
        }
    }

    __global__ void dLinearKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            dst[i] = 1.0f;
    }

    __global__ void FusedStateUpdateKernel(float *z, const float *feedback, const float *deriv,
                                           const float *e, size_t n, float ir)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            z[i] += ir * ((feedback[i] * deriv[i]) - e[i]);
        }
    }

    __global__ void ComputeErrorKernel(float *e, const float *z, const float *mu, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            e[i] = z[i] - mu[i];
        }
    }

    __global__ void AdamStepKernel(float *param, const float *grad, float *m, float *v,
                                   size_t n, float step_size, float beta1, float beta2, float eps)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);
            param[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
        }
    }

    __global__ void AdamWStepKernel(float *param, const float *grad, float *m, float *v,
                                    size_t n, float lr, float weightDecay,
                                    float step_size, float beta1, float beta2, float eps)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float g = grad[i];
            m[i] = beta1 * m[i] + (1.0f - beta1) * g;
            v[i] = beta2 * v[i] + (1.0f - beta2) * (g * g);

            param[i] -= lr * weightDecay * param[i];
            param[i] -= step_size * m[i] / (sqrtf(v[i]) + eps);
        }
    }

#pragma endregion

    void CUDABackend::Activation(ActivationType type, float *buf, size_t n) noexcept
    {
        // For in-place, pass buf as both dst and src
        ActivationInto(type, buf, buf, n);
    }

    void CUDABackend::ActivationInto(ActivationType type, float *dst, const float *src, size_t n) noexcept
    {
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        switch (type)
        {
        case ActivationType::RELU:
            ReluKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::SIGMOID:
            sigmoidKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::eSIGMOID:
            eSigmoidKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::TANH:
            tanhKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::LINEAR:
            linearKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::NONE:
        default:
            break;
        }
    }

    void CUDABackend::ActivationDerivative(ActivationType type, float *buf, size_t n, bool activated) noexcept
    {
        ActivationDerivativeInto(type, buf, buf, n);
    }

    void CUDABackend::ActivationDerivativeInto(ActivationType type, float *dst, const float *src, size_t n) noexcept
    {
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        switch (type)
        {
        case ActivationType::dRELU:
            dReluKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::dSIGMOID:
            dSigmoidKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::d_eSIGMOID:
            d_eSigmoidKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::dTANH:
            dTanhKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::dLINEAR:
            dLinearKernelInto<<<blocks, BLOCK_SIZE>>>(dst, src, n);
            break;
        case ActivationType::NONE:
        default:
            break;
        }
    }

    void CUDABackend::FusedStateUpdate(float *z, const float *feedback, const float *deriv,
                                       const float *e, size_t n, float ir) noexcept
    {
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        FusedStateUpdateKernel<<<blocks, BLOCK_SIZE>>>(z, feedback, deriv, e, n, ir);
    }

    float CUDABackend::ComputeErrorAndEnergy(float *e, const float *z, const float *mu, size_t n) noexcept
    {
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        ComputeErrorKernel<<<blocks, BLOCK_SIZE>>>(e, z, mu, n);

        float sum_of_squares = 0.0f;
        cublasSdot(handle, n, e, 1, e, 1, &sum_of_squares);

        return 0.5f * sum_of_squares;
    }

    void CUDABackend::ComputeError(float *e, const float *z, const float *mu, size_t n) noexcept
    {
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        ComputeErrorKernel<<<blocks, BLOCK_SIZE>>>(e, z, mu, n);
    }

    void CUDABackend::AdamStep(float *param, const float *grad, float *m, float *v,
                               size_t n, int t, float lr,
                               float beta1, float beta2, float eps) noexcept
    {
        // Compute scalars on host CPU
        float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(t));
        float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(t));
        float step_size = lr * Sleef_sqrtf(beta2_t) / beta1_t;

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        AdamStepKernel<<<blocks, BLOCK_SIZE>>>(param, grad, m, v, n, step_size, beta1, beta2, eps);
    }

    void CUDABackend::AdamWStep(float *param, const float *grad, float *m, float *v,
                                size_t n, int t, float lr, float weightDecay,
                                float beta1, float beta2, float eps) noexcept
    {
        float beta1_t = 1.0f - Sleef_powf_u10(beta1, static_cast<float>(t));
        float beta2_t = 1.0f - Sleef_powf_u10(beta2, static_cast<float>(t));
        float step_size = lr * Sleef_sqrtf(beta2_t) / beta1_t;

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        AdamWStepKernel<<<blocks, BLOCK_SIZE>>>(param, grad, m, v, n, lr, weightDecay, step_size, beta1, beta2, eps);
    }
}