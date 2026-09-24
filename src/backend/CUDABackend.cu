#include <deepity/backend/CUDABackend.h>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <deepity/utils/Im2Col.h>

#ifdef DEEPITY_USE_CUDA
#include <curand_kernel.h>

#include <cutlass/gemm/device/gemm.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/epilogue/thread/linear_combination.h>

#define CHECK_CUDA_LAUNCH() \
    do { \
        cudaError_t err = cudaGetLastError(); \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " -> " << cudaGetErrorString(err) << std::endl; \
        } \
    } while (0)

namespace Deep
{
    using GemmFwdRelu = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 16>,
        cutlass::gemm::GemmShape<64, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombinationRelu<
            float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    using GemmFwdLinear = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 16>,
        cutlass::gemm::GemmShape<64, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<
            float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    using GemmFwdReluScalar = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 16>,
        cutlass::gemm::GemmShape<64, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;

    using GemmFwdLinearScalar = cutlass::gemm::device::Gemm<
        float, cutlass::layout::RowMajor,
        float, cutlass::layout::ColumnMajor,
        float, cutlass::layout::RowMajor,
        float,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 16>,
        cutlass::gemm::GemmShape<64, 64, 16>,
        cutlass::gemm::GemmShape<16, 8, 8>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4>;



    CUDABackend::CUDABackend()
    {
        cudaStreamCreate(&this->stream);
        cublasCreate(&this->handle);
        cublasSetStream(this->handle, this->stream);
    }

    CUDABackend::~CUDABackend()
    {
        if (hasGraph)
        {
            cudaGraphExecDestroy(graphExec);
            cudaGraphDestroy(graph);
        }
        if (onesVector)
            cudaFree(onesVector);
        if (workspace)
            cudaFree(workspace);
        cublasDestroy(this->handle);
        cudaStreamDestroy(this->stream);
    }

    void CUDABackend::BeginGraphCapture() noexcept
    {
        cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal);
        if (err != cudaSuccess)
            std::cerr << "cudaStreamBeginCapture failed: " << cudaGetErrorString(err) << "\n";
    }

    bool CUDABackend::EndGraphCapture() noexcept
    {
        cudaGraph_t newGraph;
        cudaError_t err = cudaStreamEndCapture(stream, &newGraph);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaStreamEndCapture failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }

        if (hasGraph)
        {
            cudaGraphExecDestroy(graphExec);
            cudaGraphDestroy(graph);
            hasGraph = false;
        }

        graph = newGraph;
        err = cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
        if (err != cudaSuccess)
        {
            std::cerr << "cudaGraphInstantiate failed: " << cudaGetErrorString(err) << "\n";
            cudaGraphDestroy(graph);
            graph = nullptr;
            return false;
        }

        hasGraph = true;
        return true;
    }

    void CUDABackend::ReplayGraph() noexcept
    {
        if (!hasGraph)
        {
            std::cerr << "ReplayGraph() called before any graph was captured.\n";
            return;
        }
        cudaError_t err = cudaGraphLaunch(graphExec, stream);
        if (err != cudaSuccess)
            std::cerr << "cudaGraphLaunch failed: " << cudaGetErrorString(err) << "\n";
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
        if (ptr)
            cudaFree(ptr);
    }

    void CUDABackend::Zero(float *ptr, size_t numFloats) noexcept
    {
        if (ptr && numFloats > 0)
            cudaMemsetAsync(ptr, 0, numFloats * sizeof(float), stream);
    }

    void CUDABackend::Copy(float *dst, const float *src, size_t numFloats) noexcept
    {
        if (dst && src && numFloats > 0)
            cudaMemcpyAsync(dst, src, numFloats * sizeof(float), cudaMemcpyDefault, stream);
    }

    void CUDABackend::CopyFromHost(float *deviceDst, const float *hostSrc, size_t numFloats) noexcept
    {
        if (deviceDst && hostSrc && numFloats > 0)
            cudaMemcpy(deviceDst, hostSrc, numFloats * sizeof(float), cudaMemcpyHostToDevice);
    }

    void CUDABackend::CopyToHost(float *hostDst, const float *deviceSrc, size_t numFloats) noexcept
    {
        if (hostDst && deviceSrc && numFloats > 0)
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
        if (!buf || n == 0) return;

        curandState *state = nullptr;
        if (cudaMalloc(&state, n * sizeof(curandState)) != cudaSuccess)
            return;

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        normal_generation<<<blocks, BLOCK_SIZE, 0, stream>>>(state, buf, n, mean, stddev, seed);
        CHECK_CUDA_LAUNCH();
        cudaStreamSynchronize(stream);
        cudaFree(state);
    }

    void CUDABackend::RandomizeUniform(float *buf, size_t n, float min, float max, uint32_t seed) noexcept
    {
        if (!buf || n == 0) return;

        curandState *state = nullptr;
        if (cudaMalloc(&state, n * sizeof(curandState)) != cudaSuccess)
            return;

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        uniform_generation<<<blocks, BLOCK_SIZE, 0, stream>>>(state, buf, n, min, max - min, seed);
        CHECK_CUDA_LAUNCH();
        cudaStreamSynchronize(stream);
        cudaFree(state);
    }

    __global__ void FillOnesKernel(float *buf, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
            buf[i] = 1.0f;
    }

    void CUDABackend::PrepareForBatchSize(size_t batchSize) noexcept
    {
        if (onesCapacity < batchSize)
        {
            if (onesVector)
                cudaFree(onesVector);
            cudaMalloc(&onesVector, batchSize * sizeof(float));
            onesCapacity = batchSize;
        }
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
        FillOnesKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(onesVector, batchSize);
        CHECK_CUDA_LAUNCH();
    }

    void CUDABackend::MatMul(bool transA, bool transB, int M, int N, int K,
                            float alpha, const float *A, int lda,
                            const float *B, int ldb,
                            float beta, float *C, int ldc) noexcept
    {
        if (!A || !B || !C) return;

        cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
        cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

        cublasSgemm(this->handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
    }

    void CUDABackend::SumRows(float *dst, const float *src, size_t batchSize, size_t width) noexcept
    {
        if (!dst || !src) return;

        if (!onesVector || onesCapacity < batchSize)
            PrepareForBatchSize(batchSize);

        float alpha = 1.0f, beta = 0.0f;
        cublasSgemv(handle, CUBLAS_OP_N, width, batchSize, &alpha, src, width, onesVector, 1, &beta, dst, 1);
    }

    void CUDABackend::Scale(float *buf, size_t n, float alpha) noexcept
    {
        if (!buf || n == 0) return;
        cublasSscal(handle, n, &alpha, buf, 1);
    }

    void CUDABackend::AxpyInto(float *y, const float *x, size_t n, float alpha) noexcept
    {
        if (!x || !y || n == 0) return;
        cublasSaxpy(handle, n, &alpha, x, 1, y, 1);
    }
    
    __global__ void AddBiasBroadcastKernel(float *buf, const float *bias, size_t batchSize, size_t width)
    {
        size_t col = (size_t)blockIdx.x * blockDim.x + threadIdx.x; // maps to width
        size_t row = (size_t)blockIdx.y * blockDim.y + threadIdx.y; // maps to batchSize

        if (col < width && row < batchSize)
            buf[row * width + col] += bias[col];
    }

    void CUDABackend::AddBiasBroadcast(float *buf, const float *bias, size_t batchSize, size_t width) noexcept
    {
        dim3 threads(32, 8);
        dim3 blocks(
            (unsigned int)((width + threads.x - 1) / threads.x),
            (unsigned int)((batchSize + threads.y - 1) / threads.y));

        AddBiasBroadcastKernel<<<blocks, threads, 0, stream>>>(buf, bias, batchSize, width);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void ReluKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = fmaxf(0.0f, src[i]);
    }

    __global__ void GeluKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float xi = src[i];
            float inner = MAGIC_GELU_1 * xi * (1.0f + MAGIC_GELU_2 * xi * xi);
            float t;
#if __CUDA_ARCH__ >= 800
            asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
#else
            t = tanhf(inner);
#endif
            dst[i] = 0.5f * xi * (1.0f + t);
        }
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
        if (i < n) dst[i] = src[i];
    }

    __global__ void dReluKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = (float)(src[i] > 0.0f);
    }

    constexpr float MAGIC_GELU_2_3 = 3.0f * MAGIC_GELU_2;

    __global__ void dGeluKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float x = src[i];
            float xsq = x * x;
            float inner = MAGIC_GELU_1 * x * (1.0f + MAGIC_GELU_2 * xsq);
            float t;
#if __CUDA_ARCH__ >= 800
            asm("tanh.approx.f32 %0, %1;" : "=f"(t) : "f"(inner));
#else
            t = tanhf(inner);
#endif
            float gprime = MAGIC_GELU_1 * (1.0f + MAGIC_GELU_2_3 * xsq);
            float term1 = 0.5f * (1.0f + t);
            float term2 = 0.5f * x * gprime * (1.0f - t * t);
            dst[i] = term1 + term2;
        }
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

    __global__ void d_eSigmoidKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float a = 1.0f + fabsf(src[i]);
            dst[i] = 0.5f / (a * a);
        }
    }

    __global__ void dLinearKernelInto(float *dst, const float *src, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = 1.0f;
    }

    __global__ void FusedStateUpdateKernel(float *z, const float *feedback, const float *deriv,
                                           const float *e, size_t n, float ir)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            float fb = feedback ? feedback[i] : 0.0f;
            float dev = deriv ? deriv[i] : 1.0f;
            float err = e ? e[i] : 0.0f;
            z[i] += ir * ((fb * dev) - err);
        }
    }

    __global__ void ComputeErrorKernel(float *e, const float *z, const float *mu, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) e[i] = z[i] - mu[i];
    }

    void CUDABackend::Activation(ActivationType type, float *buf, size_t n) noexcept
    {
        ActivationInto(type, buf, buf, n);
    }

    void CUDABackend::ActivationInto(ActivationType type, float *dst, const float *src, size_t n) noexcept
    {
        if (!dst || !src || n == 0) return;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        switch (type)
        {
        case ActivationType::RELU: ReluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::GELU: GeluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::SIGMOID: sigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::eSIGMOID: eSigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::TANH: tanhKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::LINEAR: linearKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::NONE:
        default: break;
        }
        CHECK_CUDA_LAUNCH();
    }

    void CUDABackend::ActivationDerivative(ActivationType type, float *buf, size_t n, bool activated) noexcept
    {
        ActivationDerivativeInto(type, buf, buf, n);
    }

    void CUDABackend::ActivationDerivativeInto(ActivationType type, float *dst, const float *src, size_t n) noexcept
    {
        if (!dst || !src || n == 0) return;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        switch (type)
        {
        case ActivationType::dRELU: dReluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::dGELU: dGeluKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::dSIGMOID: dSigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::d_eSIGMOID: d_eSigmoidKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::dTANH: dTanhKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::dLINEAR: dLinearKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        case ActivationType::NONE:
        default: dLinearKernelInto<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, n); break;
        }
        CHECK_CUDA_LAUNCH();
    }
    
    bool CUDABackend::TryFusedForwardPass(
    ActivationType actType,
    const float *zF, const float *W, const float *bias,
    float *mu, int batchSize, int size, int nextSize) noexcept
{
    int split_k_slices = 1;
    bool wide = (nextSize % 4 == 0);

    if (actType == ActivationType::RELU)
    {
        if (wide)
        {
            GemmFwdRelu gemm_op;
            GemmFwdRelu::Arguments args({batchSize, nextSize, size}, {zF, size}, {W, size},
                                        {bias, 0}, {mu, nextSize}, {1.0f, 1.0f}, split_k_slices);
            return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
        }
        GemmFwdReluScalar gemm_op;
        GemmFwdReluScalar::Arguments args({batchSize, nextSize, size}, {zF, size}, {W, size},
                                          {bias, 0}, {mu, nextSize}, {1.0f, 1.0f}, split_k_slices);
        return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    else if (actType == ActivationType::LINEAR)
    {
        if (wide)
        {
            GemmFwdLinear gemm_op;
            GemmFwdLinear::Arguments args({batchSize, nextSize, size}, {zF, size}, {W, size},
                                          {bias, 0}, {mu, nextSize}, {1.0f, 1.0f}, split_k_slices);
            return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
        }
        GemmFwdLinearScalar gemm_op;
        GemmFwdLinearScalar::Arguments args({batchSize, nextSize, size}, {zF, size}, {W, size},
                                            {bias, 0}, {mu, nextSize}, {1.0f, 1.0f}, split_k_slices);
        return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }

    return false;
}

    void CUDABackend::FusedStateUpdate(float *z, const float *feedback, const float *deriv,
                                       const float *e, size_t n, float ir) noexcept
    {
        if (!z || n == 0) return;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        FusedStateUpdateKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(z, feedback, deriv, e, n, ir);
        CHECK_CUDA_LAUNCH();
    }

    float CUDABackend::ComputeErrorAndEnergy(float *e, const float *z, const float *mu, size_t n) noexcept
    {
        if (!e || !z || !mu || n == 0) return 0.0f;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);

        ComputeErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
        CHECK_CUDA_LAUNCH();

        float sum_of_squares = 0.0f;
        cublasSdot(handle, n, e, 1, e, 1, &sum_of_squares);
        cudaStreamSynchronize(stream);

        return 0.5f * sum_of_squares;
    }

    void CUDABackend::ComputeError(float *e, const float *z, const float *mu, size_t n) noexcept
    {
        if (!e || !z || !mu || n == 0) return;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        ComputeErrorKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(e, z, mu, n);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void IncrementCounterKernel(int *counter) { if (counter) *counter += 1; }
    void CUDABackend::IncrementCounter(int *counter) noexcept
    {
        if (!counter) return;
        IncrementCounterKernel<<<1, 1, 0, stream>>>(counter);
        CHECK_CUDA_LAUNCH();
    }
    
    __global__ void AdamStepKernel(float *param, const float *grad, float *m, float *v,
                                   size_t n, const int *t_ptr, const float *lr_ptr,
                                   float beta1, float beta2, float eps)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            int current_t = *t_ptr;
            if (current_t < 1) current_t = 1;
            float current_lr = *lr_ptr;

            float beta1_t = 1.0f - powf(beta1, static_cast<float>(current_t));
            float beta2_t = 1.0f - powf(beta2, static_cast<float>(current_t));
            float step_size = current_lr * sqrtf(beta2_t) / beta1_t;

            float g = grad[i];
            float m_val = beta1 * m[i] + (1.0f - beta1) * g;
            float v_val = beta2 * v[i] + (1.0f - beta2) * (g * g);

            m[i] = m_val;
            v[i] = v_val;
            param[i] -= step_size * m_val / (sqrtf(v_val) + eps);
        }
    }

    __global__ void AdamWStepKernel(float *param, const float *grad, float *m, float *v,
                                    size_t n, const int *t_ptr, const float *lr_ptr, float weightDecay,
                                    float beta1, float beta2, float eps)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n)
        {
            int current_t = *t_ptr;
            if (current_t < 1) current_t = 1;
            float current_lr = *lr_ptr;

            float beta1_t = 1.0f - powf(beta1, static_cast<float>(current_t));
            float beta2_t = 1.0f - powf(beta2, static_cast<float>(current_t));
            float step_size = current_lr * sqrtf(beta2_t) / beta1_t;

            float g = grad[i];
            float p = param[i];
            float m_val = beta1 * m[i] + (1.0f - beta1) * g;
            float v_val = beta2 * v[i] + (1.0f - beta2) * (g * g);

            m[i] = m_val;
            v[i] = v_val;
            p -= current_lr * weightDecay * p;
            p -= step_size * m_val / (sqrtf(v_val) + eps);
            param[i] = p;
        }
    }

    void CUDABackend::AdamStep(float *param, const float *grad, float *m, float *v,
                               size_t n, const int *t, const float *lr,
                               float beta1, float beta2, float eps) noexcept
    {
        if (!param || !grad || !m || !v || !t || !lr || n == 0) return;

        // NO host copy -- t/lr stay as device pointers, dereferenced
        // inside the kernel itself, so graph capture/replay re-reads the
        // real, current value every time instead of baking in a
        // one-time snapshot from whenever capture happened to run.
        constexpr int blockSize = 256;
        int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);
        AdamStepKernel<<<numBlocks, blockSize, 0, stream>>>(param, grad, m, v, n, t, lr, beta1, beta2, eps);
        CHECK_CUDA_LAUNCH();
    }

    void CUDABackend::AdamWStep(float *param, const float *grad, float *m, float *v,
                                size_t n, const int *t, const float *lr, float weightDecay,
                                float beta1, float beta2, float eps) noexcept
    {
        if (!param || !grad || !m || !v || !t || !lr || n == 0) return;

        constexpr int blockSize = 256;
        int numBlocks = static_cast<int>((n + blockSize - 1) / blockSize);
        AdamWStepKernel<<<numBlocks, blockSize, 0, stream>>>(param, grad, m, v, n, t, lr, weightDecay, beta1, beta2, eps);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void MultiplyIntoKernel(float *dst, const float *a, const float *b, size_t n)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) dst[i] = (a && b) ? (a[i] * b[i]) : 0.0f;
    }

    void CUDABackend::MultiplyInto(float *dst, const float *a, const float *b, size_t n) noexcept
    {
        if (!dst || n == 0) return;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        MultiplyIntoKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, a, b, n);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void FillKernel(float *buf, size_t n, float value)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) buf[i] = value;
    }

    void CUDABackend::Fill(float *buf, size_t n, float value) noexcept
    {
        if (!buf || n == 0) return;
        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE);
        FillKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(buf, n, value);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void AddBiasPerChannelKernel(float *buf, const float *bias, size_t channels, size_t spatialSize)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = channels * spatialSize;
        if (i < total)
        {
            size_t c = i / spatialSize;
            buf[i] += bias ? bias[c] : 0.0f;
        }
    }

    void CUDABackend::AddBiasPerChannel(float *buf, const float *bias, size_t channels, size_t spatialSize) noexcept
    {
        if (!buf || !bias) return;
        constexpr int BLOCK_SIZE = 256;
        size_t total = channels * spatialSize;
        const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        AddBiasPerChannelKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(buf, bias, channels, spatialSize);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void RepackForBatchedGemmKernel(float *dst, const float *src,
                                               size_t batchSize, size_t rows, size_t cols)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        size_t total = rows * batchSize * cols;
        if (i >= total) return;

        size_t row = i / (batchSize * cols);
        size_t rem = i % (batchSize * cols);
        size_t batch = rem / cols;
        size_t col = rem % cols;

        size_t srcIdx = batch * rows * cols + row * cols + col;
        dst[i] = src[srcIdx];
    }

    void CUDABackend::RepackForBatchedGemm(float *dst, const float *src,
                                           size_t batchSize, size_t rows, size_t cols) noexcept
    {
        if (!dst || !src) return;
        constexpr int BLOCK_SIZE = 256;
        size_t total = rows * batchSize * cols;
        const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        RepackForBatchedGemmKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(dst, src, batchSize, rows, cols);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void Im2ColKernel(const float *input, int channels, int height, int width,
                                 int kH, int kW, int strideH, int strideW, int padH, int padW,
                                 int outH, int outW, float *columns)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        size_t colCols = (size_t)outH * outW;
        size_t colRows = (size_t)channels * kH * kW;
        size_t total = colRows * colCols;
        if (i >= total) return;

        size_t row = i / colCols;
        size_t col = i % colCols;

        int c = (int)(row / ((size_t)kH * kW));
        int kh = (int)((row / kW) % kH);
        int kw = (int)(row % kW);

        int oh = (int)(col / outW);
        int ow = (int)(col % outW);

        int inRow = oh * strideH - padH + kh;
        int inCol = ow * strideW - padW + kw;

        if (inRow < 0 || inRow >= height || inCol < 0 || inCol >= width)
            columns[i] = 0.0f;
        else
            columns[i] = input[((size_t)c * height + inRow) * width + inCol];
    }

    void CUDABackend::Im2Col(const float *input, int channels, int height, int width,
                             int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                             float *columns) noexcept
    {
        if (!input || !columns) return;
        int outH = ConvOutDim(height, kernelH, strideH, padH);
        int outW = ConvOutDim(width, kernelW, strideW, padW);
        size_t total = (size_t)channels * kernelH * kernelW * outH * outW;

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        Im2ColKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(input, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, outH, outW, columns);
        CHECK_CUDA_LAUNCH();
    }

    __global__ void Col2ImKernel(const float *columns, int channels, int height, int width,
                                 int kH, int kW, int strideH, int strideW, int padH, int padW,
                                 int outH, int outW, float *outputImage)
    {
        size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
        size_t colCols = (size_t)outH * outW;
        size_t colRows = (size_t)channels * kH * kW;
        size_t total = colRows * colCols;
        if (i >= total) return;

        size_t row = i / colCols;
        size_t col = i % colCols;

        int c = (int)(row / ((size_t)kH * kW));
        int kh = (int)((row / kW) % kH);
        int kw = (int)(row % kW);

        int oh = (int)(col / outW);
        int ow = (int)(col % outW);

        int inRow = oh * strideH - padH + kh;
        int inCol = ow * strideW - padW + kw;

        if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width)
        {
            size_t dstIdx = ((size_t)c * height + inRow) * width + inCol;
            atomicAdd(&outputImage[dstIdx], columns[i]);
        }
    }

    void CUDABackend::Col2Im(const float *columns, int channels, int height, int width,
                             int kernelH, int kernelW, int strideH, int strideW, int padH, int padW,
                             float *outputImage) noexcept
    {
        if (!columns || !outputImage) return;
        int outH = ConvOutDim(height, kernelH, strideH, padH);
        int outW = ConvOutDim(width, kernelW, strideW, padW);
        size_t total = (size_t)channels * kernelH * kernelW * outH * outW;

        constexpr int BLOCK_SIZE = 256;
        const int blocks = static_cast<int>((total + BLOCK_SIZE - 1) / BLOCK_SIZE);
        Col2ImKernel<<<blocks, BLOCK_SIZE, 0, stream>>>(columns, channels, height, width, kernelH, kernelW, strideH, strideW, padH, padW, outH, outW, outputImage);
        CHECK_CUDA_LAUNCH();
    }
}

#endif