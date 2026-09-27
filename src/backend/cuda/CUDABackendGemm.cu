/**
 * @file CUDABackendGemm.cu
 * @brief CUDABackend's matrix-multiply paths: the plain cuBLAS MatMul/
 * SumRows pair, and the CUTLASS-fused GEMM+bias+activation path used by
 * TryFusedForwardPass. Mirrors the "GEMM" section of IComputeBackend.h (plus
 * the GEMM-flavored half of "Fused PC-specific ops"). Split out of the
 * former monolithic CUDABackend.cu -- see CUDABackendCore.cu,
 * CUDABackendElementwise.cu, CUDABackendActivations.cu,
 * CUDABackendFusedOps.cu, CUDABackendOptimizer.cu, CUDABackendConv.cu for
 * the rest.
 */
#include <deepity/backend/CUDABackend.h>

#ifdef DEEPITY_USE_CUDA
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
#include <cutlass/gemm/device/gemm.h>

namespace Deep
{
using GemmFwdRelu = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationRelu<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdLinear = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 128 / cutlass::sizeof_bits<float>::value,
                                                 float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdReluScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationRelu<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdLinearScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

void CUDABackend::MatMul(bool transA, bool transB, int M, int N, int K, float alpha, const float* A,
                         int lda, const float* B, int ldb, float beta, float* C, int ldc) noexcept
{
  if (!A || !B || !C)
    return;

  cublasOperation_t opA = transA ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t opB = transB ? CUBLAS_OP_T : CUBLAS_OP_N;

  cublasSgemm(this->handle, opB, opA, N, M, K, &alpha, B, ldb, A, lda, &beta, C, ldc);
}

void CUDABackend::SumRows(float* dst, const float* src, size_t batchSize, size_t width) noexcept
{
  if (!dst || !src)
    return;

  if (!onesVector || onesCapacity < batchSize)
    PrepareForBatchSize(batchSize);

  float alpha = 1.0f, beta = 0.0f;
  cublasSgemv(
      handle, CUBLAS_OP_N, width, batchSize, &alpha, src, width, onesVector, 1, &beta, dst, 1);
}

bool CUDABackend::TryFusedForwardPass(ActivationType actType, const float* zF, const float* W,
                                      const float* bias, float* mu, int batchSize, int size,
                                      int nextSize) noexcept
{
  int split_k_slices = 1;
  bool wide = (nextSize % 4 == 0);

  if (actType == ActivationType::RELU)
  {
    if (wide)
    {
      GemmFwdRelu gemm_op;
      GemmFwdRelu::Arguments args({batchSize, nextSize, size},
                                  {zF, size},
                                  {W, size},
                                  {bias, 0},
                                  {mu, nextSize},
                                  {1.0f, 1.0f},
                                  split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdReluScalar gemm_op;
    GemmFwdReluScalar::Arguments args({batchSize, nextSize, size},
                                      {zF, size},
                                      {W, size},
                                      {bias, 0},
                                      {mu, nextSize},
                                      {1.0f, 1.0f},
                                      split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }
  else if (actType == ActivationType::LINEAR)
  {
    if (wide)
    {
      GemmFwdLinear gemm_op;
      GemmFwdLinear::Arguments args({batchSize, nextSize, size},
                                    {zF, size},
                                    {W, size},
                                    {bias, 0},
                                    {mu, nextSize},
                                    {1.0f, 1.0f},
                                    split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdLinearScalar gemm_op;
    GemmFwdLinearScalar::Arguments args({batchSize, nextSize, size},
                                        {zF, size},
                                        {W, size},
                                        {bias, 0},
                                        {mu, nextSize},
                                        {1.0f, 1.0f},
                                        split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }

  return false;
}
} // namespace Deep

#endif
