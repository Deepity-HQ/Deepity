/**
 * @file CUDABackendGemm.cu
 * @brief CUDABackend's matrix-multiply paths: the plain cuBLAS MatMul/
 * SumRows pair, and the CUTLASS-fused GEMM+bias+activation path used by
 * TryFusedForwardPass. Mirrors the "GEMM" section of IComputeBackend.h (plus
 * the GEMM-flavored half of "Fused PC-specific ops"). Split out of the
 * former monolithic CUDABackend.cu, see CUDABackendCore.cu,
 * CUDABackendElementwise.cu, CUDABackendActivations.cu,
 * CUDABackendFusedOps.cu, CUDABackendOptimizer.cu, CUDABackendConv.cu for
 * the rest.
 *
 * @note TryFusedForwardPass() dispatches RELU/LINEAR/GELU/TANH/SIGMOID
 * to a fused CUTLASS GEMM, but every current caller passes LINEAR
 * unconditionally and applies its real activation as a separate,
 * unfused pass on the GEMM's INPUT beforehand (see
 * IComputeBackend::TryFusedForwardPass's @warning). Actually fusing a
 * layer's activation into this GEMM means restructuring that caller to
 * feed this GEMM the pre-activation belief instead and let the epilogue
 * activate the OUTPUT -- a settling-loop data-flow change, not a new
 * branch here, and unverified end-to-end since this file has no way to
 * be exercised by a caller yet.
 */
#include <deepity/backend/CUDABackend.h>

#ifdef DEEPITY_USE_CUDA
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/epilogue/thread/linear_combination_relu.h>
// Must be defined before activation.h (pulled in by
// linear_combination_sigmoid.h below) is first included anywhere in this
// translation unit: it selects Sigmoid<T>'s tanh-identity formula
// (sigmoid(x) = 0.5*tanh(x/2)+0.5) over the default 1/(1+exp(-x)). This
// engine's own CUDABackendActivations.cu sigmoidKernelInto() uses the
// same tanh-identity (and the same tanh.approx.f32 PTX instruction CUTLASS's
// fast_tanh() uses on sm_75+), so without this, GemmFwdSigmoid below would
// silently compute a numerically different (though mathematically
// equivalent) result than the unfused activation path it's meant to
// replace -- exactly the kind of fused/unfused divergence this project's
// predictive-coding settling loop can't tolerate.
#ifndef CUTLASS_USE_TANH_FOR_SIGMOID
#define CUTLASS_USE_TANH_FOR_SIGMOID
#endif
#include <cutlass/epilogue/thread/linear_combination_sigmoid.h>
#include <cutlass/epilogue/thread/linear_combination_generic.h>
#include <cutlass/epilogue/thread/activation.h>
#include <cutlass/gemm/device/gemm.h>

namespace Deep
{
// GELU_taylor, not CUTLASS's erf-exact GELU: this engine's own GELU
// (Activations.h / CUDABackendActivations.cu's GeluKernelInto) is the
// tanh approximation 0.5*x*(1+tanh(sqrt(2/pi)*x*(1+0.044715*x^2))),
// bit-for-bit CUTLASS's GELU_taylor (same constants, same fast_tanh()).
// Using plain GELU here would fuse a DIFFERENT activation than the one
// every unfused call site actually computes.
using LinearCombinationGELUTaylor = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::GELU_taylor, float, 128 / cutlass::sizeof_bits<float>::value, float,
    float, cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;
using LinearCombinationGELUTaylorScalar = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::GELU_taylor, float, 1, float, float,
    cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;

// cutlass::epilogue::thread::Tanh<float> calls fast_tanh(), the same
// tanh.approx.f32 PTX instruction (sm_75+) this engine's own tanhKernelInto()
// uses -- no dedicated linear_combination_tanh.h ships in CUTLASS, so this
// goes through the generic epilogue directly instead of a ready-made alias.
using LinearCombinationTanh = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::Tanh, float, 128 / cutlass::sizeof_bits<float>::value, float, float,
    cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;
using LinearCombinationTanhScalar = cutlass::epilogue::thread::LinearCombinationGeneric<
    cutlass::epilogue::thread::Tanh, float, 1, float, float,
    cutlass::epilogue::thread::ScaleType::Default, cutlass::FloatRoundStyle::round_to_nearest,
    /*IsHeavy=*/true>;

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

using GemmFwdGELU = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationGELUTaylor,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdGELUScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationGELUTaylorScalar,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdTanh = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationTanh,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdTanhScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>, LinearCombinationTanhScalar,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

// CUTLASS's own ready-made alias: with CUTLASS_USE_TANH_FOR_SIGMOID
// defined above, this is Sigmoid<T>'s tanh-identity branch, matching
// CUDABackendActivations.cu's sigmoidKernelInto() exactly (see the
// #define's comment up top).
using GemmFwdSigmoid = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationSigmoid<
        float, 128 / cutlass::sizeof_bits<float>::value, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 4>;

using GemmFwdSigmoidScalar = cutlass::gemm::device::Gemm<
    float, cutlass::layout::RowMajor, float, cutlass::layout::ColumnMajor, float,
    cutlass::layout::RowMajor, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80,
    cutlass::gemm::GemmShape<128, 128, 16>, cutlass::gemm::GemmShape<64, 64, 16>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombinationSigmoid<float, 1, float, float>,
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

void CUDABackend::SetAllowTF32(bool allow) noexcept
{
  cublasSetMathMode(this->handle, allow ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
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
  else if (actType == ActivationType::GELU)
  {
    if (wide)
    {
      GemmFwdGELU gemm_op;
      GemmFwdGELU::Arguments args({batchSize, nextSize, size},
                                  {zF, size},
                                  {W, size},
                                  {bias, 0},
                                  {mu, nextSize},
                                  {1.0f, 1.0f},
                                  split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdGELUScalar gemm_op;
    GemmFwdGELUScalar::Arguments args({batchSize, nextSize, size},
                                      {zF, size},
                                      {W, size},
                                      {bias, 0},
                                      {mu, nextSize},
                                      {1.0f, 1.0f},
                                      split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }
  else if (actType == ActivationType::TANH)
  {
    if (wide)
    {
      GemmFwdTanh gemm_op;
      GemmFwdTanh::Arguments args({batchSize, nextSize, size},
                                  {zF, size},
                                  {W, size},
                                  {bias, 0},
                                  {mu, nextSize},
                                  {1.0f, 1.0f},
                                  split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdTanhScalar gemm_op;
    GemmFwdTanhScalar::Arguments args({batchSize, nextSize, size},
                                      {zF, size},
                                      {W, size},
                                      {bias, 0},
                                      {mu, nextSize},
                                      {1.0f, 1.0f},
                                      split_k_slices);
    return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
  }
  else if (actType == ActivationType::SIGMOID)
  {
    if (wide)
    {
      GemmFwdSigmoid gemm_op;
      GemmFwdSigmoid::Arguments args({batchSize, nextSize, size},
                                     {zF, size},
                                     {W, size},
                                     {bias, 0},
                                     {mu, nextSize},
                                     {1.0f, 1.0f},
                                     split_k_slices);
      return gemm_op(args, nullptr, stream) == cutlass::Status::kSuccess;
    }
    GemmFwdSigmoidScalar gemm_op;
    GemmFwdSigmoidScalar::Arguments args({batchSize, nextSize, size},
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
