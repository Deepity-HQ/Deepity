#pragma once

#include <cstddef>
#include <cstring>
#include <immintrin.h>

/**
 * @file Im2Col.h
 * @brief Standalone im2col / col2im utilities: convolution expressed as
 * matrix multiplication, so ConvPCLayer can reuse cblas_sgemm rather than
 * needing hand-written conv kernels.
 *
 * VERIFIED: passed tConvVerify.cpp -- naive-reference comparison across
 * three shape configurations (max error ~1e-6, float32 GEMM accumulation
 * tolerance) AND an exact adjoint check (<Wx,y> == <x,W^Ty>, |diff|=0),
 * confirming Col2Im is the TRUE adjoint of Im2Col, not just a
 * plausible-looking inverse-shaped operation. This matters because
 * ConvPCLayer's UpdateState() feedback term relies on that adjoint
 * property the same way DiscriminativePCLayer's feedback term relies
 * on W^T being the genuine transpose.
 *
 * Layout convention: NCHW. A single batch item's input is
 * (channels, height, width), row-major, contiguous per channel.
 *
 * im2col output shape: (channels*kH*kW, outH*outW) -- each column is one
 * flattened receptive-field patch, ready to be GEMM'd against a
 * (outChannels, channels*kH*kW) weight matrix to produce
 * (outChannels, outH*outW) in one matrix multiply.
 *
 * col2im is the adjoint operation: scatters a
 * (channels*kH*kW, outH*outW) gradient buffer back into a
 * (channels, height, width) image, ACCUMULATING
 * (does not zero the destination first -- caller must memset if needed).
 *
 * Neither function knows anything about batching -- call once per batch
 * item, with pointers offset to that item's slice.
 *
 * @note Each SIMD width (AVX-512/AVX2+AVX/SSE) is handled as a separate
 * preprocessor branch with its own scalar-tail fallback along the
 * stride-1 fast path; the strided (strideW != 1) path is always scalar,
 * since a fixed vector width can't align with an arbitrary stride.
 * @version 1.0
 * @date 2026-06-30
 * @author Jack Rose
 */

namespace Deep
{
    /// @brief Computes a single convolution output dimension (height or
    /// width) from its corresponding input dimension and conv parameters.
    /// @param inDim Input dimension (height or width) before convolution.
    /// @param kernel Kernel size along this dimension.
    /// @param stride Stride along this dimension.
    /// @param pad Zero-padding applied to both sides of this dimension.
    /// @return The resulting output dimension:
    /// `(inDim + 2*pad - kernel) / stride + 1`.
    inline int ConvOutDim(int inDim, int kernel, int stride, int pad) noexcept
    {
        return (inDim + 2 * pad - kernel) / stride + 1;
    }

    /// @brief Rearranges a single (channels, height, width) input image
    /// into a (channels*kH*kW, outH*outW) column matrix, where each
    /// column holds one flattened receptive-field patch -- the standard
    /// im2col transform that lets convolution be computed as a single
    /// GEMM against a (outChannels, channels*kH*kW) weight matrix.
    ///
    /// Positions that fall outside the input (due to padding) are
    /// written as zero.
    ///
    /// @param input Pointer to a single batch item's input image, layout
    /// (channels, height, width), row-major, contiguous per channel.
    /// @param channels Number of input channels.
    /// @param height Input image height.
    /// @param width Input image width.
    /// @param kH Kernel height.
    /// @param kW Kernel width.
    /// @param strideH Vertical stride.
    /// @param strideW Horizontal stride.
    /// @param padH Vertical zero-padding.
    /// @param padW Horizontal zero-padding.
    /// @param columns Output buffer of shape
    /// (channels*kH*kW, outH*outW), fully overwritten (not accumulated
    /// into). Must be preallocated by the caller with size computed from
    /// ConvOutDim().
    /// @warning Not batch-aware -- call once per batch item, with @p input
    /// and @p columns offset to that item's slice.
    inline void Im2Col(const float *input,
                       int channels, int height, int width,
                       int kH, int kW,
                       int strideH, int strideW,
                       int padH, int padW,
                       float *columns) noexcept
    {
        const int outH = ConvOutDim(height, kH, strideH, padH);
        const int outW = ConvOutDim(width, kW, strideW, padW);
        const int colWidth = outH * outW;

        for (int c = 0; c < channels; ++c)
        {
            for (int kh = 0; kh < kH; ++kh)
            {
                for (int kw = 0; kw < kW; ++kw)
                {
                    const int rowIdx = (c * kH + kh) * kW + kw;
                    float *destRow = columns + (size_t)rowIdx * colWidth;

                    for (int oh = 0; oh < outH; ++oh)
                    {
                        const int inRow = oh * strideH - padH + kh;
                        float *dst = destRow + (size_t)oh * outW;

                        if (inRow < 0 || inRow >= height)
                        {
                            std::memset(dst, 0, (size_t)outW * sizeof(float));
                            continue;
                        }

                        const float *src =
                            input + ((size_t)c * height + inRow) * width;

                        if (strideW == 1)
                        {
                            const int first = padW > kw ? padW - kw : 0;
                            const int last = width + padW - kw < outW
                                                 ? width + padW - kw
                                                 : outW;

                            int ow = 0;

                            for (; ow < first; ++ow)
                                dst[ow] = 0.0f;

                            const float *s = src + first - padW + kw;
                            float *d = dst + first;
                            const int count = last - first;
                            int i = 0;

#if defined(__AVX512F__)
                            for (; i + 16 <= count; i += 16)
                                _mm512_storeu_ps(
                                    d + i,
                                    _mm512_loadu_ps(s + i));
#elif defined(__AVX2__) || defined(__AVX__)
                            for (; i + 8 <= count; i += 8)
                                _mm256_storeu_ps(
                                    d + i,
                                    _mm256_loadu_ps(s + i));
#elif defined(__SSE__)
                            for (; i + 4 <= count; i += 4)
                                _mm_storeu_ps(
                                    d + i,
                                    _mm_loadu_ps(s + i));
#endif

                            for (; i < count; ++i)
                                d[i] = s[i];

                            for (ow = last; ow < outW; ++ow)
                                dst[ow] = 0.0f;
                        }
                        else
                        {
                            for (int ow = 0; ow < outW; ++ow)
                            {
                                const int inCol =
                                    ow * strideW - padW + kw;

                                dst[ow] =
                                    (inCol < 0 || inCol >= width)
                                        ? 0.0f
                                        : src[inCol];
                            }
                        }
                    }
                }
            }
        }
    }

    /// @brief The adjoint of Im2Col(): scatters a
    /// (channels*kH*kW, outH*outW) column-gradient buffer back into a
    /// (channels, height, width) image.
    ///
    /// Verified (see file-level note) to be the TRUE mathematical adjoint
    /// of Im2Col(), not just an inverse-shaped operation -- this is what
    /// makes it safe to use for ConvPCLayer's feedback term.
    ///
    /// @param columns Input buffer of shape
    /// (channels*kH*kW, outH*outW), typically a gradient produced by a
    /// GEMM against the same weight matrix used in the forward Im2Col()
    /// pass.
    /// @param channels Number of output-image channels.
    /// @param height Output image height (the original pre-Im2Col
    /// input height).
    /// @param width Output image width (the original pre-Im2Col
    /// input width).
    /// @param kH Kernel height.
    /// @param kW Kernel width.
    /// @param strideH Vertical stride.
    /// @param strideW Horizontal stride.
    /// @param padH Vertical zero-padding.
    /// @param padW Horizontal zero-padding.
    /// @param outputImage Output buffer of shape (channels, height,
    /// width). ACCUMULATED into (values are added, not overwritten) --
    /// the caller must zero this buffer first if a fresh result is
    /// wanted.
    /// @warning Not batch-aware -- call once per batch item, with
    /// @p columns and @p outputImage offset to that item's slice.
    /// @warning Accumulates into @p outputImage rather than overwriting
    /// it; failing to memset the destination first will add this call's
    /// result on top of whatever was already there.
    inline void Col2Im(const float *columns,
                       int channels, int height, int width,
                       int kH, int kW,
                       int strideH, int strideW,
                       int padH, int padW,
                       float *outputImage) noexcept
    {
        const int outH = ConvOutDim(height, kH, strideH, padH);
        const int outW = ConvOutDim(width, kW, strideW, padW);
        const int colWidth = outH * outW;

        for (int c = 0; c < channels; ++c)
        {
            for (int kh = 0; kh < kH; ++kh)
            {
                for (int kw = 0; kw < kW; ++kw)
                {
                    const int rowIdx = (c * kH + kh) * kW + kw;
                    const float *srcRow =
                        columns + (size_t)rowIdx * colWidth;

                    for (int oh = 0; oh < outH; ++oh)
                    {
                        const int inRow = oh * strideH - padH + kh;

                        if (inRow < 0 || inRow >= height)
                            continue;

                        float *dst =
                            outputImage +
                            ((size_t)c * height + inRow) * width;

                        const float *src =
                            srcRow + (size_t)oh * outW;

                        if (strideW == 1)
                        {
                            const int first = padW > kw ? padW - kw : 0;
                            const int last = width + padW - kw < outW
                                                 ? width + padW - kw
                                                 : outW;

                            const float *s = src + first;
                            float *d = dst + first - padW + kw;
                            const int count = last - first;
                            int i = 0;

#if defined(__AVX512F__)
                            for (; i + 16 <= count; i += 16)
                            {
                                const __m512 a = _mm512_loadu_ps(s + i);
                                const __m512 b = _mm512_loadu_ps(d + i);
                                _mm512_storeu_ps(
                                    d + i,
                                    _mm512_add_ps(a, b));
                            }
#elif defined(__AVX2__) || defined(__AVX__)
                            for (; i + 8 <= count; i += 8)
                            {
                                const __m256 a = _mm256_loadu_ps(s + i);
                                const __m256 b = _mm256_loadu_ps(d + i);
                                _mm256_storeu_ps(
                                    d + i,
                                    _mm256_add_ps(a, b));
                            }
#elif defined(__SSE__)
                            for (; i + 4 <= count; i += 4)
                            {
                                const __m128 a = _mm_loadu_ps(s + i);
                                const __m128 b = _mm_loadu_ps(d + i);
                                _mm_storeu_ps(
                                    d + i,
                                    _mm_add_ps(a, b));
                            }
#endif

                            for (; i < count; ++i)
                                d[i] += s[i];
                        }
                        else
                        {
                            for (int ow = 0; ow < outW; ++ow)
                            {
                                const int inCol =
                                    ow * strideW - padW + kw;

                                if (inCol >= 0 && inCol < width)
                                    dst[inCol] += src[ow];
                            }
                        }
                    }
                }
            }
        }
    }
}