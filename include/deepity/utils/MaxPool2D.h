#pragma once

#include <cstddef>
#include <limits>

/**
 * @file MaxPool2D.h
 * @brief Standalone 2D max-pooling forward/backward, single-image contract
 * (see Im2Col.h for the same convention). Backward sends each output
 * position's gradient to the single input position that produced it (the
 * argmax Deep::MaxPool2DForward recorded), the exact adjoint of a hard max,
 * not an approximation -- no pre-activation buffer needed, unlike an
 * activation function's derivative.
 *
 * Layout convention: NCHW, matching Im2Col.h.
 */

namespace Deep
{
    /// @brief Computes a single pooling output dimension (height or
    /// width), no padding: `(inDim - pool) / stride + 1`.
    inline int PoolOutDim(int inDim, int pool, int stride) noexcept
    {
        return (inDim - pool) / stride + 1;
    }

    /// @brief Max-pools a single (channels, height, width) image into
    /// (channels, outH, outW), recording which input position won each
    /// output position's max for MaxPool2DBackward() to use.
    /// @param argmax Output buffer, same shape as @p output, each entry
    /// the flat (row*width+col) offset within its channel's input plane.
    /// @warning Not batch-aware, call once per batch item.
    inline void MaxPool2DForward(const float* input, int channels, int height, int width,
                                 int poolH, int poolW, int strideH, int strideW, float* output,
                                 int* argmax) noexcept
    {
        const int outH = PoolOutDim(height, poolH, strideH);
        const int outW = PoolOutDim(width, poolW, strideW);

        for (int c = 0; c < channels; ++c)
        {
            const float* inPlane = input + (size_t)c * height * width;
            float* outPlane = output + (size_t)c * outH * outW;
            int* argPlane = argmax + (size_t)c * outH * outW;

            for (int oh = 0; oh < outH; ++oh)
            {
                for (int ow = 0; ow < outW; ++ow)
                {
                    float bestVal = -std::numeric_limits<float>::infinity();
                    int bestIdx = 0;

                    for (int ph = 0; ph < poolH; ++ph)
                    {
                        const int ih = oh * strideH + ph;
                        for (int pw = 0; pw < poolW; ++pw)
                        {
                            const int iw = ow * strideW + pw;
                            const int idx = ih * width + iw;
                            const float v = inPlane[idx];
                            if (v > bestVal)
                            {
                                bestVal = v;
                                bestIdx = idx;
                            }
                        }
                    }

                    outPlane[oh * outW + ow] = bestVal;
                    argPlane[oh * outW + ow] = bestIdx;
                }
            }
        }
    }

    /// @brief The adjoint of MaxPool2DForward(): scatters a
    /// (channels, outH, outW) gradient back into a (channels, height,
    /// width) image at the positions @p argmax recorded.
    /// @param inputGrad ACCUMULATED into (does not zero first, matching
    /// Deep::Col2Im's own contract -- overlapping pooling windows, i.e.
    /// stride < pool, can make the same input position the argmax for
    /// more than one output position).
    /// @warning Not batch-aware, call once per batch item.
    inline void MaxPool2DBackward(const float* outputGrad, const int* argmax, int channels,
                                  int height, int width, int outH, int outW,
                                  float* inputGrad) noexcept
    {
        for (int c = 0; c < channels; ++c)
        {
            const float* gradPlane = outputGrad + (size_t)c * outH * outW;
            const int* argPlane = argmax + (size_t)c * outH * outW;
            float* inPlane = inputGrad + (size_t)c * height * width;

            for (int i = 0; i < outH * outW; ++i)
                inPlane[argPlane[i]] += gradPlane[i];
        }
    }
}
