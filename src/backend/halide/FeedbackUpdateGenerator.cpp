#include "Halide.h"

// Fuses SimplePCLayer::UpdateState()'s layerAbove branch into one kernel:
// a GEMM (feedbackScratch = e_above @ W) whose output is consumed directly,
// row by row, by the elementwise state update
//   z += ir * (f'(z) * feedbackScratch - e)
// without ever writing feedbackScratch to memory -- see
// schedule()'s compute_at below. SIGMOID only for now (the only activation
// MNIST exercises); see IComputeBackend.h's TryFusedFeedbackUpdate for the
// scalar-reference formula this must match
// (deepity/utils/Activations.h::ActivationDerivativeScalar, dSIGMOID case).
//
// Buffer layout: all 2D buffers are plain row-major C arrays passed in with
// no padding, dim(0) = the fastest-varying (contiguous, innermost) index --
// Halide's own convention -- which for a row-major (rows, cols) C array
// means dim(0) = the column index, dim(1) = the row index. So:
//   e_above: (batch, nextSize) -> e_above(nextSizeIdx, batchIdx)
//   W:       (nextSize, size)  -> W(sizeIdx, nextSizeIdx)
//   z_in/e/z_out: (batch, size) -> (sizeIdx, batchIdx)
namespace {
class FeedbackUpdateSigmoidGenerator : public Halide::Generator<FeedbackUpdateSigmoidGenerator> {
public:
    Input<Halide::Buffer<float, 2>> e_above{"e_above"};
    Input<Halide::Buffer<float, 2>> W{"W"};
    Input<Halide::Buffer<float, 2>> z_in{"z_in"};
    Input<Halide::Buffer<float, 2>> e{"e"};
    Input<float> ir{"ir"};
    Output<Halide::Buffer<float, 2>> z_out{"z_out"};

    Halide::Var x, y;
    Halide::Func feedback;
    Halide::RDom rk;

    void generate() {
        rk = Halide::RDom(0, e_above.dim(0).extent());

        feedback(x, y) = 0.0f;
        feedback(x, y) += e_above(rk, y) * W(x, rk);

        Halide::Expr zval = z_in(x, y);
        Halide::Expr sig = 1.0f / (1.0f + Halide::exp(-zval));
        Halide::Expr deriv = sig * (1.0f - sig);
        z_out(x, y) = zval + ir * (deriv * feedback(x, y) - e(x, y));
    }

    void schedule() {
        using Halide::TailStrategy::GuardWithIf;

        // Register-tiled schedule, adapted from Halide's own
        // test/performance/matrix_multiplication.cpp benchmark (which hits
        // 80-90% of AVX2/AVX-512 peak): tile the output into small blocks
        // that fit in registers (inner_tile_x columns x inner_tile_y rows),
        // vectorize+unroll the tile so the k-reduction accumulates straight
        // into registers instead of re-touching memory, and parallelize
        // over tiles. Measured standalone (scratch gemm_tile_bench, same
        // (M,N,K) shapes as this project's feedback GEMM) at parity with
        // MKL single-threaded and ~1.6-2x behind multi-threaded for the
        // size=784/512, nextSize=512 shapes -- a large improvement over the
        // untiled schedule this replaced (which was ~4x slower end-to-end),
        // though still behind MKL's mature multi-level cache blocking. The
        // nextSize=10 (output-layer) shape is NOT handled well by this
        // schedule (reduction too short to amortize tile overhead) --
        // TryFusedFeedbackUpdate in CPUBackend.cpp must keep falling back
        // to the separate oneDNN GEMM + elementwise path for that shape.
        Halide::Var xo, yo, xi, yi, yii;
        const int vec = 8; // AVX2 lane width.
        const int innerTileX = 3 * vec;
        const int innerTileY = 4;

        z_out.tile(x, y, xo, yo, xi, yi, innerTileX, innerTileY, GuardWithIf)
            .vectorize(xi, vec, GuardWithIf)
            .unroll(xi)
            .unroll(yi)
            .parallel(yo);

        // Recomputed fresh per output tile, never materialized as a whole
        // (batch, size) buffer -- this is the actual fusion win over the
        // separate-GEMM-then-separate-elementwise-pass baseline.
        feedback.compute_at(z_out, yo).vectorize(x, vec, GuardWithIf);

        feedback.update(0)
            .split(y, y, yii, innerTileY, GuardWithIf)
            .reorder(x, yii, y, rk)
            .vectorize(x, vec, GuardWithIf)
            .unroll(yii);
    }
};
} // namespace

HALIDE_REGISTER_GENERATOR(FeedbackUpdateSigmoidGenerator, deepity_feedback_update_sigmoid)
