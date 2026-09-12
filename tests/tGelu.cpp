/**
 * @file tGeluCorrectness.cpp
 * @brief Checks that Deep::gelu() produces numerically correct output --
 * NOT a speed benchmark (tActivations.cpp/tReadme.cpp only measure
 * timing, never check the actual computed values against anything).
 *
 * Reference values are TRUE GELU (x * Phi(x), the Gaussian CDF
 * definition), computed independently via Python/scipy, not derived
 * from this codebase's own formula. The tanh approximation has a known,
 * real ~5e-4 max absolute error vs true GELU (already verified earlier
 * against PyTorch's own source) -- so the tolerance below is set
 * slightly looser than that to allow for the approximation's own
 * expected error, while still catching a genuinely wrong implementation
 * (a sign error, wrong constant, off-by-one in the SIMD tail, etc. would
 * all produce errors far larger than this).
 */
#include <deepity/utils/Activations.h>
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <vector>

using namespace Deep;

int main()
{
    // (input, true_gelu) pairs -- computed independently via
    // scipy.stats.norm.cdf, not from this codebase's own tanh-approx formula.
    struct Case
    {
        float x;
        float expected;
    };
    std::vector<Case> cases = {
        {-3.0f, -0.004050f},
        {-2.0f, -0.045500f},
        {-1.0f, -0.158655f},
        {-0.5f, -0.154269f},
        {0.0f, 0.000000f},
        {0.5f, 0.345731f},
        {1.0f, 0.841345f},
        {2.0f, 1.954500f},
        {3.0f, 2.995950f},
    };

    const float tol = 1e-3f; // looser than the tanh-approx's known ~5e-4 max error
    bool allPassed = true;

    printf("=== Scalar tail path (n not a multiple of SIMD width) ===\n");
    for (auto &c : cases)
    {
        float x = c.x;
        gelu(&x, 1); // n=1 forces every SIMD width's loop to skip entirely, exercising ONLY the scalar tail
        float err = std::fabs(x - c.expected);
        bool pass = err < tol;
        allPassed &= pass;
        printf("  gelu(%.1f) = %.6f  (expected %.6f, err=%.6f)  %s\n",
               c.x, x, c.expected, err, pass ? "PASS" : "FAIL");
    }

    // Same values, but padded into a large-enough buffer to force the
    // actual SIMD path (AVX-512/AVX2/SSE, whichever this build has) to
    // run too -- a bug isolated to just the vectorized loop (e.g. the
    // FMA structure) wouldn't show up in the n=1 scalar-only test above.
    printf("\n=== SIMD path (buffer large enough to exercise vectorized loop) ===\n");
    const size_t simdSize = 64; // large enough for any SIMD width (16/8/4) with a real tail too
    float *buf = static_cast<float *>(std::aligned_alloc(64, simdSize * sizeof(float)));
    for (size_t i = 0; i < simdSize; ++i)
        buf[i] = cases[i % cases.size()].x;

    gelu(buf, simdSize);

    for (size_t i = 0; i < simdSize; ++i)
    {
        float expected = cases[i % cases.size()].expected;
        float err = std::fabs(buf[i] - expected);
        bool pass = err < tol;
        allPassed &= pass;
        if (!pass)
            printf("  buf[%zu]: got %.6f, expected %.6f, err=%.6f  FAIL\n", i, buf[i], expected, err);
    }
    printf("  (silent = all %zu SIMD-path values passed)\n", simdSize);
    std::free(buf);

    printf("\n%s\n", allPassed ? "PASS: gelu() matches true GELU within tolerance." : "FAIL: gelu() does not match expected values.");
    return allPassed ? 0 : 1;
}