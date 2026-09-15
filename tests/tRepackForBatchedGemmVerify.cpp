/**
 * @file tRepackForBatchedGemmVerify.cpp
 * @brief Standalone, hand-computable check for
 * IComputeBackend::RepackForBatchedGemm -- verifies it correctly
 * transforms a [batchSize, rows, cols] (batch-major) array into
 * [rows, batchSize, cols] (row-major, batch second), completely
 * independent of SimpleConvPCLayer.
 *
 * batchSize=2, rows=3, cols=2, source values 0..11 sequential for easy
 * hand-verification:
 *   src (batch-major): batch=0: [[0,1],[2,3],[4,5]]
 *                       batch=1: [[6,7],[8,9],[10,11]]
 *   expected dst (row-major, batch second):
 *     row=0: [0,1, 6,7]   (batch=0's row 0, then batch=1's row 0)
 *     row=1: [2,3, 8,9]
 *     row=2: [4,5, 10,11]
 *   expected dst flat: [0,1,6,7, 2,3,8,9, 4,5,10,11]
 */
#include <deepity/backend/CPUBackend.h>
#include <cstdio>
#include <cmath>
#include <vector>

using namespace Deep;

int main()
{
    CPUBackend backend;

    const size_t batchSize = 2, rows = 3, cols = 2;
    std::vector<float> src(batchSize * rows * cols);
    for (size_t i = 0; i < src.size(); ++i)
        src[i] = (float)i;

    std::vector<float> dst(rows * batchSize * cols, -999.0f);

    backend.RepackForBatchedGemm(dst.data(), src.data(), batchSize, rows, cols);

    const std::vector<float> expected = {0, 1, 6, 7, 2, 3, 8, 9, 4, 5, 10, 11};

    bool pass = true;
    for (size_t i = 0; i < expected.size(); ++i)
    {
        bool ok = std::fabs(dst[i] - expected[i]) < 1e-6f;
        pass &= ok;
        if (!ok)
            printf("  [%zu] got %g, expected %g  MISMATCH\n", i, dst[i], expected[i]);
    }

    printf("dst: ");
    for (float v : dst)
        printf("%g ", v);
    printf("\n");

    printf(pass ? "PASS\n" : "FAIL\n");
    return pass ? 0 : 1;
}
