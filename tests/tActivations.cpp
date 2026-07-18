#include <math.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "Activations.h"

typedef struct {
    double max_abs;
    double max_rel;
    double rms;
    float worst_x;
    float expected;
    float actual;
} Stats;

static Stats test_tanh(size_t n)
{
    float *input = (float *)malloc(n * sizeof(float));
    float *test  = (float *)malloc(n * sizeof(float));

    if (!input || !test) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    Stats s = {0};

    double sum_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {

        // Uniformly sample [-10,10]
        float x = -10.0f + 20.0f * (float)i / (float)(n - 1);

        input[i] = x;
        test[i] = x;
    }

    Deep::tanh(test, n);

    for (size_t i = 0; i < n; ++i) {

        float expected = tanhf(input[i]);
        float actual   = test[i];

        double abs_err = fabs((double)actual - expected);
        double rel_err = abs_err / fmax(fabs(expected), 1e-30);

        sum_sq += abs_err * abs_err;

        if (abs_err > s.max_abs) {
            s.max_abs = abs_err;
            s.worst_x = input[i];
            s.expected = expected;
            s.actual = actual;
        }

        if (rel_err > s.max_rel)
            s.max_rel = rel_err;
    }

    s.rms = sqrt(sum_sq / n);

    free(input);
    free(test);

    return s;
}

static Stats test_dtanh(size_t n)
{
    float *input = (float *)malloc(n * sizeof(float));
    float *test  = (float *)malloc(n * sizeof(float));

    if (!input || !test) {
        fprintf(stderr, "Allocation failed\n");
        exit(1);
    }

    Stats s = {0};

    double sum_sq = 0.0;

    for (size_t i = 0; i < n; ++i) {

        float x = -10.0f + 20.0f * (float)i / (float)(n - 1);

        input[i] = x;
        test[i] = x;
    }

    Deep::dTanh(test, n);

    for (size_t i = 0; i < n; ++i) {

        float t = tanhf(input[i]);
        float expected = 1.0f - t * t;
        float actual   = test[i];

        double abs_err = fabs((double)actual - expected);
        double rel_err = abs_err / fmax(fabs(expected), 1e-30);

        sum_sq += abs_err * abs_err;

        if (abs_err > s.max_abs) {
            s.max_abs = abs_err;
            s.max_rel = rel_err;
            s.worst_x = input[i];
            s.expected = expected;
            s.actual = actual;
        }

        if (rel_err > s.max_rel)
            s.max_rel = rel_err;
    }

    s.rms = sqrt(sum_sq / n);

    free(input);
    free(test);

    return s;
}

int main(void)
{
    const size_t N = 1000000;

    Stats t = test_tanh(N);
    Stats d = test_dtanh(N);

    printf("===== tanh =====\n");
    printf("Max abs error : %.9e\n", t.max_abs);
    printf("Max rel error : %.9e\n", t.max_rel);
    printf("RMS error     : %.9e\n", t.rms);
    printf("Worst x       : %f\n", t.worst_x);
    printf("Expected      : %.9f\n", t.expected);
    printf("Actual        : %.9f\n\n", t.actual);

    printf("===== dTanh =====\n");
    printf("Max abs error : %.9e\n", d.max_abs);
    printf("Max rel error : %.9e\n", d.max_rel);
    printf("RMS error     : %.9e\n", d.rms);
    printf("Worst x       : %f\n", d.worst_x);
    printf("Expected      : %.9f\n", d.expected);
    printf("Actual        : %.9f\n", d.actual);

    return 0;
}