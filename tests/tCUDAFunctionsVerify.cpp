/**
 * @file tCUDAFunctionsVerify.cpp
 * @brief Verifies every IComputeBackend method not already covered by
 * tMatMulVerify.cpp -- activations, derivatives, Scale, AxpyInto,
 * FusedStateUpdate, ComputeErrorAndEnergy, AdamStep, AdamWStep -- on
 * both CPUBackend and CUDABackend, against expected values computed
 * independently (via a separate Python script, not derived from this
 * codebase's own formulas).
 *
 * IMPORTANT PRECISION NOTE: CUDABackend's tanh/sigmoid kernels use the
 * hardware-approximate `tanh.approx.f32` PTX instruction, trading
 * precision for speed -- CPUBackend uses SLEEF's high-precision
 * (u10 = <=1.0 ULP error) implementation. These will NOT match to tight
 * precision even when both are correct. Functions using this instruction
 * (SIGMOID, TANH, and their derivatives) use a loose tolerance (1e-3);
 * everything else (exact arithmetic: RELU, LINEAR, eSIGMOID, Scale,
 * AxpyInto, FusedStateUpdate, ComputeErrorAndEnergy, Adam/AdamW) uses a
 * tight one (1e-4), since a loose match there would hide a real bug.
 */
#include <deepity/backend/Backend.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>

using namespace Deep;

namespace
{
    int g_failures = 0;

    void Check(const std::vector<float> &actual, const std::vector<float> &expected,
               const char *testName, const char *backendName, float tolerance)
    {
        bool ok = true;
        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (std::fabs(actual[i] - expected[i]) > tolerance)
            {
                printf("  [%s / %s] MISMATCH at index %zu: got %.6f, expected %.6f (tol %.1e)\n",
                       backendName, testName, i, actual[i], expected[i], tolerance);
                ok = false;
            }
        }
        if (ok)
            printf("  [%s / %s] PASSED\n", backendName, testName);
        else
            g_failures++;
    }

    void CheckScalar(float actual, float expected, const char *testName,
                     const char *backendName, float tolerance)
    {
        Check({actual}, {expected}, testName, backendName, tolerance);
    }

    void RunAllTests(DeviceType device, const char *backendName)
    {
        auto backend = CreateBackend(device);
        const std::vector<float> xs = {-2.0f, -1.0f, -0.5f, 0.0f, 0.5f, 1.0f, 2.0f};
        const size_t n = xs.size();

        // --- Activations (independently computed via Python) ---------
        {
            Tensor t(backend.get(), device, xs);
            backend->Activation(ActivationType::RELU, t.Data(), n);
            std::vector<float> out;
            t.CopyToHost(out);
            Check(out, {0, 0, 0, 0, 0.5f, 1.0f, 2.0f}, "relu", backendName, 1e-4f);
        }
        {
            Tensor t(backend.get(), device, xs);
            backend->Activation(ActivationType::SIGMOID, t.Data(), n);
            std::vector<float> out;
            t.CopyToHost(out);
            Check(out, {0.11920292f, 0.26894142f, 0.37754067f, 0.5f, 0.62245933f, 0.73105858f, 0.88079708f},
                  "sigmoid", backendName, 1e-3f); // approx tanh-based on GPU
        }
        {
            Tensor t(backend.get(), device, xs);
            backend->Activation(ActivationType::eSIGMOID, t.Data(), n);
            std::vector<float> out;
            t.CopyToHost(out);
            Check(out, {0.16666667f, 0.25f, 0.33333333f, 0.5f, 0.66666667f, 0.75f, 0.83333333f},
                  "e_sigmoid", backendName, 1e-4f); // exact arithmetic, no approx instruction
        }
        {
            Tensor t(backend.get(), device, xs);
            backend->Activation(ActivationType::TANH, t.Data(), n);
            std::vector<float> out;
            t.CopyToHost(out);
            Check(out, {-0.96402758f, -0.76159416f, -0.46211716f, 0.0f, 0.46211716f, 0.76159416f, 0.96402758f},
                  "tanh", backendName, 1e-3f); // approx instruction on GPU
        }
        {
            Tensor t(backend.get(), device, xs);
            backend->Activation(ActivationType::LINEAR, t.Data(), n);
            std::vector<float> out;
            t.CopyToHost(out);
            Check(out, {-2, -1, -0.5f, 0, 0.5f, 1, 2}, "linear (identity)", backendName, 1e-4f);
        }

        // --- Derivatives, raw-input (...Into) variants ----------------
        {
            Tensor src(backend.get(), device, xs);
            Tensor dst(backend.get(), device, n);
            backend->ActivationDerivativeInto(ActivationType::dRELU, dst.Data(), src.Data(), n);
            std::vector<float> out;
            dst.CopyToHost(out);
            Check(out, {0, 0, 0, 0, 1.0f, 1.0f, 1.0f}, "dRelu", backendName, 1e-4f);
        }
        {
            Tensor src(backend.get(), device, xs);
            Tensor dst(backend.get(), device, n);
            backend->ActivationDerivativeInto(ActivationType::dSIGMOID, dst.Data(), src.Data(), n);
            std::vector<float> out;
            dst.CopyToHost(out);
            Check(out, {0.10499359f, 0.19661193f, 0.23500371f, 0.25f, 0.23500371f, 0.19661193f, 0.10499359f},
                  "dSigmoid", backendName, 1e-3f);
        }
        {
            Tensor src(backend.get(), device, xs);
            Tensor dst(backend.get(), device, n);
            backend->ActivationDerivativeInto(ActivationType::d_eSIGMOID, dst.Data(), src.Data(), n);
            std::vector<float> out;
            dst.CopyToHost(out);
            Check(out, {0.05555556f, 0.125f, 0.22222222f, 0.5f, 0.22222222f, 0.125f, 0.05555556f},
                  "d_eSigmoid", backendName, 1e-4f);
        }
        {
            Tensor src(backend.get(), device, xs);
            Tensor dst(backend.get(), device, n);
            backend->ActivationDerivativeInto(ActivationType::dTANH, dst.Data(), src.Data(), n);
            std::vector<float> out;
            dst.CopyToHost(out);
            Check(out, {0.07065082f, 0.41997434f, 0.78644773f, 1.0f, 0.78644773f, 0.41997434f, 0.07065082f},
                  "dTanh", backendName, 1e-3f);
        }
        {
            Tensor src(backend.get(), device, xs);
            Tensor dst(backend.get(), device, n);
            backend->ActivationDerivativeInto(ActivationType::dLINEAR, dst.Data(), src.Data(), n);
            std::vector<float> out;
            dst.CopyToHost(out);
            Check(out, {1, 1, 1, 1, 1, 1, 1}, "dLinear", backendName, 1e-4f);
        }

        // --- Scale: [1,2,3,4] *= 2.0 -----------------------------------
        {
            Tensor t(backend.get(), device, std::vector<float>{1, 2, 3, 4});
            backend->Scale(t.Data(), 4, 2.0f);
            std::vector<float> out;
            t.CopyToHost(out);
            Check(out, {2, 4, 6, 8}, "Scale", backendName, 1e-4f);
        }

        // --- AxpyInto: y=[1,1,1,1] += 2.0 * x=[1,2,3,4] -----------------
        {
            Tensor y(backend.get(), device, std::vector<float>{1, 1, 1, 1});
            Tensor x(backend.get(), device, std::vector<float>{1, 2, 3, 4});
            backend->AxpyInto(y.Data(), x.Data(), 4, 2.0f);
            std::vector<float> out;
            y.CopyToHost(out);
            Check(out, {3, 5, 7, 9}, "AxpyInto", backendName, 1e-4f);
        }

        // --- FusedStateUpdate: z += ir*(feedback*deriv - e) -------------
        {
            Tensor z(backend.get(), device, std::vector<float>{1.0f, 2.0f});
            Tensor feedback(backend.get(), device, std::vector<float>{0.5f, 1.0f});
            Tensor deriv(backend.get(), device, std::vector<float>{2.0f, 0.5f});
            Tensor e(backend.get(), device, std::vector<float>{0.1f, 0.2f});
            backend->FusedStateUpdate(z.Data(), feedback.Data(), deriv.Data(), e.Data(), 2, 0.1f);
            std::vector<float> out;
            z.CopyToHost(out);
            Check(out, {1.09f, 2.03f}, "FusedStateUpdate", backendName, 1e-4f);
        }

        // --- ComputeErrorAndEnergy: e=z-mu, returns 0.5*sum(e^2) --------
        {
            Tensor z(backend.get(), device, std::vector<float>{3.0f, 5.0f});
            Tensor mu(backend.get(), device, std::vector<float>{1.0f, 2.0f});
            Tensor e(backend.get(), device, 2);
            float energy = backend->ComputeErrorAndEnergy(e.Data(), z.Data(), mu.Data(), 2);
            std::vector<float> eOut;
            e.CopyToHost(eOut);
            Check(eOut, {2.0f, 3.0f}, "ComputeErrorAndEnergy (e)", backendName, 1e-4f);
            CheckScalar(energy, 6.5f, "ComputeErrorAndEnergy (energy)", backendName, 1e-4f);
        }

        // --- AdamStep: param=1.0, grad=0.5, m=v=0, t=1 ------------------
        {
            Tensor param(backend.get(), device, std::vector<float>{1.0f});
            Tensor grad(backend.get(), device, std::vector<float>{0.5f});
            Tensor m(backend.get(), device, 1);
            Tensor v(backend.get(), device, 1);
            backend->AdamStep(param.Data(), grad.Data(), m.Data(), v.Data(), 1, 1, 0.1f);
            std::vector<float> out;
            param.CopyToHost(out);
            CheckScalar(out[0], 0.9000000632455132f, "AdamStep", backendName, 1e-4f);
        }

        // --- AdamWStep: same, plus weightDecay=0.01 ---------------------
        {
            Tensor param(backend.get(), device, std::vector<float>{1.0f});
            Tensor grad(backend.get(), device, std::vector<float>{0.5f});
            Tensor m(backend.get(), device, 1);
            Tensor v(backend.get(), device, 1);
            backend->AdamWStep(param.Data(), grad.Data(), m.Data(), v.Data(), 1, 1, 0.1f, 0.01f);
            std::vector<float> out;
            param.CopyToHost(out);
            CheckScalar(out[0], 0.8990000632455132f, "AdamWStep", backendName, 1e-4f);
        }
    }
}

int main()
{
    printf("--- CPUBackend ---\n");
    RunAllTests(DeviceType::DEVICE_CPU, "CPU");

#ifdef DEEPITY_USE_CUDA
    printf("\n--- CUDABackend ---\n");
    RunAllTests(DeviceType::DEVICE_GPU, "CUDA");
#else
    printf("\n--- CUDABackend skipped (DEEPITY_USE_CUDA not defined) ---\n");
#endif

    if (g_failures > 0)
    {
        printf("\nFAILED: %d check(s) mismatched.\n", g_failures);
        return 1;
    }

    printf("\nPASSED: all backend functions verified correct.\n");
    return 0;
}