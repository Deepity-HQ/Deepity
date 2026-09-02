#pragma once
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <utility>

/**
 * @file Profile.h
 * @brief Per-LAYER, phase-level timing accumulator.
 *
 * Changes from the first draft, per review:
 *  1. Accumulators are now PER LAYER (keyed by layer pointer), not one
 *     global blob -- so "feedback GEMM" for the 784->512 layer is reported
 *     separately from the 512->10 layer, per review point 4.
 *  2. Added calculateStateTotal/updateStateTotal top-level timers, so you
 *     can see e.g. "UpdateState 78%" before drilling into its sub-phases.
 *  3. Print() no longer claims to BE wall-clock time -- PrintAllProfiles()
 *     takes an explicit externally-measured wall-clock duration and prints
 *     it alongside the phase sum as a reconciliation check, per review
 *     point 2. If they don't roughly match, that's itself a finding.
 *  4. "ownGradientTerm" renamed to "localErrorTerm" (it's dz_dt = -p*e,
 *     not a conventional backprop gradient -- review point 5).
 *
 * Still gated behind PCN_PROFILE; a no-op in normal builds.
 */

namespace Deep
{
    struct ProfileAccumulator
    {
        double errorConstruction = 0.0;
        double energyCalculation = 0.0;
        double forwardGemm = 0.0;
        double biasAdd = 0.0;
        double activationForward = 0.0;

        double activationDerivative = 0.0;
        double localErrorTerm = 0.0; // was ownGradientTerm -- dz_dt = -p*e, not a backprop gradient
        double bottomUpConstruction = 0.0;
        double feedbackGemm = 0.0;
        double saxpyUpdate = 0.0;

        double updateWeightsTotal = 0.0;
        double updatePrecisionTotal = 0.0;

        // Top-level timers -- wrap the ENTIRE function body, separate from
        // (and in addition to) the sub-phase timers above, so you can see
        // "UpdateState 78%" before drilling into its five sub-phases.
        double calculateStateTotal = 0.0;
        double updateStateTotal = 0.0;

        long long calculateStateCalls = 0;
        long long updateStateCalls = 0;

        double PhaseSum() const noexcept
        {
            // Sub-phases only -- does NOT include calculateStateTotal/
            // updateStateTotal, since those measure the SAME work the
            // sub-phases already measure (they'd double-count if summed
            // together). calculateStateTotal/updateStateTotal exist purely
            // as an independent top-level cross-check against the sub-phase
            // sum, and against everything outside CalculateState/UpdateState
            // (e.g. UpdateWeights, UpdatePrecision, caller-side overhead).
            return errorConstruction + energyCalculation + forwardGemm + biasAdd + activationForward
                 + activationDerivative + localErrorTerm + bottomUpConstruction + feedbackGemm + saxpyUpdate
                 + updateWeightsTotal + updatePrecisionTotal;
        }

        void Print(const std::string &label) const
        {
            double phaseSum = PhaseSum();
            auto row = [&](const char *name, double t)
            {
                std::cout << "    " << std::left << std::setw(24) << name
                           << std::right << std::setw(10) << std::fixed << std::setprecision(4) << t << " s"
                           << std::setw(8) << std::setprecision(1) << (phaseSum > 0 ? 100.0 * t / phaseSum : 0.0) << "%\n";
            };

            std::cout << "\n--- " << label << " (" << calculateStateCalls
                       << " CalculateState calls, " << updateStateCalls << " UpdateState calls) ---\n";
            std::cout << "  CalculateState() [top-level: " << std::fixed << std::setprecision(4)
                       << calculateStateTotal << " s]\n";
            row("error construction", errorConstruction);
            row("energy calculation", energyCalculation);
            row("forward GEMM", forwardGemm);
            row("bias add", biasAdd);
            row("activation (fwd)", activationForward);
            std::cout << "  UpdateState() [top-level: " << updateStateTotal << " s]\n";
            row("activation derivative", activationDerivative);
            row("local error term", localErrorTerm);
            row("bottom-up construction", bottomUpConstruction);
            row("feedback GEMM", feedbackGemm);
            row("SAXPY update", saxpyUpdate);
            std::cout << "  Other:\n";
            row("UpdateWeights (total)", updateWeightsTotal);
            row("UpdatePrecision (total)", updatePrecisionTotal);
            std::cout << "  Sub-phase sum: " << phaseSum << " s"
                       << "  (top-level CalcState+UpdateState: " << (calculateStateTotal + updateStateTotal) << " s"
                       << " -- these should be CLOSE; large divergence means the sub-phase"
                       << " timers aren't capturing everything the function does)\n";
        }
    };

    struct ProfileEntry
    {
        std::string label;
        ProfileAccumulator acc;
    };

    // Registry: linear vector, not a hash map -- small layer counts, and
    // preserves insertion order (== network layer order) for printing.
    inline std::vector<std::pair<const void *, ProfileEntry>> &ProfileRegistry()
    {
        static std::vector<std::pair<const void *, ProfileEntry>> registry;
        return registry;
    }

    // Looks up (or creates, on first call) this layer's accumulator. Label
    // is generated from the layer's own GetInputSize()/GetOutputSize() --
    // works for any layer type with those methods (DiscriminativePCLayer,
    // ConvPCLayer, ...) via duck-typed template, no coupling to a specific
    // class or forward declaration needed.
    template <typename LayerT>
    inline ProfileAccumulator &LayerProfile(LayerT *layer)
    {
        auto &registry = ProfileRegistry();
        for (auto &entry : registry)
            if (entry.first == layer)
                return entry.second.acc;

        ProfileEntry entry;
        entry.label = "layer(" + std::to_string(layer->GetInputSize()) + "->"
                     + std::to_string(layer->GetOutputSize()) + ") @ " + std::to_string((size_t)layer);
        registry.emplace_back(layer, std::move(entry));
        return registry.back().second.acc;
    }

    // Call once, after the training run, with an INDEPENDENTLY measured
    // wall-clock duration (std::chrono around the whole run in the harness,
    // NOT derived from these accumulators) -- this is the reconciliation
    // check from review point 2.
    inline void PrintAllProfiles(double wallClockSeconds)
    {
        double grandTotal = 0.0;
        for (auto &entry : ProfileRegistry())
        {
            entry.second.acc.Print(entry.second.label);
            grandTotal += entry.second.acc.PhaseSum();
        }
        std::cout << "\n=== Reconciliation ===\n";
        std::cout << "  Wall-clock (measured around whole run): " << std::fixed << std::setprecision(4)
                   << wallClockSeconds << " s\n";
        std::cout << "  Sum of all layers' phase sums:          " << grandTotal << " s\n";
        std::cout << "  Unaccounted-for time:                   " << (wallClockSeconds - grandTotal) << " s"
                   << "  (" << (wallClockSeconds > 0 ? 100.0 * (wallClockSeconds - grandTotal) / wallClockSeconds : 0.0)
                   << "% -- data loading, Python/pybind overhead, timer overhead itself, etc.)\n";
    }

#ifdef PCN_PROFILE
    class ScopedTimer
    {
    public:
        explicit ScopedTimer(double &accumulator) noexcept
            : accumulator(accumulator), start(std::chrono::steady_clock::now()) {}
        ~ScopedTimer() noexcept
        {
            auto end = std::chrono::steady_clock::now();
            accumulator += std::chrono::duration<double>(end - start).count();
        }
        ScopedTimer(const ScopedTimer &) = delete;
        ScopedTimer &operator=(const ScopedTimer &) = delete;
    private:
        double &accumulator;
        std::chrono::steady_clock::time_point start;
    };
#define PCN_TIME(accumulator) ::Deep::ScopedTimer _pcn_timer_##__LINE__(accumulator)
#else
    class ScopedTimer
    {
    public:
        explicit ScopedTimer(double &) noexcept {}
    };
#define PCN_TIME(accumulator) do {} while (0)
#endif
} // namespace Deep