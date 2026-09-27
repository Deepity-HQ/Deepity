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
    /// @brief One layer's accumulated time-per-phase, in seconds --
    /// see the file-level note for the reconciliation invariants
    /// (PhaseSum() vs. the top-level totals vs. external wall-clock).
    struct ProfileAccumulator
    {
        double errorConstruction = 0.0;  ///< CalculateState: building the error term.
        double energyCalculation = 0.0;  ///< CalculateState: the energy reduction itself.
        double forwardGemm = 0.0;        ///< CalculateState: the forward prediction GEMM.
        double biasAdd = 0.0;            ///< CalculateState: bias-add after the forward GEMM.
        double activationForward = 0.0;  ///< CalculateState: forward activation function.

        double activationDerivative = 0.0; ///< UpdateState: activation derivative.
        double localErrorTerm = 0.0; ///< UpdateState: dz_dt = -p*e (was ownGradientTerm -- not a backprop gradient).
        double bottomUpConstruction = 0.0; ///< UpdateState: assembling the bottom-up term.
        double feedbackGemm = 0.0;         ///< UpdateState: the feedback GEMM.
        double saxpyUpdate = 0.0;          ///< UpdateState: the final z += ir*dz SAXPY.

        double updateWeightsTotal = 0.0;    ///< UpdateWeights(), whole-call total.
        double updatePrecisionTotal = 0.0;  ///< UpdatePrecision(), whole-call total.

        // Top-level timers -- wrap the ENTIRE function body, separate from
        // (and in addition to) the sub-phase timers above, so you can see
        // "UpdateState 78%" before drilling into its five sub-phases.
        double calculateStateTotal = 0.0; ///< CalculateState(), whole-call total.
        double updateStateTotal = 0.0;    ///< UpdateState(), whole-call total.

        long long calculateStateCalls = 0; ///< Number of CalculateState() calls recorded.
        long long updateStateCalls = 0;    ///< Number of UpdateState() calls recorded.

        /// @brief Sum of every sub-phase timer (excludes the top-level
        /// calculateStateTotal/updateStateTotal, which measure the same
        /// work and would double-count if included -- see the inline note).
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

        /// @brief Prints this accumulator's per-phase breakdown (with
        /// percentages of PhaseSum()) plus the top-level/sub-phase
        /// reconciliation check, to stdout.
        /// @param label Heading printed above this accumulator's rows,
        /// e.g. a layer's shape and identity (see LayerProfile()).
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

    /// @brief One registered layer's profiling data: its print label
    /// paired with its accumulator.
    struct ProfileEntry
    {
        std::string label;    ///< Printed heading, e.g. "layer(784->512) @ 0x...".
        ProfileAccumulator acc; ///< That layer's accumulated phase timings.
    };

    // Registry: linear vector, not a hash map -- small layer counts, and
    // preserves insertion order (== network layer order) for printing.
    /// @brief The process-wide table of every layer that's had
    /// LayerProfile() called on it at least once, in first-call order.
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
    /// @brief Looks up (or creates, on first call) `layer`'s accumulator
    /// in the process-wide registry, keyed by its address.
    /// @tparam LayerT Any layer type exposing GetInputSize()/GetOutputSize().
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
    /// @brief Prints every registered layer's profile, then a grand-total
    /// reconciliation against an independently measured wall-clock time.
    /// @param wallClockSeconds Wall-clock duration of the whole run,
    /// measured separately (e.g. std::chrono around the training loop) --
    /// NOT derived from these accumulators.
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
    /// @brief RAII stopwatch: adds the elapsed time since construction
    /// into `accumulator` when it goes out of scope. Used via PCN_TIME()
    /// to time one phase of CalculateState()/UpdateState() with a single
    /// line at the top of the block being measured.
    class ScopedTimer
    {
    public:
        /// @param accumulator Reference to the ProfileAccumulator field
        /// this timer's elapsed duration is added into on destruction.
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
    /// @brief Times the rest of the enclosing block, adding its duration
    /// into `accumulator`. A no-op unless PCN_PROFILE is defined.
    #define PCN_TIME(accumulator) ::Deep::ScopedTimer _pcn_timer_##__LINE__(accumulator)
#else
    /// @brief No-op stand-in for the PCN_PROFILE build of ScopedTimer
    /// above, so PCN_TIME() call sites compile identically either way.
    class ScopedTimer
    {
    public:
        /// @brief No-op; the accumulator reference is discarded.
        explicit ScopedTimer(double &) noexcept {}
    };
    /// @brief No-op unless PCN_PROFILE is defined -- see the PCN_PROFILE
    /// branch above.
    #define PCN_TIME(accumulator) do {} while (0)
#endif
} // namespace Deep