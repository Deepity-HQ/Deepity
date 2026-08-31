#include "DirectKPPCLayer.h"

namespace Deep
{
    void DirectKPPCLayer::BindMemory(MemoryArena &arena)
    {
        return;
    }

    size_t DirectKPPCLayer::GetRequiredFloats() const noexcept
    {
        return 0;
    }

    void DirectKPPCLayer::RandomizeWeights(std::mt19937 &seedGenerator) noexcept
    {
        return;
    }

    float DirectKPPCLayer::CalculateState() noexcept;
    void DirectKPPCLayer::ComputeMuOnly() noexcept;
    void DirectKPPCLayer::UpdateState() noexcept;   // Will now pull from terminalLayer->GetErrors()
    void DirectKPPCLayer::UpdateWeights() noexcept; // Must compute \Delta W AND \Delta \Psi

    void ClampState(const std::vector<float> &inputData) noexcept;
    void UnclampState() noexcept;
    void ResetState() noexcept;

}