// Bond observation tables shared by all paths in each first-sensitivity block.
#pragma once
#include "fixed_income/price_gradients/bond_barrier_first_policy.cuh"
#include "monte_carlo/price_gradients/device_prepared_first_sensitivity_kernel.cuh"
namespace ai_factory::workbench::fixed_income::price_gradients {
template<typename Pricing, typename Preparation, typename HostPlan, typename Inputs>
void launch_bond_barrier_first_sensitivities(
    const HostPlan& host, Inputs inputs,
    const pg::LaunchConfiguration& configuration, pg::Outputs outputs,
    typename HostPlan::StencilOutputs stencil_outputs,
    const char* name, const char* variant
) {
    using Policy = BondBarrierFirstPolicy<Pricing, Preparation>;
    std::size_t stride = 0U;
    if constexpr (requires { Pricing::block_workspace_bytes(host.products.data(), host.products.size(), host.time.simulation_steps_per_day); }) {
        stride = Pricing::block_workspace_bytes(host.products.data(), host.products.size(), host.time.simulation_steps_per_day);
        const std::size_t nodes = host.sensitivities.empty() ? 1U : Policy::kNodeCapacity;
        if (stride > std::numeric_limits<std::size_t>::max() / nodes)
            throw std::overflow_error("Bond sensitivity table overflow.");
        stride *= nodes;
    }
    const auto plan = monte_carlo::price_gradients::make_device_prepared_plan(host);
    const auto sensitivities = std::max(std::size_t{1}, plan.sensitivity_count);
    const auto blocks = std::min(configuration.result_count, configuration.block_count / sensitivities) * sensitivities;
    monte_carlo::BlockPreparationStorage storage(blocks, stride);
    monte_carlo::price_gradients::launch_device_prepared_first_sensitivities<Policy, Preparation>(
        inputs, plan, configuration, outputs, stencil_outputs, name, variant, storage.view());
}
} // namespace ai_factory::workbench::fixed_income::price_gradients
