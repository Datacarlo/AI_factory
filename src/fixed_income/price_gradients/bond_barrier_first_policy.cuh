// First-order CRN sensitivities of a grid-monitored zero-coupon bond barrier.
#pragma once

#include "philox/distributions.cuh"
#include "price_gradients/reconstruction.cuh"
#include "price_gradients/sensitivity_task.cuh"
#include "price_gradients/time_configuration.hpp"

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace ai_factory::workbench::fixed_income::price_gradients {

namespace pg = ::ai_factory::workbench::price_gradients;

// Each node replays the same Philox path. This keeps the stochastic discount,
// bond monitoring, and barrier event paired across the finite-difference nodes.
template<typename Pricing, typename Preparation>
struct BondBarrierFirstPolicy {
    static constexpr std::size_t kNodeCapacity = 3U;
    using Scenario = typename Preparation::Scenario;
    using Nodes = pg::SensitivityNodes<Scenario, kNodeCapacity>;
    using Stencil = pg::SensitivityStencil<kNodeCapacity>;

    struct CentralPolicy {
        using PreparedRow = typename Pricing::PreparedRow;

        __device__ __forceinline__ static PreparedRow prepare(
            const Scenario& scenario, pg::TimeConfiguration time
        ) {
            const typename Pricing::TimeConfiguration schedule_time{
                time.dt, time.simulation_steps_per_day
            };
            PreparedRow row{};
            if constexpr (requires { scenario.curve; }) {
                row = Pricing::prepare_row(
                    scenario.model, scenario.curve, scenario.product, schedule_time
                );
            } else {
                row = Pricing::prepare_row(
                    scenario.model, scenario.product, schedule_time
                );
            }
            // U is continuously bumped; the contract's observation calendar is frozen.
            row.maturity_years = scenario.maturity_years;
            return row;
        }

        __device__ __forceinline__ static void prepare_block(
            PreparedRow& row, unsigned char* data, std::size_t stride
        ) requires requires { Pricing::prepare_block(row, data, stride); } {
            Pricing::prepare_block(row, data, stride);
        }

        __device__ __forceinline__ static float evaluate_path(
            const PreparedRow& row, philox::PhiloxKey key, std::size_t path
        ) {
            return Pricing::evaluate_path(row, key, path);
        }
    };

    // Centered first differences use only the two endpoints. The launcher
    // still requests the central payoff from the first sensitivity block.
    __host__ __device__ static pg::CentralRequirement central_requirement(
        const Stencil& stencil
    ) {
        return stencil.kind == pg::StencilKind::centered
            ? pg::CentralRequirement::none : pg::CentralRequirement::payoff;
    }

    struct PreparedRow {
        typename Pricing::PreparedRow pricing[kNodeCapacity];
        const Stencil* stencil;
        std::uint8_t node_count;
        std::uint8_t first_node;
    };

    __device__ __forceinline__ static PreparedRow prepare(
        const Nodes* nodes,
        const Stencil* stencil,
        pg::TimeConfiguration time,
        bool include_central
    ) {
        PreparedRow row{};
        row.stencil = stencil;
        row.first_node = include_central ? 0U : 1U;
        row.node_count = static_cast<std::uint8_t>(
            pg::active_node_count(*stencil)
        );
        for (unsigned int node = row.first_node; node < row.node_count; ++node) {
            row.pricing[node] = CentralPolicy::prepare((*nodes)[node], time);
        }
        return row;
    }

    __device__ __forceinline__ static void prepare_block(
        PreparedRow& row, unsigned char* data, std::size_t stride
    ) requires requires { Pricing::prepare_block(row.pricing[0], data, stride); } {
        const std::size_t node_stride = stride / kNodeCapacity;
        for (unsigned int node = row.first_node; node < row.node_count; ++node)
            Pricing::prepare_block(row.pricing[node], data + node * node_stride, node_stride);
    }

    struct PathResult {
        float price;
        float sensitivity;
    };

    template<bool IncludeCentral, bool IncludePayoff>
    __device__ __forceinline__ static PathResult evaluate_path(
        const PreparedRow& row,
        philox::PhiloxKey key,
        std::size_t path
    ) {
        static_assert(IncludeCentral || !IncludePayoff);
        pg::SensitivityValues<kNodeCapacity> values{};
        for (unsigned int node = IncludeCentral ? 0U : 1U; node < row.node_count; ++node) {
            values[node] = Pricing::evaluate_path(
                row.pricing[node], key, path
            );
        }
        PathResult result{};
        if constexpr (IncludePayoff) result.price = values[0U];
        result.sensitivity = pg::reconstruct_first_sensitivity(
            *row.stencil, values
        );
        return result;
    }
};

}  // namespace ai_factory::workbench::fixed_income::price_gradients
