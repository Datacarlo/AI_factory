// Reuse affine coefficients across all paths of a grid-monitored bond barrier.
#pragma once
#include "product/zero_coupon_bond_up_and_out/pricing_policy.cuh"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
namespace ai_factory::workbench::product {
template<typename Base, typename Analytics>
struct PreparedBondBarrierPolicy : Base {
    using BondCoefficients = typename Analytics::BondCoefficients;
    struct PreparedRow : Base::PreparedRow {
        const BondCoefficients* bond_coefficients;
    };
    template<typename... Args>
    __device__ __forceinline__ static PreparedRow prepare_row(Args&&... args) {
        return {Base::prepare_row(static_cast<Args&&>(args)...), nullptr};
    }
    static std::size_t block_workspace_bytes(
        const ZeroCouponBondUpAndOutParameters* products,
        std::size_t count, std::uint32_t steps_per_day
    ) {
        std::size_t max_days = 0U;
        for (std::size_t i = 0U; i < count; ++i)
            max_days = std::max(max_days, static_cast<std::size_t>(products[i].option_expiry_days));
        const auto limit = std::numeric_limits<std::size_t>::max() / sizeof(BondCoefficients);
        if (steps_per_day == 0U || max_days > (limit - 1U) / steps_per_day)
            throw std::overflow_error("Bond observation table overflow.");
        return (max_days * steps_per_day + 1U) * sizeof(BondCoefficients);
    }
    __device__ __forceinline__ static void prepare_block(
        PreparedRow& row, unsigned char* storage, [[maybe_unused]] std::size_t bytes
    ) {
        // The host launcher reserves transition_count + 1 entries per node.
        auto* table = reinterpret_cast<BondCoefficients*>(storage);
        if (threadIdx.x == 0U) row.bond_coefficients = table;
        for (std::size_t k = threadIdx.x; k <= row.transition_count; k += blockDim.x) {
            const float time = k == row.transition_count ? row.expiry_years
                : static_cast<float>(k) * row.dt;
            table[k] = Analytics::bond_coefficients(row.analytics, time, row.maturity_years);
        }
        __syncthreads();
    }
    struct Handler : Base::Handler {
        const BondCoefficients* table;
        std::uint32_t observation = 0U;
        using State = typename Base::Schedule::Dynamics::State;
        __device__ __forceinline__ bool observe_cached(const State& state) {
            const auto& c = table[observation++];
            if constexpr (requires { state.state.state_x; }) {
                this->terminal_bond = expf(fmaf(-c.B.state_x, state.state.state_x,
                    fmaf(-c.B.state_y, state.state.state_y, c.log_A)));
            } else {
                this->terminal_bond = expf(fmaf(-c.B, state.state, c.log_A));
            }
            this->invalid = !isfinite(this->terminal_bond);
            this->breached = this->terminal_bond >= this->barrier;
            return !this->invalid && !this->breached;
        }
        __device__ __forceinline__ bool on_initial_state(const State& s) { return observe_cached(s); }
        __device__ __forceinline__ bool on_observation(std::uint32_t, const State& s) { return observe_cached(s); }
    };
    __device__ __forceinline__ static float evaluate_path(
        const PreparedRow& row, philox::PhiloxKey key, std::size_t path
    ) {
        Handler handler{{row.analytics, row.barrier, row.maturity_years,
            row.expiry_years, row.dt, row.transition_count}, row.bond_coefficients};
        return Base::evaluate_path_with_handler(row, key, path, handler);
    }
};
// Preserve historical arithmetic where the affine cache failed bitwise qualification.
template<typename Analytics>
inline constexpr bool cache_barrier_bonds = [] {
    if constexpr (requires { Analytics::cache_bond_coefficients; })
        return Analytics::cache_bond_coefficients;
    else return true;
}();
template<typename Base, typename Analytics>
using PreparedBondBarrierPricingPolicy = std::conditional_t<
    cache_barrier_bonds<Analytics>, PreparedBondBarrierPolicy<Base, Analytics>, Base>;
} // namespace ai_factory::workbench::product
