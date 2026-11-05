// Multi-asset contracts only: single-asset autocall policies remain independent.
#pragma once

#include "multi_asset/equity_basket_observation.cuh"
#include "product/multi_asset_autocall/parameter_domain.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace ai_factory::workbench::product {

struct AutocallObservation {
    float values[3]{}; // worst, best, equally weighted mean of normalized spots
    __host__ __device__ float get(AutocallIndicator indicator) const {
        return values[static_cast<unsigned>(indicator)];
    }
};

// Each selector is constant across a row. Only requested statistics are reduced,
// once each, even when coupon, protection and loss use the same statistic.
template<bool Athena, bool RememberMissedCoupons = false>
struct MultiAssetAutocallPathPolicy {
    static_assert(!Athena || !RememberMissedCoupons);
    static constexpr bool kTerminalCashflow = false;
    static constexpr bool kDiscountedCashflows = true;
    static constexpr bool kAthena = Athena;
    static constexpr bool kCouponMemory = RememberMissedCoupons;
    using ProductParameters = MultiAssetAutocallParameters;

    struct PreparedProduct {
        ProductParameters parameters;
        std::uint32_t observation_count;
        float coupon_per_observation;
        unsigned intermediate_mask;
        unsigned final_mask;
    };

    __host__ __device__ static unsigned indicator_mask(AutocallIndicator x) {
        return 1U << static_cast<unsigned>(x);
    }

    __host__ static bool valid_parameters(const ProductParameters& p,
                                          std::uint32_t steps_per_day) {
        return valid_multi_asset_autocall_parameters(p, Athena, steps_per_day);
    }

    __host__ static void validate_time_configuration(float dt, std::uint32_t steps_per_day) {
        if (!std::isfinite(dt) || !(dt > 0.0f) || steps_per_day == 0U
            || std::abs(dt * float(steps_per_day) - 1.0f / 252.0f) > 1.e-8f)
            throw std::invalid_argument("autocall dt must equal 1/(252*steps_per_day)");
    }

    __host__ __device__ static bool is_observation_step(
        std::uint32_t step, std::uint32_t, std::uint32_t steps_per_day,
        const ProductParameters& p
    ) {
        return (step + 1U) % (p.observation_interval_days * steps_per_day) == 0U;
    }

    __host__ __device__ static std::uint32_t observation_index(
        std::uint32_t step, std::uint32_t steps_per_day, const ProductParameters& p
    ) {
        // Called only at an observation: completed simulation steps -> zero-based fixing.
        return (step + 1U) / (p.observation_interval_days * steps_per_day) - 1U;
    }

    __host__ __device__ static PreparedProduct prepare(
        const ProductParameters& p, float /* terminal discount, unused */
    ) {
        const unsigned coupon = Athena ? 0U : indicator_mask(p.coupon_indicator);
        return {p, p.maturity_days / p.observation_interval_days,
                p.annual_coupon_rate * (float(p.observation_interval_days) / 252.0f),
                indicator_mask(p.autocall_indicator) | coupon,
                indicator_mask(p.protection_indicator) | indicator_mask(p.loss_indicator)
                    | coupon | (Athena ? indicator_mask(p.autocall_indicator) : 0U)};
    }

    struct Handler {
        PreparedProduct product;
        float present_value = 0.0f;
        float accrued_coupon = 0.0f;

        __host__ __device__ bool on_observation(
            std::uint32_t index, const AutocallObservation& value, float discount
        ) {
            const auto& p = product.parameters;
            const bool maturity = index + 1U == product.observation_count;
            const bool called = (Athena || !maturity)
                && value.get(p.autocall_indicator) >= p.autocall_barrier;
            if constexpr (Athena) {
                accrued_coupon += product.coupon_per_observation;
                if (!maturity && called) {
                    present_value = discount * (1.0f + accrued_coupon);
                    return false;
                }
                if (maturity) {
                    const float capital = called ? 1.0f + accrued_coupon
                        : (value.get(p.protection_indicator) >= p.protection_barrier
                            ? 1.0f : value.get(p.loss_indicator));
                    present_value = discount * capital;
                    return false;
                }
            } else {
                float coupon;
                if constexpr (RememberMissedCoupons) {
                    accrued_coupon += product.coupon_per_observation;
                    coupon = value.get(p.coupon_indicator) >= p.coupon_barrier
                        ? accrued_coupon : 0.0f;
                    if (coupon > 0.0f) accrued_coupon = 0.0f;
                } else {
                    coupon = value.get(p.coupon_indicator) >= p.coupon_barrier
                        ? product.coupon_per_observation : 0.0f;
                }
                if (maturity || called) {
                    const float capital = !maturity
                        || value.get(p.protection_indicator) >= p.protection_barrier
                            ? 1.0f : value.get(p.loss_indicator);
                    present_value += discount * (capital + coupon);
                    return false;
                }
                present_value = fmaf(discount, coupon, present_value);
            }
            return true;
        }
    };

    __host__ __device__ static Handler make_handler(const PreparedProduct& p) {
        return {p};
    }
    __host__ __device__ static float finalize(const PreparedProduct&, const Handler& h) {
        return h.present_value; // Every payment already discounted at its own fixing.
    }

    template<typename Pack, unsigned TeamSize>
    __device__ __forceinline__ static AutocallObservation observe(
        const typename Pack::States& states, const typename Pack::Prepared& prepared,
        unsigned lane, const PreparedProduct& product, bool maturity
    ) {
        AutocallObservation result;
        const unsigned mask = maturity ? product.final_mask : product.intermediate_mask;
        if (mask & 1U) result.values[0] = equity::team_basket_spot_ratio<
            equity::BasketSpotAggregation::Minimum, Pack, TeamSize>(states, prepared, lane);
        if (mask & 2U) result.values[1] = equity::team_basket_spot_ratio<
            equity::BasketSpotAggregation::Maximum, Pack, TeamSize>(states, prepared, lane);
        if (mask & 4U) result.values[2] = equity::team_basket_spot_ratio<
            equity::BasketSpotAggregation::ArithmeticMean, Pack, TeamSize>(states, prepared, lane);
        return result;
    }

    template<typename Pack, typename RoughPack, std::size_t RoughCount, unsigned TeamSize>
    __device__ __forceinline__ static AutocallObservation observe_joint_group(
        const typename Pack::States& states, const typename Pack::Prepared& prepared,
        const float (&logs)[RoughCount], const float (&initial)[RoughCount], unsigned lane,
        const PreparedProduct& product, bool maturity
    ) {
        AutocallObservation result;
        const unsigned mask = maturity ? product.final_mask : product.intermediate_mask;
        if (mask & 1U) result.values[0] = equity::team_joint_basket_spot_ratio<
            equity::BasketSpotAggregation::Minimum, Pack, RoughCount, TeamSize>(
                states, prepared, logs, initial, lane);
        if (mask & 2U) result.values[1] = equity::team_joint_basket_spot_ratio<
            equity::BasketSpotAggregation::Maximum, Pack, RoughCount, TeamSize>(
                states, prepared, logs, initial, lane);
        if (mask & 4U) result.values[2] = equity::team_joint_basket_spot_ratio<
            equity::BasketSpotAggregation::ArithmeticMean, Pack, RoughCount, TeamSize>(
                states, prepared, logs, initial, lane);
        return result;
    }
};

using MultiAssetAthenaAutocallPathPolicy = MultiAssetAutocallPathPolicy<true>;
using MultiAssetPhoenixAutocallPathPolicy = MultiAssetAutocallPathPolicy<false, false>;
using MultiAssetPhoenixMemoryAutocallPathPolicy = MultiAssetAutocallPathPolicy<false, true>;

} // namespace ai_factory::workbench::product
