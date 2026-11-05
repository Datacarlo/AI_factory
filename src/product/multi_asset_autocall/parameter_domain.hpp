// Host-side contract checks shared by JSON loaders and pricing launchers.
#pragma once
#include "product/multi_asset_autocall/parameters.hpp"
#include <cmath>
#include <limits>

namespace ai_factory::workbench::product {
inline bool valid_multi_asset_autocall_parameters(
    const MultiAssetAutocallParameters& p, bool athena,
    std::uint32_t steps_per_day = 1U
) {
    const auto indicator = [](AutocallIndicator i) {
        return static_cast<unsigned>(i) <= static_cast<unsigned>(AutocallIndicator::basket);
    };
    const auto positive = [](float x) { return std::isfinite(x) && x > 0.0f; };
    return p.maturity_days > 0U && p.observation_interval_days > 0U
        && p.maturity_days % p.observation_interval_days == 0U
        && steps_per_day > 0U
        && std::uint64_t(p.maturity_days) * steps_per_day
            <= std::numeric_limits<std::uint32_t>::max()
        && positive(p.autocall_barrier) && positive(p.protection_barrier)
        && std::isfinite(p.annual_coupon_rate) && p.annual_coupon_rate >= 0.0f
        && indicator(p.autocall_indicator) && indicator(p.protection_indicator)
        && indicator(p.loss_indicator)
        && (athena || (positive(p.coupon_barrier) && indicator(p.coupon_indicator)));
}
} // namespace ai_factory::workbench::product
