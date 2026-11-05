// Unit-notional autocalls on S_i(t) / S_i(0), never on absolute spot levels.
#pragma once

#include <cstdint>
#include <type_traits>

namespace ai_factory::workbench::product {

enum class AutocallIndicator : std::uint32_t { worst_of, best_of, basket };

struct MultiAssetAutocallParameters {
    std::uint32_t maturity_days;
    std::uint32_t observation_interval_days;
    float autocall_barrier;
    float protection_barrier;
    float annual_coupon_rate;
    AutocallIndicator autocall_indicator;
    AutocallIndicator protection_indicator;
    AutocallIndicator loss_indicator;
    // Used only by Phoenix and Phoenix Memory, absent from Athena JSON rows.
    float coupon_barrier;
    AutocallIndicator coupon_indicator;
};

static_assert(std::is_trivially_copyable_v<MultiAssetAutocallParameters>);
static_assert(std::is_trivially_default_constructible_v<MultiAssetAutocallParameters>);

} // namespace ai_factory::workbench::product
