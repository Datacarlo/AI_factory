#include "product/multi_asset_autocall/dataset.hpp"
#include "product/multi_asset_autocall/parameter_domain.hpp"
#include "datasets/validation.hpp"
#include <stdexcept>
#include <string>

namespace ai_factory::workbench::product {
namespace {
AutocallIndicator indicator(const nlohmann::json& p, const char* field) {
    const auto value = p.at(field).get<std::string>();
    if (value == "worst_of") return AutocallIndicator::worst_of;
    if (value == "best_of") return AutocallIndicator::best_of;
    if (value == "basket") return AutocallIndicator::basket;
    throw std::invalid_argument(std::string(field) + " must be worst_of, best_of or basket.");
}
std::uint32_t day_count(const nlohmann::json& p, const char* field) {
    const auto& v = p.at(field);
    if (!v.is_number_integer() || v.get<double>() <= 0.0
        || v.get<double>() > std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument(std::string(field) + " must be a positive integer day count.");
    return v.get<std::uint32_t>();
}
std::vector<MultiAssetAutocallParameters> load(
    const std::filesystem::path& path, bool athena, const char* family
) {
    return datasets::load_parameter_rows<MultiAssetAutocallParameters>(
        path, datasets::ParameterDatasetFamily::Product, family,
        [athena](const nlohmann::json& p, const std::string& prefix) {
            MultiAssetAutocallParameters result{
                day_count(p, "maturity"), day_count(p, "observation_interval"),
                p.at("autocall_barrier").get<float>(),
                p.at("protection_barrier").get<float>(),
                p.at("annual_coupon_rate").get<float>(),
                indicator(p, "autocall_indicator"),
                indicator(p, "protection_indicator"), indicator(p, "loss_indicator")};
            if (!athena) {
                result.coupon_barrier = p.at("coupon_barrier").get<float>();
                result.coupon_indicator = indicator(p, "coupon_indicator");
            }
            if (!valid_multi_asset_autocall_parameters(result, athena))
                throw std::invalid_argument(prefix + "invalid autocall parameters or observation calendar.");
            return result;
        },
        [athena, family](const nlohmann::json& document) {
            if (document.at("product_family") != family)
                throw std::invalid_argument("Multi-asset autocall family disagrees with its loader.");
            const auto& time = document.at("time_convention");
            if (time.at("unit") != "business_day" || time.at("days_per_year") != 252)
                throw std::invalid_argument("Multi-asset autocalls require 252 business days per year.");
            const auto& quotes = document.at("quote_convention");
            for (const char* name : {"autocall_barrier", "protection_barrier", "coupon_barrier"}) {
                if (athena && std::string_view(name) == "coupon_barrier") continue;
                const auto& q = quotes.at(name);
                if (q.at("mode") != "ratio_to_initial_spot" || q.at("unit") != "dimensionless"
                    || q.at("observation") != "indicator(S_i(t)/S_i(0))")
                    throw std::invalid_argument("Multi-asset autocall barriers apply to normalized spot indicators.");
            }
        });
}
} // namespace
std::vector<MultiAssetAutocallParameters> load_multi_asset_athena_autocalls(const std::filesystem::path& p) {
    return load(p, true, "Multi-asset Athena Autocalls");
}
std::vector<MultiAssetAutocallParameters> load_multi_asset_phoenix_autocalls(const std::filesystem::path& p) {
    return load(p, false, "Multi-asset Phoenix Autocalls");
}
std::vector<MultiAssetAutocallParameters> load_multi_asset_phoenix_memory_autocalls(const std::filesystem::path& p) {
    return load(p, false, "Multi-asset Phoenix Memory Autocalls");
}
} // namespace ai_factory::workbench::product
