#include "tools/datasets/parameter_dataset.hpp"
#include "datasets/validation.hpp"
#include <cmath>
int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint32_t maturities[]={126,189,252,315,378,441,504,567,630,756};
    GeneratedRows rows;
    for(auto maturity:maturities) for(unsigned i=0;i<10;++i) {
        const float w=float(i)/9.0f;
        rows.rows.push_back({{"maturity",maturity},{"observation_interval",63U},
            {"autocall_barrier",1.0f+0.10f*w},{"protection_barrier",0.55f+0.20f*w},
            {"annual_coupon_rate",0.04f+0.08f*w},{"autocall_indicator","worst_of"},
            {"protection_indicator","worst_of"},{"loss_indicator","worst_of"}});
        rows.rows.back()["coupon_barrier"]=0.65f+0.20f*w;
        rows.rows.back()["coupon_indicator"]="worst_of";
    }
    rows.construction={{"method","ten maturities times ten paired profiles"},{"maturities_business_days",maturities},{"profile_weight","w=i/9, i=0..9"},{"autocall_barrier_rule","1+0.10*w"},{"protection_barrier_rule","0.55+0.20*w"},{"annual_coupon_rate_rule","0.04+0.08*w"}};
    rows.construction["coupon_barrier_rule"]="0.65+0.20*w";
    write_product_dataset("phoenix_autocalls_multi_01", "Multi-asset Phoenix Autocalls", "datasets/parameters/product/multi_asset_phoenix_autocall/phoenix_autocalls_multi_01.json", "catalog/parameters/product/multi_asset_phoenix_autocall/phoenix_autocalls_multi_01/generation.yaml", "",
        nlohmann::ordered_json::parse(R"json({"maturity": "Business days, 252 per year.", "observation_interval": "63 business days; no observation at inception.", "annual_coupon_rate": "Coupon per fixing = annual_coupon_rate * 63/252.", "autocall_barrier": "Dimensionless worst-of performance threshold.", "protection_barrier": "Dimensionless worst-of performance threshold.", "coupon_barrier": "Dimensionless worst-of performance threshold.", "autocall_indicator": "worst_of", "protection_indicator": "worst_of", "loss_indicator": "worst_of", "coupon_indicator": "worst_of"})json"), nlohmann::ordered_json::parse(R"json({"contract": "Phoenix", "notional": 1, "observation": "min_i(S_i(t)/S_i(0))", "protection": "at maturity only", "discount": "each cashflow discounted by exp(-integral_0^t r_s ds)", "definition": "src/product/multi_asset_autocall/README.md"})json"), rows, nlohmann::ordered_json::parse(R"json({"autocall_barrier": {"mode": "ratio_to_initial_spot", "unit": "dimensionless", "observation": "indicator(S_i(t)/S_i(0))"}, "protection_barrier": {"mode": "ratio_to_initial_spot", "unit": "dimensionless", "observation": "indicator(S_i(t)/S_i(0))"}, "coupon_barrier": {"mode": "ratio_to_initial_spot", "unit": "dimensionless", "observation": "indicator(S_i(t)/S_i(0))"}})json"));
    validate_product_dataset_file("datasets/parameters/product/multi_asset_phoenix_autocall/phoenix_autocalls_multi_01.json");
}
