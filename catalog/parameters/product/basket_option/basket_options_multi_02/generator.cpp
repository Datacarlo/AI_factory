#include "tools/datasets/parameter_dataset.hpp"
#include "datasets/validation.hpp"
#include <cmath>
int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint32_t maturities[]={21,63,126,189,252,315,378,441,504,756};
    GeneratedRows rows;
    for(auto maturity:maturities) for(unsigned i=0;i<10;++i) {
        const float w=float(i)/9.0f;
        const float slope=-0.15f+(0.3f)*w;
        rows.rows.push_back({{"strike",std::exp(slope*business_days_to_years(maturity))},{"maturity",maturity}});
    }
    rows.construction={{"method","ten maturities times ten exponential strikes"},{"maturities_business_days",maturities},{"log_moneyness_slope_min",-0.15},{"log_moneyness_slope_max",0.15}};
    write_product_dataset("basket_options_multi_02", "Arithmetic-mean basket option", "datasets/parameters/product/basket_option/basket_options_multi_02.json", "catalog/parameters/product/basket_option/basket_options_multi_02/generation.yaml", "",
        nlohmann::ordered_json::parse(R"json({"strike": "Strike on normalized asset indicator.", "maturity": "Business days, 252 per year."})json"), nlohmann::ordered_json::parse(R"json({"expression": "max(side * (mean_i(S_i(T)/S_i(0)) - strike), 0), side=+1 call/-1 put", "notional": "one normalized unit"})json"), rows, nlohmann::ordered_json::parse(R"json({"strike": {"mode": "strike_on_mean_normalized_spot", "unit": "dimensionless", "underlying": "mean_i(S_i(T)/S_i(0))"}})json"));
    validate_product_dataset_file("datasets/parameters/product/basket_option/basket_options_multi_02.json");
}
