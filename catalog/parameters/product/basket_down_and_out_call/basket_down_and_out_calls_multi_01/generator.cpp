#include "tools/datasets/parameter_dataset.hpp"
#include "datasets/validation.hpp"

#include <cmath>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench::datasets;
    constexpr std::uint32_t maturities[] = {
        42U,126U,252U,378U,504U,630U,756U,882U,1008U,1512U
    };
    GeneratedRows rows;
    for (const auto maturity : maturities) {
        const float years=business_days_to_years(maturity);
        for (std::uint32_t i=0U;i<10U;++i) {
            const float weight=static_cast<float>(i)/9.0f;
            const float slope=-0.075f+0.15f*weight;
            rows.rows.push_back({
                {"strike",std::exp(slope*years)},
                {"lower_barrier",0.70f+0.20f*weight},
                {"maturity",maturity},
                {"monitoring_interval_days",2U}
            });
        }
    }
    rows.construction={{"method","ten maturities times ten paired strikes and barriers"},
        {"maturities_business_days",maturities},
        {"log_moneyness_slope_min",-0.075f},
        {"log_moneyness_slope_max",0.075f},
        {"lower_barrier_min",0.70f},
        {"lower_barrier_max",0.90f},
        {"monitoring_interval_business_days",2U},
        {"barrier_convention","discrete observations every two business days including maturity; equality knocks out; inception knocks out if barrier is at least one; no bridge correction"},
        {"core_only",true}};
    write_product_dataset(
        "basket_down_and_out_calls_multi_01",
        "Arithmetic-mean basket down-and-out call",
        "datasets/parameters/product/basket_down_and_out_call/basket_down_and_out_calls_multi_01.json",
        "catalog/parameters/product/basket_down_and_out_call/basket_down_and_out_calls_multi_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/product/basket_down_and_out_call/basket_down_and_out_calls_multi_01.json",
        {{"strike","Dimensionless strike on the mean of initial-spot-normalized assets."},
         {"lower_barrier","Dimensionless lower barrier on that same mean."},
         {"maturity","Business days from valuation; 252 days per year."},
         {"monitoring_interval_days","Business days between observations."}},
        {{"expression","1{mean_i(S_i(t)/S_i(0)) > lower_barrier at every observation t} * max(mean_i(S_i(T)/S_i(0)) - strike, 0)"},
         {"notional","one normalized basket unit"}},
        rows,
        {{"strike",{{"mode","strike_on_mean_normalized_spot"},{"unit","dimensionless"},{"underlying","mean_i(S_i(T)/S_i(0))"}}},
         {"lower_barrier",{{"mode","lower_barrier_on_mean_normalized_spot"},{"unit","dimensionless"},{"underlying","mean_i(S_i(t)/S_i(0))"}}}}
    );
    validate_product_dataset_file(
        "datasets/parameters/product/basket_down_and_out_call/basket_down_and_out_calls_multi_01.json");
}
