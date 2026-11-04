// Five equities, correlated CIR rate and a daily monitored basket barrier.
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "tools/pricing/five_equity_cir_generation.cuh"
#include "datasets/validation.hpp"

#include <cmath>
#include <stdexcept>

int main(int argc, char** argv) {
    namespace wb = ai_factory::workbench;
    const auto load_product=[](const std::filesystem::path& dataset_path) {
        return wb::datasets::load_parameter_rows<
            wb::product::BasketDownAndOutCallParameters>(
            dataset_path,wb::datasets::ParameterDatasetFamily::Product,
            "Basket down-and-out call",
            [](const nlohmann::json& parameters,const std::string& prefix) {
                const wb::product::BasketDownAndOutCallParameters product{
                    parameters.at("strike").get<float>(),
                    parameters.at("lower_barrier").get<float>(),
                    parameters.at("maturity").get<std::uint32_t>(),
                    parameters.at("monitoring_interval_days").get<std::uint32_t>()};
                if (!std::isfinite(product.strike) || product.strike<0.0f
                    || !std::isfinite(product.lower_barrier)
                    || product.lower_barrier<=0.0f
                    || product.maturity_days==0U
                    || product.monitoring_interval_days==0U)
                    throw std::invalid_argument(prefix+"invalid basket barrier contract");
                return product;
            });
    };
    return wb::offline::pricing::generate_five_equity_cir_prices<
        wb::product::BasketDownAndOutCallPathPolicy,1U>(
        argc,argv,load_product,
        "datasets/parameters/product/basket_down_and_out_call/basket_down_and_out_calls_multi_01.json",
        "datasets/multi_asset/basket_down_and_out_call/five_equity_cir_basket_down_and_out_calls_01.json",
        "catalog/multi_asset/basket_down_and_out_call/five_equity_cir_basket_down_and_out_calls_01/generation.yaml",
        "call", {128U,0U,256U,16U,1U,1U}, 131072U);
}
