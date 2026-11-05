#include "product/basket_call/dataset.hpp"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::sabr::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::BasketCallPathPolicy, 1U>(
        argc,argv,wb::model::equity::sabr::load_models,wb::product::load_basket_options,
        "equity/markovian/sabr","sabr",{"spot","volatility"},
        "datasets/parameters/product/basket_option/basket_options_multi_02.json","datasets/multi_asset/basket_option/five_sabr_cir_basket_calls_100k_01.json","catalog/multi_asset/basket_option/five_sabr_cir_basket_calls_100k_01/generation.yaml","call",202611040900ULL);
}
