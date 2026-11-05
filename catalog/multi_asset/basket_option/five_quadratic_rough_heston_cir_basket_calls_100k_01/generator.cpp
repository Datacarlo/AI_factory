#include "product/basket_call/dataset.hpp"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::simulation::MultiAssetPack<>, wb::offline::pricing::FiveRough<wb::model::equity::quadratic_rough_heston::CausalFftPathPolicy>, wb::product::BasketCallPathPolicy, 1U>(
        argc,argv,wb::model::equity::quadratic_rough_heston::load_models,wb::product::load_basket_options,
        "equity/rough/quadratic_rough_heston","quadratic_rough_heston",{"spot"},
        "datasets/parameters/product/basket_option/basket_options_multi_02.json","datasets/multi_asset/basket_option/five_quadratic_rough_heston_cir_basket_calls_100k_01.json","catalog/multi_asset/basket_option/five_quadratic_rough_heston_cir_basket_calls_100k_01/generation.yaml","call",202611041100ULL);
}
