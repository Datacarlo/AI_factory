#include "product/best_of_call/dataset.hpp"
#include "product/best_of_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::simulation::MultiAssetPack<>, wb::offline::pricing::FiveRough<wb::model::equity::quadratic_rough_heston::CausalFftPathPolicy>, wb::product::BestOfPutPathPolicy, 1U>(
        argc,argv,wb::model::equity::quadratic_rough_heston::load_models,wb::product::load_best_of_options,
        "equity/rough/quadratic_rough_heston","quadratic_rough_heston",{"spot"},
        "datasets/parameters/product/best_of_option/best_of_options_multi_02.json","datasets/multi_asset/best_of_option/five_quadratic_rough_heston_cir_best_of_puts_100k_01.json","catalog/multi_asset/best_of_option/five_quadratic_rough_heston_cir_best_of_puts_100k_01/generation.yaml","put",202611041100ULL);
}
