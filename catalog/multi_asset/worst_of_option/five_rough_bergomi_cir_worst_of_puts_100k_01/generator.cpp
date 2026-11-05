#include "product/worst_of_call/dataset.hpp"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::simulation::MultiAssetPack<>, wb::offline::pricing::FiveRough<wb::model::equity::rough_multi_asset::HybridPolicy<wb::model::equity::rough_bergomi::PathPolicy>>, wb::product::WorstOfPutPathPolicy, 1U>(
        argc,argv,wb::model::equity::rough_bergomi::load_models,wb::product::load_worst_of_options,
        "equity/rough/rough_bergomi","rough_bergomi",{"spot","variance"},
        "datasets/parameters/product/worst_of_option/worst_of_options_multi_02.json","datasets/multi_asset/worst_of_option/five_rough_bergomi_cir_worst_of_puts_100k_01.json","catalog/multi_asset/worst_of_option/five_rough_bergomi_cir_worst_of_puts_100k_01/generation.yaml","put",202611041000ULL);
}
