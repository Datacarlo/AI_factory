#include "product/basket_call/dataset.hpp"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::simulation::MultiAssetPack<>, wb::offline::pricing::FiveRough<wb::model::equity::rough_multi_asset::HybridPolicy<wb::model::equity::rough_bergomi::PathPolicy>>, wb::product::BasketCallPathPolicy, 8U>(
        argc,argv,wb::model::equity::rough_bergomi::load_models,wb::product::load_basket_options,
        "equity/rough/rough_bergomi","rough_bergomi",{"spot","variance"},
        "datasets/parameters/product/basket_option/basket_options_multi_02.json","datasets/multi_asset/basket_option/five_rough_bergomi_cir_basket_calls_100k_01.json","catalog/multi_asset/basket_option/five_rough_bergomi_cir_basket_calls_100k_01/generation.yaml","call",202611041000ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,128U,16U,2U,4U},8192U);
}
