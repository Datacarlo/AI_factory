#include "product/multi_asset_autocall/dataset.hpp"
#include "product/multi_asset_autocall/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::simulation::MultiAssetPack<>, wb::offline::pricing::FiveRough<wb::model::equity::rough_multi_asset::HybridPolicy<wb::model::equity::rough_bergomi::PathPolicy>>, wb::product::MultiAssetAthenaAutocallPathPolicy, 8U>(
        argc,argv,wb::model::equity::rough_bergomi::load_models,wb::product::load_multi_asset_athena_autocalls,
        "equity/rough/rough_bergomi","rough_bergomi",{"spot","variance"},
        "datasets/parameters/product/multi_asset_athena_autocall/athena_autocalls_multi_01.json","datasets/multi_asset/multi_asset_athena_autocall/five_rough_bergomi_cir_athena_autocalls_100k_01.json","catalog/multi_asset/multi_asset_athena_autocall/five_rough_bergomi_cir_athena_autocalls_100k_01/generation.yaml","",202611041000ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,128U,16U,2U,4U},8192U);
}
