#include "product/multi_asset_autocall/dataset.hpp"
#include "product/multi_asset_autocall/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::heston::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::MultiAssetPhoenixMemoryAutocallPathPolicy, 1U>(
        argc,argv,wb::model::equity::heston::load_models,wb::product::load_multi_asset_phoenix_memory_autocalls,
        "equity/markovian/heston","heston",{"spot","variance"},
        "datasets/parameters/product/multi_asset_phoenix_memory_autocall/phoenix_memory_autocalls_multi_01.json","datasets/multi_asset/multi_asset_phoenix_memory_autocall/five_heston_cir_phoenix_memory_autocalls_100k_01.json","catalog/multi_asset/multi_asset_phoenix_memory_autocall/five_heston_cir_phoenix_memory_autocalls_100k_01/generation.yaml","",202611040800ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,256U,1U,1U,1U},8192U);
}
