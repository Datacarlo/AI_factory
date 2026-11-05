#include "product/multi_asset_autocall/dataset.hpp"
#include "product/multi_asset_autocall/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::simulation::MultiAssetPack<>, wb::offline::pricing::FiveRough<wb::model::equity::quadratic_rough_heston::CausalFftPathPolicy>, wb::product::MultiAssetPhoenixMemoryAutocallPathPolicy, 1U>(
        argc,argv,wb::model::equity::quadratic_rough_heston::load_models,wb::product::load_multi_asset_phoenix_memory_autocalls,
        "equity/rough/quadratic_rough_heston","quadratic_rough_heston",{"spot"},
        "datasets/parameters/product/multi_asset_phoenix_memory_autocall/phoenix_memory_autocalls_multi_01.json","datasets/multi_asset/multi_asset_phoenix_memory_autocall/five_quadratic_rough_heston_cir_phoenix_memory_autocalls_100k_01.json","catalog/multi_asset/multi_asset_phoenix_memory_autocall/five_quadratic_rough_heston_cir_phoenix_memory_autocalls_100k_01/generation.yaml","",202611041100ULL);
}
