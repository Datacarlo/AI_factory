#include "product/multi_asset_autocall/dataset.hpp"
#include "product/multi_asset_autocall/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::black_scholes::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::MultiAssetAthenaAutocallPathPolicy, 1U>(
        argc,argv,wb::model::equity::black_scholes::load_models,wb::product::load_multi_asset_athena_autocalls,
        "equity/markovian/black_scholes","black_scholes",{"spot"},
        "datasets/parameters/product/multi_asset_athena_autocall/athena_autocalls_multi_01.json","datasets/multi_asset/multi_asset_athena_autocall/five_black_scholes_cir_athena_autocalls_100k_01.json","catalog/multi_asset/multi_asset_athena_autocall/five_black_scholes_cir_athena_autocalls_100k_01/generation.yaml","",202611040700ULL);
}
