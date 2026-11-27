#include "model/equity/markovian/variance_gamma/dynamics_impl.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/multi_asset_autocall/dataset.hpp"
#include "product/multi_asset_autocall/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::variance_gamma::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::MultiAssetAthenaAutocallPathPolicy, 1U, wb::model::fixed_income::vasicek::CorrelatedRatePolicy>(
        argc,argv,wb::model::equity::variance_gamma::load_models,wb::product::load_multi_asset_athena_autocalls,
        "equity/markovian/variance_gamma","variance_gamma",{"spot"},
        "datasets/parameters/product/multi_asset_athena_autocall/athena_autocalls_multi_01.json","datasets/multi_asset/multi_asset_athena_autocall/five_variance_gamma_vasicek_athena_autocalls_100k_01.json","catalog/multi_asset/multi_asset_athena_autocall/five_variance_gamma_vasicek_athena_autocalls_100k_01/generation.yaml","",202611271000ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,256U,1U,1U,1U},8192U);
}
