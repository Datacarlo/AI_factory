#include "model/equity/markovian/bates/dynamics_impl.cuh"
#include "model/equity/markovian/bates/dataset.hpp"
#include "product/best_of_call/dataset.hpp"
#include "product/best_of_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::bates::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::BestOfCallPathPolicy, 1U, wb::model::fixed_income::vasicek::CorrelatedRatePolicy>(
        argc,argv,wb::model::equity::bates::load_models,wb::product::load_best_of_options,
        "equity/markovian/bates","bates",{"spot","variance"},
        "datasets/parameters/product/best_of_option/best_of_options_multi_02.json","datasets/multi_asset/best_of_option/five_bates_vasicek_best_of_calls_100k_01.json","catalog/multi_asset/best_of_option/five_bates_vasicek_best_of_calls_100k_01/generation.yaml","call",202611271200ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,256U,1U,1U,1U},8192U);
}
