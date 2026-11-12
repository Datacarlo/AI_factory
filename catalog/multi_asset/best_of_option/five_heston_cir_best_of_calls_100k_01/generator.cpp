#include "product/best_of_call/dataset.hpp"
#include "product/best_of_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::heston::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::BestOfCallPathPolicy, 1U>(
        argc,argv,wb::model::equity::heston::load_models,wb::product::load_best_of_options,
        "equity/markovian/heston","heston",{"spot","variance"},
        "datasets/parameters/product/best_of_option/best_of_options_multi_02.json","datasets/multi_asset/best_of_option/five_heston_cir_best_of_calls_100k_01.json","catalog/multi_asset/best_of_option/five_heston_cir_best_of_calls_100k_01/generation.yaml","call",202611040800ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,256U,1U,1U,1U},8192U);
}
