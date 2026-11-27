#include "model/equity/markovian/cev/dynamics_impl.cuh"
#include "model/equity/markovian/cev/dataset.hpp"
#include "product/basket_call/dataset.hpp"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::cev::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::BasketCallPathPolicy, 1U, wb::model::fixed_income::vasicek::CorrelatedRatePolicy>(
        argc,argv,wb::model::equity::cev::load_models,wb::product::load_basket_options,
        "equity/markovian/cev","cev",{"spot"},
        "datasets/parameters/product/basket_option/basket_options_multi_02.json","datasets/multi_asset/basket_option/five_cev_vasicek_basket_calls_100k_01.json","catalog/multi_asset/basket_option/five_cev_vasicek_basket_calls_100k_01/generation.yaml","call",202611271100ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,256U,1U,1U,1U},8192U);
}
