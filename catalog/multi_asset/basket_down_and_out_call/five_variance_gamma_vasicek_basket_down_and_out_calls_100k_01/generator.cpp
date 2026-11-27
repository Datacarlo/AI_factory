#include "model/equity/markovian/variance_gamma/dynamics_impl.cuh"
#include "model/equity/markovian/variance_gamma/dataset.hpp"
#include "product/basket_down_and_out_call/dataset.hpp"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::variance_gamma::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::BasketDownAndOutCallPathPolicy, 1U, wb::model::fixed_income::vasicek::CorrelatedRatePolicy>(
        argc,argv,wb::model::equity::variance_gamma::load_models,wb::product::load_basket_down_and_out_calls,
        "equity/markovian/variance_gamma","variance_gamma",{"spot"},
        "datasets/parameters/product/basket_down_and_out_call/basket_down_and_out_calls_multi_02.json","datasets/multi_asset/basket_down_and_out_call/five_variance_gamma_vasicek_basket_down_and_out_calls_100k_01.json","catalog/multi_asset/basket_down_and_out_call/five_variance_gamma_vasicek_basket_down_and_out_calls_100k_01/generation.yaml","call",202611271000ULL,
        wb::offline::pricing::CrossAssetBatchGeometry{128U,0U,256U,1U,1U,1U},8192U);
}
