#include "product/best_of_call/dataset.hpp"
#include "product/best_of_call/pricing_policy.cuh"
#include "tools/pricing/homogeneous_equity_cir_generation.cuh"
int main(int argc,char** argv) {
    namespace wb=ai_factory::workbench;
    return wb::offline::pricing::generate_homogeneous_equity_cir_prices<
        wb::offline::pricing::FiveMarkov<wb::model::equity::black_scholes::DynamicsPolicy>, wb::simulation::MultiRoughPack<>, wb::product::BestOfPutPathPolicy, 1U>(
        argc,argv,wb::model::equity::black_scholes::load_models,wb::product::load_best_of_options,
        "equity/markovian/black_scholes","black_scholes",{"spot"},
        "datasets/parameters/product/best_of_option/best_of_options_multi_02.json","datasets/multi_asset/best_of_option/five_black_scholes_cir_best_of_puts_100k_01.json","catalog/multi_asset/best_of_option/five_black_scholes_cir_best_of_puts_100k_01/generation.yaml","put",202611040700ULL);
}
