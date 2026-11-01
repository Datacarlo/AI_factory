// Five equity factors with a correlated CIR short-rate numeraire.
#include "product/worst_of_call/dataset.hpp"
#include "product/worst_of_call/pricing_policy.cuh"
#include "tools/pricing/five_equity_cir_generation.cuh"

int main(int argc, char** argv) {
    namespace wb = ai_factory::workbench;
    return wb::offline::pricing::generate_five_equity_cir_prices<
        wb::product::WorstOfCallPathPolicy, 1U>(
        argc, argv, wb::product::load_worst_of_options,
        "datasets/product/worst_of_option/worst_of_options_multi_01.json",
        "datasets/multi_asset/worst_of_option/five_equity_cir_worst_of_calls_01.json",
        "catalog/multi_asset/worst_of_option/five_equity_cir_worst_of_calls_01/generation.yaml",
        "call", {128U,0U,256U,16U,1U,1U}, 131072U);
}
