// Five equity factors with a correlated CIR short-rate numeraire.
#include "product/basket_call/dataset.hpp"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/five_equity_cir_generation.cuh"

int main(int argc, char** argv) {
    namespace wb = ai_factory::workbench;
    return wb::offline::pricing::generate_five_equity_cir_prices<
        wb::product::BasketPutPathPolicy, 1U>(
        argc, argv, wb::product::load_basket_options,
        "datasets/parameters/product/basket_option/basket_options_multi_01.json",
        "datasets/multi_asset/basket_option/five_equity_cir_basket_puts_01.json",
        "catalog/multi_asset/basket_option/five_equity_cir_basket_puts_01/generation.yaml",
        "put", {128U,0U,256U,16U,1U,1U}, 131072U);
}
