// Generated black_scholes forward_start_option selected-gradient recipe.
#include "model/equity/markovian/black_scholes/product/forward_start_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/forward_start_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/parameters/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/parameters/product/forward_start_option/forward_start_options_01_shard00.json", "datasets/single_asset/equity/markovian/black_scholes/price_gradients/forward_start_puts/black_scholes_01__forward_start_puts_01__01_cartesian_shard00_price_gradient.json", "catalog/single_asset/equity/markovian/black_scholes/price_gradients/forward_start_puts/black_scholes_01__forward_start_puts_01__01_cartesian_shard00_price_gradient/generation.yaml", "catalog/single_asset/equity/markovian/black_scholes/prices/forward_start_puts/black_scholes_01__forward_start_puts_01__01_cartesian/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"product.moneyness", {0.005, pg::BumpScale::relative}},
        {"product.maturity_years", {0.001984126984126984, pg::BumpScale::absolute}}
            }}, {1.0f/504.0f, 2U}, true};
        recipe.exact_schedule_transition = true;
        recipe.option_side = "put";
        recipe.reconstructible_sensitivities = {{"model.spot", "price / model.spot", "standard_error / model.spot"}};
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_forward_start_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<false>(recipe,
            {offline::cuda_tuning::PricingFamily::closed_form, "black_scholes", "forward_start_option", ""}, 0ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_forward_start_option_price_gradients,
            model::equity::black_scholes::launch_black_scholes_forward_start_option_price_gradients_cuda<OptionSide::put>,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_forward_start_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
