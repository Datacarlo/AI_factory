// Generated black_scholes American selected-gradient recipe with explicit exercise replay.
#include "model/equity/markovian/black_scholes/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/parameters/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "datasets/parameters/product/american_option/american_options_01_shard00.json", "datasets/single_asset/equity/markovian/black_scholes/price_gradients/american_puts/black_scholes_01__american_puts_01__01_cartesian_shard00_price_gradient.json", "catalog/single_asset/equity/markovian/black_scholes/price_gradients/american_puts/black_scholes_01__american_puts_01__01_cartesian_shard00_price_gradient/generation.yaml", "catalog/single_asset/equity/markovian/black_scholes/prices/american_puts/black_scholes_01__american_puts_01__01_cartesian_shard00/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.volatility", {0.005, pg::BumpScale::relative}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        recipe.option_side = "put";
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_exercise_time;
        recipe.exercise_replay = exercise_replay;
        const auto models = model::equity::black_scholes::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "black_scholes", "american_option", ""}, 11668826780545843200ULL,
            models, products, model::equity::black_scholes::prepare_black_scholes_american_option_price_gradients,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::black_scholes::launch_black_scholes_american_option_price_gradients_with_replay_cuda<
                    OptionSide::put>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::black_scholes::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
