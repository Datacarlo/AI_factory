// Generated heston_3_2 American selected-gradient recipe with explicit exercise replay.
#include "model/equity/markovian/heston_3_2/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace lspg = longstaff_schwartz::price_gradients;
    namespace pg = price_gradients;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/parameters/model/equity/markovian/heston_3_2/parameters/heston_3_2_01.json", "datasets/parameters/product/american_option/american_options_01_shard00.json", "datasets/single_asset/equity/markovian/heston_3_2/price_gradients/american_calls/heston_3_2_01__american_calls_01__01_cartesian_shard00_price_gradient.json", "catalog/single_asset/equity/markovian/heston_3_2/price_gradients/american_calls/heston_3_2_01__american_calls_01__01_cartesian_shard00_price_gradient/generation.yaml", "catalog/single_asset/equity/markovian/heston_3_2/prices/american_calls/heston_3_2_01__american_calls_01__01_cartesian_shard00/recipe.yaml",
            PriceConstruction::CartesianProduct, {{
        {"model.spot", {0.005, pg::BumpScale::relative}},
        {"model.initial_variance", {0.001, pg::BumpScale::absolute}},
        {"model.rho", {0.002, pg::BumpScale::absolute}},
        {"model.risk_free_rate", {0.0005, pg::BumpScale::absolute}},
        {"model.dividend_yield", {0.0005, pg::BumpScale::absolute}},
        {"model.mean_reversion", {0.005, pg::BumpScale::relative}},
        {"model.long_run_variance", {0.001, pg::BumpScale::absolute}},
        {"model.volatility_of_variance", {0.005, pg::BumpScale::relative}},
        {"product.strike", {0.005, pg::BumpScale::relative}}
            }}, {1.0f/504.0f, 2U}, false};
        recipe.option_side = "call";
        constexpr auto exercise_replay =
            lspg::ExerciseReplayStrategy::frozen_exercise_time;
        recipe.exercise_replay = exercise_replay;
        const auto models = model::equity::heston_3_2::load_models(recipe.model_input);
        const auto products = product::load_american_options(recipe.product_input);
        return offline::pricing::price_gradients::execute_dataset<
            true, pg::SensitivityOrders::first>(recipe,
            {offline::cuda_tuning::PricingFamily::equity_lsm, "heston_3_2", "american_option", ""}, 11668829310281580544ULL,
            models, products, model::equity::heston_3_2::prepare_heston_3_2_american_option_price_gradients,
            offline::pricing::price_gradients::with_exercise_replay(
                model::equity::heston_3_2::launch_heston_3_2_american_option_price_gradients_with_replay_cuda<
                    OptionSide::call>,
                exercise_replay
            ),
            offline::cuda_tuning::kProductionPathsPerPrice,
            model::equity::heston_3_2::prepare_american_option_price_gradient_stencils_cuda);
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
