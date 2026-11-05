// Generated fixed-income central-price mirror of g2_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian_shard00_price_gradient.
#include "model/fixed_income/g2_plus_plus/product/svensson/european_swaption_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/svensson/dataset.hpp"
#include "product/european_swaption/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2_plus_plus::svensson;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/parameters/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json", "datasets/parameters/product/european_swaption/european_swaptions_01_shard00.json", "datasets/single_asset/fixed_income/g2_plus_plus/prices/svensson/european_payer_swaptions/g2_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian_shard00.json", "catalog/single_asset/fixed_income/g2_plus_plus/prices/svensson/european_payer_swaptions/g2_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian_shard00/generation.yaml",
            "catalog/single_asset/fixed_income/g2_plus_plus/prices/svensson/european_payer_swaptions/g2_plus_plus_01__svensson_01__european_payer_swaptions_01__01_cartesian_shard00/recipe.yaml", PriceConstruction::SampledCurveCartesianProduct,
            {}, {1.0f / 504.0f, 2U}, true
        };
        recipe.price_only_output = true;
        recipe.option_side = "payer";
        recipe.product_quote_mode = "additive_to_initial_par_swap_rate";
        datasets::price_gradients::require_product_quote_convention(recipe);
        recipe.curve_input = "datasets/parameters/curve/svensson/svensson_01.json";
        const auto models =
            model::fixed_income::g2_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::svensson::load_curves(recipe.curve_input);
        const auto product_dataset =
            product::load_european_swaptions(recipe.product_input);
        const auto& products = product_dataset.products;
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_g2_plus_plus_svensson_european_swaption_price_gradients(
                    model_rows, curve_rows, product_rows,
                    construction, time, configuration
                );
        };
        const auto launch = [](
            const auto& plan,
            auto inputs,
            auto stencils,
            const auto& configuration,
            auto outputs
        ) {
            return model_namespace::
                launch_g2_plus_plus_svensson_european_swaption_price_gradients_cuda<
                    SwaptionSide::payer
                >(
                    plan, inputs, stencils, configuration, outputs
                );
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            true,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::fixed_income_mc,
                "g2_plus_plus",
                "european_swaption",
                "svensson",
            },
            11668829177137594368ULL,
            models,
            curves, products,
            prepare,
            launch,
            offline::cuda_tuning::kProductionPathsPerPrice,
            model_namespace::
                prepare_european_swaption_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
