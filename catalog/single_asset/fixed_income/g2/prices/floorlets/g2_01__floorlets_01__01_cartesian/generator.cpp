// Generated fixed-income central-price mirror of g2_01__floorlets_01__01_cartesian_price_gradient.
#include "model/fixed_income/g2/product/rate_option_price_gradients.cuh"
#include "model/fixed_income/g2/dataset.hpp"
#include "product/rate_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/parameters/model/fixed_income/g2/parameters/g2_01.json", "datasets/parameters/product/rate_option/rate_options_01.json", "datasets/single_asset/fixed_income/g2/prices/floorlets/g2_01__floorlets_01__01_cartesian.json", "catalog/single_asset/fixed_income/g2/prices/floorlets/g2_01__floorlets_01__01_cartesian/generation.yaml",
            "catalog/single_asset/fixed_income/g2/prices/floorlets/g2_01__floorlets_01__01_cartesian/recipe.yaml", PriceConstruction::CartesianProduct,
            {}, {1.0f / 504.0f, 2U}, true
        };
        recipe.price_only_output = true;
        recipe.option_side = "put";
        recipe.product_quote_mode = "additive_to_initial_simple_forward_rate";
        datasets::price_gradients::require_product_quote_convention(recipe);
        const auto models =
            model::fixed_income::g2::load_models(recipe.model_input);
        const auto products =
            product::load_rate_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_g2_rate_option_price_gradients(
                    model_rows, product_rows,
                    construction, time, configuration
                );
        };
        return offline::pricing::price_gradients::execute_dataset<
            false,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::closed_form,
                "g2",
                "rate_option",
                "",
            },
            0ULL,
            models,
            products,
            prepare,
            model_namespace::
                launch_g2_rate_option_price_gradients_cuda<
                    OptionSide::put
                >,
            0U,
            model_namespace::
                prepare_rate_option_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
