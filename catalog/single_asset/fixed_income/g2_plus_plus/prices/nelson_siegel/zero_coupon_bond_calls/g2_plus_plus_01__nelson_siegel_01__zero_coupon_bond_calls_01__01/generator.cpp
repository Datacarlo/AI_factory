// Generated fixed-income central-price mirror of g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01_price_gradient.
#include "model/fixed_income/g2_plus_plus/product/nelson_siegel/zero_coupon_bond_option_price_gradients.cuh"
#include "model/fixed_income/g2_plus_plus/dataset.hpp"
#include "curve/nelson_siegel/dataset.hpp"
#include "product/zero_coupon_bond_option/dataset.hpp"
#include "tools/pricing/price_gradients/generation.cuh"

int main() {
    using namespace ai_factory::workbench;
    namespace pg = price_gradients;
    namespace model_namespace = model::fixed_income::g2_plus_plus::nelson_siegel;
    try {
        datasets::price_gradients::Recipe recipe{
            "datasets/parameters/model/fixed_income/g2_plus_plus/parameters/g2_plus_plus_01.json", "datasets/parameters/product/zero_coupon_bond_option/zero_coupon_bond_options_01.json", "datasets/single_asset/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01.json", "catalog/single_asset/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/generation.yaml",
            "catalog/single_asset/fixed_income/g2_plus_plus/prices/nelson_siegel/zero_coupon_bond_calls/g2_plus_plus_01__nelson_siegel_01__zero_coupon_bond_calls_01__01/recipe.yaml", PriceConstruction::Aligned,
            {}, {1.0f / 504.0f, 2U}, true
        };
        recipe.price_only_output = true;
        recipe.option_side = "call";
        recipe.product_quote_mode = "ratio_to_initial_forward_bond_price";
        datasets::price_gradients::require_product_quote_convention(recipe);
        recipe.curve_input = "datasets/parameters/curve/nelson_siegel/nelson_siegel_01.json";
        const auto models =
            model::fixed_income::g2_plus_plus::load_models(recipe.model_input);
        const auto curves = curve::nelson_siegel::load_curves(recipe.curve_input);
        const auto products =
            product::load_zero_coupon_bond_options(recipe.product_input);
        const auto prepare = [](
            const auto& model_rows,
            const auto& curve_rows, const auto& product_rows,
            PriceConstruction construction,
            pg::TimeConfiguration time,
            const pg::PriceGradientConfiguration& configuration
        ) {
            return model_namespace::
                prepare_g2_plus_plus_nelson_siegel_zero_coupon_bond_option_price_gradients(
                    model_rows, curve_rows, product_rows,
                    construction, time, configuration
                );
        };
        return offline::pricing::price_gradients::execute_curve_dataset<
            false,
            pg::SensitivityOrders::first
        >(
            recipe,
            {
                offline::cuda_tuning::PricingFamily::closed_form,
                "g2_plus_plus",
                "zero_coupon_bond_option",
                "nelson_siegel",
            },
            0ULL,
            models,
            curves, products,
            prepare,
            model_namespace::
                launch_g2_plus_plus_nelson_siegel_zero_coupon_bond_option_price_gradients_cuda<
                    OptionSide::call
                >,
            0U,
            model_namespace::
                prepare_zero_coupon_bond_option_price_gradient_stencils_cuda
        );
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
