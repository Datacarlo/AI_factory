// Generated SABR American-put price-dataset recipe.
#include "model/equity/markovian/sabr/product/american_option.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "product/american_option/dataset.hpp"
#include "tools/pricing/american_option_price_generation.cuh"

#include <cstddef>
#include <cstdint>

int main() {
    using namespace ai_factory::workbench;
    namespace model_binding = model::equity::sabr;
    namespace pricing = offline::pricing;

    constexpr float dt = 1.0f / 504.0f;
    constexpr std::uint32_t simulation_steps_per_day = 2U;

    const pricing::EquityPriceRecipe recipe{
        "datasets/parameters/model/equity/markovian/sabr/parameters/sabr_01.json",
        "datasets/parameters/product/american_option/american_options_01_shard00.json",
        "datasets/single_asset/equity/markovian/sabr/prices/american_puts/"
        "sabr_01__american_puts_01__01_cartesian_shard00.json",
        "catalog/single_asset/equity/markovian/sabr/prices/american_puts/"
        "sabr_01__american_puts_01__01_cartesian_shard00/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/markovian/sabr/prices/"
        "american_puts/sabr_01__american_puts_01__01_cartesian_shard00.json",
        "absorbed Lamperti spot transition with lognormal alpha + Longstaff-Schwartz",
        PriceConstruction::CartesianProduct,
    };
    const pricing::AmericanOptionProfile profile{
        ::ai_factory::workbench::offline::cuda_tuning::kProductionPathsPerPrice,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseThreadsPerBlock,
        ::ai_factory::workbench::offline::cuda_tuning::kEarlyExerciseBlocksPerPrice,
        11668829323166482432ULL,
        "1 / 504",
        "sabr American put",
        "Two-factor Laguerre degree 2",
        "FP64 normal equations and Cholesky on GPU",
        nlohmann::ordered_json{{"simulation_steps_per_day", simulation_steps_per_day}},
        nlohmann::ordered_json::array({"spot", "alpha"}),
        nlohmann::ordered_json::array({"spot / strike", "alpha / alpha0"}),
        nlohmann::ordered_json::array({"1", "L1(spot / strike)", "L2(spot / strike)", "alpha / alpha0", "(alpha / alpha0)^2", "L1(spot / strike) * (alpha / alpha0)"}),
        false,
        ::ai_factory::workbench::offline::cuda_tuning::PricingIdentity{::ai_factory::workbench::offline::cuda_tuning::PricingFamily::equity_lsm, "sabr", "american_option", ""},
    };

    return pricing::generate_american_option_equity_price_dataset(
        recipe,
        profile,
        model_binding::load_models,
        [&](const offline::cuda_tuning::PricingLaunchPlan& plan,
            const auto* device_models, std::size_t model_count,
            const auto* host_products, const auto* device_products,
            std::size_t product_count, PriceConstruction construction,
            std::size_t result_count, std::size_t paths_per_price,
            float* device_prices, float* device_standard_errors) {
            return model_binding::launch_sabr_american_option_cuda<
                OptionSide::put
            >(
                device_models,
                model_count,
                host_products,
                device_products,
                product_count,
                construction,
                result_count,
                paths_per_price,
                dt,
                simulation_steps_per_day,
                plan.profile.threads_per_block,
                plan.profile.blocks_per_price,
                profile.seed,
                device_prices,
                device_standard_errors
            );
        }
    );
}
