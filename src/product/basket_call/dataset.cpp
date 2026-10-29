#include "product/basket_call/dataset.hpp"
#include "datasets/validation.hpp"

#include <cmath>
#include <stdexcept>

namespace ai_factory::workbench::product {

std::vector<BasketCallParameters> load_basket_options(
    const std::filesystem::path& dataset_path
) {
    return datasets::load_parameter_rows<BasketCallParameters>(
        dataset_path, datasets::ParameterDatasetFamily::Product,
        "Arithmetic-mean basket option",
        [](const nlohmann::json& parameters, const std::string& prefix) {
            const BasketCallParameters product{
                parameters.at("strike").get<float>(),
                parameters.at("maturity").get<std::uint32_t>()
            };
            if (!std::isfinite(product.strike)
                || product.strike < 0.0f || product.maturity_days == 0U)
                throw std::invalid_argument(
                    prefix + "strike and maturity are out of domain.");
            return product;
        }
    );
}

std::vector<BasketCallParameters> load_basket_calls(
    const std::filesystem::path& dataset_path
) {
    return load_basket_options(dataset_path);
}

}  // namespace ai_factory::workbench::product
