#pragma once

#include "product/basket_call/parameters.hpp"

#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {

std::vector<BasketCallParameters> load_basket_options(
    const std::filesystem::path& dataset_path
);

std::vector<BasketCallParameters> load_basket_calls(
    const std::filesystem::path& dataset_path
);

}  // namespace ai_factory::workbench::product
