#pragma once

#include "product/basket_call/parameters.hpp"
#include "product/terminal_basket_option_path_policy.cuh"

namespace ai_factory::workbench::product {

using BasketCallPathPolicy = detail::TerminalBasketOptionPathPolicy<
    BasketCallParameters, equity::BasketSpotAggregation::ArithmeticMean, OptionSide::call>;
using BasketPutPathPolicy = detail::TerminalBasketOptionPathPolicy<
    BasketCallParameters, equity::BasketSpotAggregation::ArithmeticMean, OptionSide::put>;

}  // namespace ai_factory::workbench::product
