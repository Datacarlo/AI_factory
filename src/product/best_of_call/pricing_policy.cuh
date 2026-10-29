#pragma once

#include "product/best_of_call/parameters.hpp"
#include "product/terminal_basket_option_path_policy.cuh"

namespace ai_factory::workbench::product {

using BestOfCallPathPolicy = detail::TerminalBasketOptionPathPolicy<
    BestOfCallParameters, equity::BasketSpotAggregation::Maximum, OptionSide::call>;
using BestOfPutPathPolicy = detail::TerminalBasketOptionPathPolicy<
    BestOfCallParameters, equity::BasketSpotAggregation::Maximum, OptionSide::put>;

}  // namespace ai_factory::workbench::product
