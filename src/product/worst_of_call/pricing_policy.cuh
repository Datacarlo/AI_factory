#pragma once

#include "product/terminal_basket_option_path_policy.cuh"
#include "product/worst_of_call/parameters.hpp"

namespace ai_factory::workbench::product {

using WorstOfCallPathPolicy = detail::TerminalBasketOptionPathPolicy<
    WorstOfCallParameters, equity::BasketSpotAggregation::Minimum, OptionSide::call>;
using WorstOfPutPathPolicy = detail::TerminalBasketOptionPathPolicy<
    WorstOfCallParameters, equity::BasketSpotAggregation::Minimum, OptionSide::put>;

}  // namespace ai_factory::workbench::product
