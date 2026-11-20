// Model-specific normalization; observation storage and regression inputs are shared.
#pragma once

#include "model/equity/markovian/sabr/dynamics.cuh"
#include "product/american_option/continuation_state.cuh"
#include <cmath>

namespace ai_factory::workbench::model::equity::sabr {

struct AmericanOptionContinuationState
    : product::SpotAndScaledStateContinuationState<
        DynamicsPolicy, &State::alpha, &ModelParameters::initial_volatility
    > {
    __device__ __forceinline__ static PreparedState prepare(
        const ModelParameters& parameters
    ) {
        return {1.0f / (parameters.initial_volatility
            * powf(parameters.spot, 1.0f - parameters.beta))};
    }
};

}  // namespace ai_factory::workbench::model::equity::sabr
