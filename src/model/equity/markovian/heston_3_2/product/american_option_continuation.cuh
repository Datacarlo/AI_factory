// Model-specific normalization; observation storage and regression inputs are shared.
#pragma once

#include "model/equity/markovian/heston_3_2/dynamics.cuh"
#include "product/american_option/continuation_state.cuh"
#include <cmath>

namespace ai_factory::workbench::model::equity::heston_3_2 {

struct AmericanOptionContinuationState
    : product::SpotAndScaledStateContinuationState<
        DynamicsPolicy, &State::reciprocal_variance, &ModelParameters::long_run_variance
    > {
    __device__ __forceinline__ static PreparedState prepare(
        const ModelParameters& parameters
    ) {
        return {parameters.long_run_variance};
    }
};

}  // namespace ai_factory::workbench::model::equity::heston_3_2
