// Device primitives for the standard normal distribution.
#pragma once

#include <cuda_runtime.h>

namespace ai_factory::workbench {

// Evaluate the standard-normal cumulative distribution in FP32.
__device__ __forceinline__ float normal_cdf(float value) {
    constexpr float inverse_sqrt_two = 0.70710678118654752440f;
    return 0.5f * erfcf(-value * inverse_sqrt_two);
}

// log(1 - Phi(z)), without rounding Phi(z) to one in the positive tail.
// erfcx(x) = exp(x*x)*erfc(x); the negative tail uses log1p to retain
// probabilities close to one. In particular no artificial tail clipping.
__device__ __forceinline__ float normal_log_survival(float value) {
    constexpr float inverse_sqrt_two = 0.70710678118654752440f;
    if (value >= 0.0f) {
        return logf(erfcxf(value * inverse_sqrt_two))
            - 0.69314718055994530942f - 0.5f * value * value;
    }
    return log1pf(-0.5f * erfcf(-value * inverse_sqrt_two));
}

}  // namespace ai_factory::workbench
