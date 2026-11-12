// Brownian-driven, positivity-preserving CIR step for cross-asset coupling.
#pragma once
#include "fixed_income/no_curve.hpp"
#include "philox/domains.cuh"
#include "model/fixed_income/cir/parameter_domain.hpp"
#include <cuda_runtime.h>

namespace ai_factory::workbench::model::fixed_income::cir {
struct CorrelatedRatePolicy {
    using Parameters = ModelParameters;
    using CurveParameters = ::ai_factory::workbench::fixed_income::NoCurveParameters;
    struct State {
        float short_rate;
        float integral;
        float raw_rate;
    };
    struct Prepared {
        float kappa_dt, theta, sigma_sqrt_dt, dt, initial_rate;
    };
    static constexpr unsigned kBrownianDimension = 1U;
    static constexpr bool kUsesCurve = false;

    __host__ static bool valid_parameters(const Parameters& p) {
        return cir::valid_parameters(p);
    }
    __device__ __forceinline__ static Prepared prepare(
        const Parameters& p, const CurveParameters&, float dt) {
        return {p.process.mean_reversion*dt,p.process.long_term_mean,
                p.process.volatility*sqrtf(dt),dt,p.initial_state};
    }
    __device__ __forceinline__ static State initial_state(const Prepared& p) {
        return {p.initial_rate,0.0f,p.initial_rate};
    }
    __device__ __forceinline__ static float one_step_transition(
        const Prepared& p, const float* z, philox::PhiloxKey,
        std::uint64_t, std::uint32_t, std::uint32_t, State& state) {
        const float previous=state.short_rate;
        // Keep the auxiliary Euler state signed. Only the drift/diffusion
        // coefficients and observable short rate use its positive part.
        const float raw=state.raw_rate+p.kappa_dt*(p.theta-previous)
            +p.sigma_sqrt_dt*sqrtf(fmaxf(previous,0.0f))*z[0];
        const float next=fmaxf(0.0f,raw);
        const float increment=0.5f*p.dt*(previous+next);
        state.short_rate=next;
        state.integral+=increment;
        state.raw_rate=raw;
        return increment;
    }
};
}  // namespace ai_factory::workbench::model::fixed_income::cir
