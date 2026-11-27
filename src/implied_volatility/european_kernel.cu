#include "implied_volatility/european.hpp"
#include "tools/cuda/pricing_runner.cuh"
#include <cmath>

namespace ai_factory::workbench::implied_volatility {
namespace {
// Work in total standard deviation. A bracket also covers prices with very low vega.
__device__ float invert(Quote x, bool call) {
    if (!(x.spot > 0.0f && x.strike > 0.0f && x.years >= 0.0f)
        || !isfinite(x.price) || !isfinite(x.rate) || !isfinite(x.dividend_yield))
        return NAN;

    const float carry_spot = -x.dividend_yield * x.years;
    const float carry_strike = -x.rate * x.years;
    const float spot = x.spot * expf(carry_spot);
    const float strike = x.strike * expf(carry_strike);
    if (!(spot > 0.0f && strike > 0.0f && isfinite(spot) && isfinite(strike) && isfinite(x.years)))
        return NAN;

    // Small carry corrections would lose precision if both legs were rounded first.
    const float spread = fmaxf(fabsf(carry_spot), fabsf(carry_strike)) < 0.5f
        ? fmaf(x.spot, expm1f(carry_spot),
               fmaf(-x.strike, expm1f(carry_strike), x.spot - x.strike))
        : spot - strike;
    const float lower = fmaxf(0.0f, call ? spread : -spread);
    const float upper = call ? spot : strike;
    if (x.years == 0.0f) return 0.0f;
    // The upper bound has infinite implied volatility. Its FP32 predecessor
    // is the closest price we can invert to obtain a finite value.
    const bool at_upper = x.price >= upper;
    const float inversion_price = at_upper ? nextafterf(upper, 0.0f) : x.price;
    if (inversion_price <= lower) return 0.0f;

    // Put-call parity gives the same OTM time value for either side.
    const float a = fminf(spot, strike), b = fmaxf(spot, strike);
    const float log_moneyness = -log1pf(fabsf(spread) / a);
    const float target = inversion_price - lower;
    const float root_t = sqrtf(x.years);
    constexpr float inv_sqrt_two = 0.7071067811865475f;
    constexpr float inv_sqrt_two_pi = 0.3989422804014327f;
    float lo = 0.0f, hi = 128.0f, v = 0.25f;
    for (int i = 0; i < 192; ++i) {
        const float d1 = log_moneyness / v + 0.5f * v;
        const float d2 = d1 - v;
        // Near ATM use erf to retain the difference between two CDFs close to 1/2.
        const float value = fabsf(d1) < 0.5f && fabsf(d2) < 0.5f
            ? 0.5f * (fmaf(a, erff(d1 * inv_sqrt_two), -b * erff(d2 * inv_sqrt_two))
                      - fabsf(spread))
            : 0.5f * fmaf(a, erfcf(-d1 * inv_sqrt_two), -b * erfcf(-d2 * inv_sqrt_two));
        const float slope = a * expf(-0.5f * d1 * d1) * inv_sqrt_two_pi;
        // At saturation, evaluate the distance from the upper bound directly:
        // subtracting two nearly equal prices can erase the final FP32 tick.
        const float error = at_upper
            ? (upper - inversion_price) - 0.5f * (a * erfcf(d1 * inv_sqrt_two)
                                                  + b * erfcf(-d2 * inv_sqrt_two))
            : value - target;
        if (error > 0.0f) hi = v;
        else lo = v;
        const float midpoint = lo + 0.5f * (hi - lo);
        if (error == 0.0f || midpoint == lo || midpoint == hi) {
            return v / root_t;
        }
        const float candidate = v - error / slope;
        const float margin = 0.05f * (hi - lo);
        v = candidate > lo + margin && candidate < hi - margin ? candidate : midpoint;
    }
    return NAN;
}

__global__ void invert_kernel(const Quote* quotes, float* results, size_t size, bool call) {
    const size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < size) results[i] = invert(quotes[i], call);
}
}

void invert_batch(std::span<const Quote> quotes, bool call, std::span<float> results) {
    if (quotes.size() != results.size()) throw std::invalid_argument("IV batch size mismatch");
    if (quotes.empty()) return;
    offline::cuda::DeviceBuffer<Quote> inputs(quotes.size());
    offline::cuda::DeviceBuffer<float> outputs(results.size());
    inputs.copy_from(quotes.data());
    invert_kernel<<<(quotes.size()+255)/256, 256>>>(inputs.data(), outputs.data(), quotes.size(), call);
    check_cuda(cudaGetLastError(), "European implied volatility launch");
    outputs.copy_to(results.data());
}
}
