// Dense Asian maturity bumps must extend/shorten the whole regular grid.
#include "cuda/check_cuda.cuh"
#include "model/equity/markovian/black_scholes/product/geometric_asian_option_impl.cuh"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace {
namespace wb = ai_factory::workbench;
namespace bs = wb::model::equity::black_scholes;
using Call = bs::GeometricAsianOptionClosedFormPricingPolicy<wb::OptionSide::call>;
using Put = bs::GeometricAsianOptionClosedFormPricingPolicy<wb::OptionSide::put>;
constexpr int cases = 3, nodes = 5;
struct Result { float call, put, central_call, central_put; };
__global__ void maturity_probe(Result* output) {
    const int id = threadIdx.x;
    if (id >= cases * nodes) return;
    const int model_id = id / nodes;
    const unsigned days = model_id == 0 ? 1U : model_id == 1 ? 21U : 504U;
    const float dt = 1.0f / 504.0f;
    const unsigned steps = 2 * days + id % nodes - 2;
    // The zero-time boundary is checked separately from positive stencil nodes.
    const bs::ModelParameters model{1.7f, 0.02f, 0.01f, 0.2f};
    const wb::product::GeometricAsianOptionParameters product{1.53f, days};
    const auto row = Call::prepare_sensitivity_row(model, product, {dt, 2}, steps * dt);
    const auto central = Call::prepare_row(model, product, {dt, 2});
    output[id] = {Call::evaluate_price(row), Put::evaluate_price(row),
                  Call::evaluate_price(central), Put::evaluate_price(central)};
}

// Independent Gaussian construction: log G is the mean of log S(k*dt),
// including k=0. Compute its covariance from min(t_i,t_j), not the CUDA formula.
double reference(unsigned steps, bool call) {
    const double dt = static_cast<double>(1.0f / 504.0f);
    const double T = static_cast<float>(steps * (1.0f / 504.0f));
    const double sigma = static_cast<double>(0.2f), r = static_cast<double>(0.02f);
    const double q = static_cast<double>(0.01f), K = static_cast<double>(1.53f);
    double time_sum = 0, covariance = 0;
    for (unsigned i = 0; i <= steps; ++i) {
        time_sum += i * dt;
        for (unsigned j = 0; j <= steps; ++j) covariance += std::min(i, j) * dt;
    }
    const double count = steps + 1.0;
    const double mean = std::log(static_cast<double>(1.7f))
        + (r - q - sigma * sigma / 2) * time_sum / count;
    const double variance = sigma * sigma * covariance / (count * count);
    const double sign = call ? 1 : -1;
    if (steps == 0) return std::max(sign * (std::exp(mean) - K), 0.0);
    const double d2 = (mean - std::log(K)) / std::sqrt(variance);
    const double d1 = d2 + std::sqrt(variance);
    auto Phi = [](double x) { return 0.5 * std::erfc(-x / std::sqrt(2.0)); };
    return std::exp(-r * T) * sign
        * (std::exp(mean + variance / 2) * Phi(sign * d1) - K * Phi(sign * d2));
}
}
int main() {
    int devices = 0;
    auto availability = cudaGetDeviceCount(&devices);
    if (availability == cudaErrorNoDevice || availability == cudaErrorInsufficientDriver
        || devices == 0) return 77;
    wb::check_cuda(availability, "device discovery");
    std::array<Result, cases * nodes> output{};
    Result* device = nullptr;
    wb::check_cuda(cudaMalloc(&device, sizeof(output)), "allocation");
    maturity_probe<<<1, cases * nodes>>>(device);
    wb::check_cuda(cudaGetLastError(), "launch");
    wb::check_cuda(cudaMemcpy(output.data(), device, sizeof(output), cudaMemcpyDeviceToHost), "copy");
    wb::check_cuda(cudaFree(device), "release");
    for (int c = 0; c < cases; ++c) {
        const unsigned central_steps = c == 0 ? 2U : c == 1 ? 42U : 1008U;
        for (int n = 0; n < nodes; ++n) {
            const auto& v = output[c * nodes + n];
            const unsigned steps = central_steps + n - 2;
            if (std::abs(v.call - reference(steps, true)) > 5e-7
                || std::abs(v.put - reference(steps, false)) > 5e-7)
                throw std::runtime_error("Asian maturity node differs from Gaussian reference");
            if (n == 2 && (v.call != v.central_call || v.put != v.central_put))
                throw std::runtime_error("Central pricing changed");
        }
        for (bool call : {false, true}) {
            const auto& lo = output[c * nodes + 1];
            const auto& hi = output[c * nodes + 3];
            const double width = 2.0 * (1.0f / 504.0f);
            const double actual = (call ? hi.call - lo.call : hi.put - lo.put) / width;
            const double expected = (reference(central_steps + 1, call)
                - reference(central_steps - 1, call)) / width;
            if (std::abs(actual - expected) > 2e-4)
                throw std::runtime_error("Asian maturity gradient differs from Gaussian reference");
        }
    }
    std::cout << "Call/put grid nodes, maturity gradients and unchanged central prices: PASS\n";
}
