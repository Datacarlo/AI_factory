// QE copula, stable Gaussian tails, and independent auxiliary Philox sources.
#include "cuda/check_cuda.cuh"
#include "model/equity/markovian/bates/dynamics_impl.cuh"
#include "model/equity/markovian/merton/dynamics_impl.cuh"
#include "model/equity/markovian/kou/dynamics_impl.cuh"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace {
using namespace ai_factory::workbench;
namespace heston = model::equity::heston;
namespace bates = model::equity::bates;
struct Result {
    float log_tail, variance, log_spot, bates_variance, spot_difference;
    unsigned int identical, streams_ok;
};
constexpr unsigned int kPaths = 1U << 18U;
constexpr float kDt = 1.0f / 504.0f;
const heston::ModelParameters kParameters{
    1.0f, 0.03f, 0.01f, 0.0f, 1.6f, 0.032f, 0.67f, -0.7f};

__global__ void check_qe_kernel(heston::ModelParameters p, Result* results) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const auto key = philox::make_key(81379U);
    philox::NormalRandomContext normals(key, i);
    const float zv = philox::next_normal(normals.uniforms, normals.normals);
    const float independent = philox::next_normal(normals.uniforms, normals.normals);
    const auto prepared = heston::prepare_model(p, kDt);
    const heston::BrownianStep shock{
        prepared.rho * zv + prepared.correlation_residual * independent, zv};
    auto a = heston::initial_state(prepared), b = a;
    const auto domain = philox::source_step_domain<2U>(7U);
    philox::DomainUniformSequence auxiliary(key, i, domain);
    philox::DomainUniformSequence other(key, i, philox::source_step_domain<3U>(7U));
    heston::DynamicsPolicy::one_step_transition(prepared, shock, auxiliary, a);
    heston::DynamicsPolicy::one_step_transition(prepared, shock, other, b);
    const bates::ModelParameters bp{p.spot, p.risk_free_rate, p.dividend_yield,
        p.initial_variance, p.kappa, p.theta, p.gamma, p.rho, 8.0f, -0.1f, 0.2f};
    const auto bd = bates::prepare_model(bp, kDt);
    auto bs = bates::initial_state(bd);
    bates::DynamicsPolicy::one_step_transition(bd, shock, auxiliary, bs);

    // Consumption of count/parent streams must not advance marks or change
    // their address. Check consecutive quads, not just the first draw.
    auto count = auxiliary.source<2U>();
    for (unsigned int j = 0; j < (i % 17U); ++j) (void)count.next();
    for (unsigned int j = 0; j < (i % 9U); ++j) (void)auxiliary.next();
    auto marks = auxiliary.source<3U>();
    unsigned int streams_ok = 1U;
    for (unsigned int group = 0; group < 2U; ++group) {
        const auto q = philox::domain_uniform_quad(key, i, 8U * group + 3U, domain);
        streams_ok &= marks.next() == q.first;
        streams_ok &= marks.next() == q.second;
        streams_ok &= marks.next() == q.third;
        streams_ok &= marks.next() == q.fourth;
    }
    results[i] = {normal_log_survival(-12.0f + float(i % 97U) * 0.25f),
        a.variance, a.log_spot, bs.variance, a.log_spot - b.log_spot,
        unsigned(a.variance == b.variance), streams_ok};
}

// Deterministic inputs reach tails that a small Monte Carlo test will not visit.
__global__ void check_qe_tail_kernel(heston::ModelParameters p, float2* output) {
    const unsigned int i = threadIdx.x;
    const float z = -12.0f + float(i) * 0.25f;
    const auto prepared = heston::prepare_model(p, kDt);
    auto state = heston::initial_state(prepared);
    heston::correlated_one_step_transition(prepared, z, 0.0f, state);
    output[i] = make_float2(state.variance, normal_log_survival(z));
}
void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
} // namespace

int main() {
    using namespace ai_factory::workbench;
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
    Result* device = nullptr;
    check_cuda(cudaMalloc(&device, kPaths * sizeof(Result)), "QE allocate");
    std::vector<Result> results(kPaths);
    // Exercise both branches with the same parameter family.
    for (const float initial_variance : {0.0f, 0.04f}) {
        auto p = kParameters;
        p.initial_variance = initial_variance;
        check_qe_kernel<<<kPaths / 256U, 256U>>>(p, device);
        check_cuda(cudaGetLastError(), "QE launch");
        check_cuda(cudaMemcpy(results.data(), device, kPaths * sizeof(Result),
                             cudaMemcpyDeviceToHost), "QE copy");
        double sum = 0.0, squares = 0.0, spot = 0.0, spot_squares = 0.0;
        for (unsigned int i = 0; i < kPaths; ++i) {
            const auto& r = results[i];
            require(r.identical && r.streams_ok, "QE copula or Philox isolation broken");
            // Inlined spot arithmetic may contract differently at two call sites.
            require(std::abs(r.spot_difference) < 2e-7f * (1.0f + std::abs(r.log_spot)),
                    "Identical Brownian paths produced different Heston spots");
            require(r.variance == r.bates_variance, "Bates changed correlated QE variance");
            const double z = -12.0 + double(i % 97U) * 0.25;
            const double reference = std::log(0.5 * std::erfc(z / std::sqrt(2.0)));
            require(std::abs(r.log_tail - reference) < 1.5e-5,
                    "Gaussian log survival differs from double erfc");
            sum += r.variance;
            squares += double(r.variance) * r.variance;
            const double s = std::exp(double(r.log_spot));
            spot += s;
            spot_squares += s * s;
        }
        const double mean = p.theta + (p.initial_variance - p.theta)
            * std::exp(-double(p.kappa) * kDt);
        const double empirical = sum / kPaths;
        const double se = std::sqrt((squares / kPaths - empirical * empirical) / kPaths);
        require(std::abs(empirical - mean) < 6.0 * se + 2e-8,
                "QE marginal mean changed");
        const double average_spot = spot / kPaths;
        const double spot_se = std::sqrt((spot_squares / kPaths
            - average_spot * average_spot) / kPaths);
        require(std::abs(average_spot - std::exp(double(p.risk_free_rate
                - p.dividend_yield) * kDt)) < 6.0 * spot_se + 2e-7,
                "Correlated QE-M lost its spot martingale");
    }
    check_cuda(cudaFree(device), "QE free");
    float2* tails = nullptr;
    check_cuda(cudaMalloc(&tails, 97U * sizeof(float2)), "tail allocate");
    check_qe_tail_kernel<<<1,97>>>(kParameters, tails);
    std::vector<float2> values(97U);
    check_cuda(cudaMemcpy(values.data(), tails, 97U * sizeof(float2),
                         cudaMemcpyDeviceToHost), "tail copy");
    const double decay = std::exp(-double(kParameters.kappa) * kDt);
    const double mean = kParameters.theta * (1.0 - decay);
    const double psi = double(kParameters.gamma) * kParameters.gamma
        / (2.0 * kParameters.kappa * kParameters.theta);
    const double q = 2.0 / (psi + 1.0);
    for (unsigned int i = 0; i < 97U; ++i) {
        const double z = -12.0 + i * 0.25;
        const double tail = 0.5 * std::erfc(z / std::sqrt(2.0));
        const double reference = tail >= q ? 0.0 : std::log(q / tail) * mean / q;
        // FP32 conditional moments inherit the existing variance-decay rounding.
        require(std::isfinite(values[i].x)
            && std::abs(values[i].x - reference) < 2e-4 * reference + 1e-8,
            "Correlated QE tail quantile differs from independent inverse CDF");
    }
    check_cuda(cudaFree(tails), "tail free");
}
