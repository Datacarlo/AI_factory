// Regression for the published quadratic rough-Heston samples_02 path 77.
#include "cuda/check_cuda.cuh"
#include <sample.cuh>
#include "model/equity/rough/quadratic_rough_heston/dynamics_impl.cuh"
#include "model/equity/rough/quadratic_rough_heston/sample_device_preparation.cuh"
#include "model/equity/rough/quadratic_rough_heston/markovian_n_factor_preparation.hpp"
#include "model/equity/rough/quadratic_rough_heston/sample.cuh"
#include "tools/sampling/host_philox.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace {

using namespace ai_factory::workbench;
namespace quadratic = model::equity::quadratic_rough_heston;
namespace sampling = offline::sampling;

// Version-1 recipe domain from capability_manifest.py.
constexpr std::uint64_t kParameterSeed = 11668828502827728896ULL;
constexpr std::uint64_t kDynamicsSeed = 11668828504975212544ULL;
constexpr std::size_t kPublishedParameterIndex = 76U;
constexpr std::uint32_t kPublishedMaturityDays = 173U;
constexpr float kSampleDt = 1.0f / 504.0f;

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

quadratic::ModelParameters published_parameter_row() {
    sampling::HostUniformSequence uniforms(
        kParameterSeed,
        kPublishedParameterIndex
    );
    return {
        sampling::uniform({1.0f, 1.0f}, uniforms),
        sampling::uniform({0.001f, 0.08f}, uniforms),
        sampling::uniform({0.0f, 0.06f}, uniforms),
        sampling::uniform({0.03f, 0.2f}, uniforms),
        sampling::uniform({0.1f, 0.8f}, uniforms),
        sampling::uniform({0.02f, 0.2f}, uniforms),
        sampling::uniform({0.0005f, 0.02f}, uniforms),
        sampling::uniform({0.3f, 3.0f}, uniforms),
        sampling::uniform({0.3f, 2.0f}, uniforms),
        sampling::uniform({0.01f, 0.2f}, uniforms),
    };
}

std::vector<quadratic::ModelParameters> domain_corner_models() {
    return {
        {1.0f, 0.001f, 0.06f, 0.03f, 0.10f, 0.02f,
         0.0005f, 0.30f, 0.30f, 0.01f},
        {1.0f, 0.08f, 0.0f, 0.20f, 0.80f, 0.20f,
         0.02f, 3.0f, 2.0f, 0.20f},
        {1.0f, -0.03f, 0.10f, -0.30f, 0.02f, -0.20f,
         0.0001f, 0.05f, 0.05f, 0.005f},
        {1.0f, 0.12f, 0.0f, 0.50f, 2.0f, 0.50f,
         0.10f, 6.0f, 4.0f, 0.45f},
    };
}

template<std::size_t FactorCount>
void exercise_domain_corners() {
    constexpr std::size_t paths_per_parameter = 1'024U;
    constexpr std::uint32_t maturity_days = 504U;
    const auto models = domain_corner_models();
    const auto prepared = quadratic::prepare_dynamics<FactorCount>(
        models,
        2.0f,
        kSampleDt
    );
    const std::size_t sample_count = models.size() * paths_per_parameter;
    quadratic::PreparedDynamics<FactorCount>* device_prepared = nullptr;
    float* device_spots = nullptr;
    check_cuda(
        cudaMalloc(&device_prepared, prepared.size() * sizeof(prepared[0])),
        "quadratic rough-Heston domain prepared allocation"
    );
    check_cuda(
        cudaMalloc(&device_spots, sample_count * sizeof(float)),
        "quadratic rough-Heston domain output allocation"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            prepared.data(),
            prepared.size() * sizeof(prepared[0]),
            cudaMemcpyHostToDevice
        ),
        "quadratic rough-Heston domain prepared copy"
    );
    quadratic::launch_quadratic_rough_heston_terminal_samples_cuda<
        FactorCount
    >(
        device_prepared,
        models.size(),
        paths_per_parameter,
        maturity_days,
        0U,
        sample_count,
        256U,
        models.size(),
        11668828504975212544ULL,
        device_spots
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "quadratic rough-Heston domain synchronize"
    );
    std::vector<float> spots(sample_count);
    check_cuda(
        cudaMemcpy(
            spots.data(),
            device_spots,
            spots.size() * sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "quadratic rough-Heston domain output copy"
    );
    for (const float spot : spots) {
        require(
            std::isfinite(spot) && spot >= 0.0f,
            "Quadratic rough-Heston domain sweep produced a non-finite spot."
        );
    }
    check_cuda(
        cudaFree(device_spots),
        "quadratic rough-Heston domain output free"
    );
    check_cuda(
        cudaFree(device_prepared),
        "quadratic rough-Heston domain prepared free"
    );
}

void check_generated_shared_fit() {
    using Dynamics = quadratic::DynamicsPolicy<7U>;
    using Policy = sample::ModelSamplingPolicy<
        simulation::FixedStepTerminalSchedule<Dynamics>,
        sample::SpotSampleObservation<Dynamics>>;
    constexpr std::size_t rows = 2U;
    constexpr std::size_t paths_per_parameter = 250U;
    constexpr std::size_t samples = rows * paths_per_parameter;
    const quadratic::SampleParameterSampler::Configuration bounds{{
        {1.0f, 1.0f}, {0.03f, 0.20f}, {0.1f, 0.8f},
        {0.02f, 0.20f}, {0.0005f, 0.015f}, {0.1f, 0.8f},
        {0.1f, 0.6f}, {0.05f, 0.20f}}};
    quadratic::ModelParameters *fused_parameters = nullptr,
                               *split_parameters = nullptr;
    std::uint32_t *fused_days = nullptr, *split_days = nullptr;
    float *fused_spots = nullptr, *split_spots = nullptr;
    check_cuda(cudaMalloc(&fused_parameters,
        rows * sizeof(quadratic::ModelParameters)),
        "quadratic fused parameters allocation");
    check_cuda(cudaMalloc(&split_parameters,
        rows * sizeof(quadratic::ModelParameters)),
        "quadratic split parameters allocation");
    check_cuda(cudaMalloc(&fused_days,
        samples * sizeof(std::uint32_t)),
        "quadratic fused days allocation");
    check_cuda(cudaMalloc(&split_days,
        samples * sizeof(std::uint32_t)),
        "quadratic split days allocation");
    check_cuda(cudaMalloc(&fused_spots,
        samples * sizeof(float)),
        "quadratic fused samples allocation");
    check_cuda(cudaMalloc(&split_spots,
        samples * sizeof(float)),
        "quadratic split samples allocation");

    constexpr std::uint64_t parameter_seed = 4920011U;
    constexpr std::uint64_t schedule_seed = 4920012U;
    constexpr std::uint64_t dynamics_seed = 4920013U;
    sample::launch_prepared_samples_cuda<Policy>(
        sample::GeneratedPreparedInputSource<
            quadratic::SampleParameterSampler,
            quadratic::SampleDevicePreparer<7U>>{
            bounds, {2.0f, sample::kSampleFixedStepDt}, fused_parameters},
        sample::UniformMaturityCalendarSource{{2U, 4U}, fused_days},
        {rows, paths_per_parameter, 0U, samples},
        sample::kFixedStepSampleTimeConfiguration, {256U, rows},
        {parameter_seed, schedule_seed, dynamics_seed},
        {fused_spots}, "quadratic_rough_heston.samples",
        "generated.fused.reference", "quadratic fused reference");
    quadratic::launch_quadratic_rough_heston_generated_random_terminal_samples_cuda<7U>(
        split_parameters, bounds, rows, paths_per_parameter,
        2U, 4U, 0U, samples, 256U, rows,
        parameter_seed, schedule_seed, dynamics_seed,
        split_days, split_spots);
    check_cuda(cudaDeviceSynchronize(), "quadratic generated replay");
    std::vector<quadratic::ModelParameters> a_parameters(rows),
                                             b_parameters(rows);
    std::vector<std::uint32_t> a_days(samples), b_days(samples);
    std::vector<float> a_spots(samples), b_spots(samples);
    check_cuda(cudaMemcpy(a_parameters.data(), fused_parameters,
        rows * sizeof(quadratic::ModelParameters), cudaMemcpyDeviceToHost),
        "quadratic fused parameters copy");
    check_cuda(cudaMemcpy(b_parameters.data(), split_parameters,
        rows * sizeof(quadratic::ModelParameters), cudaMemcpyDeviceToHost),
        "quadratic split parameters copy");
    check_cuda(cudaMemcpy(a_days.data(), fused_days,
        samples * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
        "quadratic fused days copy");
    check_cuda(cudaMemcpy(b_days.data(), split_days,
        samples * sizeof(std::uint32_t), cudaMemcpyDeviceToHost),
        "quadratic split days copy");
    check_cuda(cudaMemcpy(a_spots.data(), fused_spots,
        samples * sizeof(float), cudaMemcpyDeviceToHost),
        "quadratic fused samples copy");
    check_cuda(cudaMemcpy(b_spots.data(), split_spots,
        samples * sizeof(float), cudaMemcpyDeviceToHost),
        "quadratic split samples copy");
    require(std::memcmp(a_parameters.data(), b_parameters.data(),
        rows * sizeof(quadratic::ModelParameters)) == 0,
        "separate quadratic lift changed model parameters");
    require(a_days == b_days,
        "separate quadratic lift changed maturity days");
    require(std::memcmp(a_spots.data(), b_spots.data(),
        samples * sizeof(float)) == 0,
        "separate quadratic lift changed samples");
    check_cuda(cudaFree(fused_parameters), "quadratic fused parameters free");
    check_cuda(cudaFree(split_parameters), "quadratic split parameters free");
    check_cuda(cudaFree(fused_days), "quadratic fused days free");
    check_cuda(cudaFree(split_days), "quadratic split days free");
    check_cuda(cudaFree(fused_spots), "quadratic fused samples free");
    check_cuda(cudaFree(split_spots), "quadratic split samples free");
}

}  // namespace

int main() {
    int device_count = 0;
    const cudaError_t availability = cudaGetDeviceCount(&device_count);
    if (availability == cudaErrorNoDevice
        || availability == cudaErrorInsufficientDriver
        || device_count == 0) return 77;
    check_cuda(availability, "quadratic rough-Heston sample cudaGetDeviceCount");

    const quadratic::ModelParameters model = published_parameter_row();
    require(
        model.risk_free_rate > 0.0731f
            && model.risk_free_rate < 0.0733f
            && model.initial_feedback > 0.1534f
            && model.initial_feedback < 0.1537f,
        "Published quadratic rough-Heston row 77 no longer matches its seed."
    );
    const auto prepared = quadratic::prepare_dynamics<7U>(
        model,
        2.0f,
        kSampleDt
    );

    quadratic::PreparedDynamics<7U>* device_prepared = nullptr;
    float* device_first = nullptr;
    float* device_replay = nullptr;
    check_cuda(
        cudaMalloc(&device_prepared, sizeof(prepared)),
        "quadratic rough-Heston prepared allocation"
    );
    check_cuda(
        cudaMalloc(&device_first, sizeof(float)),
        "quadratic rough-Heston first output allocation"
    );
    check_cuda(
        cudaMalloc(&device_replay, sizeof(float)),
        "quadratic rough-Heston replay output allocation"
    );
    check_cuda(
        cudaMemcpy(
            device_prepared,
            &prepared,
            sizeof(prepared),
            cudaMemcpyHostToDevice
        ),
        "quadratic rough-Heston prepared copy"
    );

    // The sample engine adds the parameter index to the dynamics seed.  With
    // one local prepared row, pass the resulting published row key directly.
    constexpr std::uint64_t row_dynamics_seed =
        kDynamicsSeed + kPublishedParameterIndex;
    quadratic::launch_quadratic_rough_heston_terminal_samples_cuda<7U>(
        device_prepared,
        1U,
        1U,
        kPublishedMaturityDays,
        0U,
        1U,
        128U,
        1U,
        row_dynamics_seed,
        device_first
    );
    quadratic::launch_quadratic_rough_heston_terminal_samples_cuda<7U>(
        device_prepared,
        1U,
        1U,
        kPublishedMaturityDays,
        0U,
        1U,
        256U,
        1U,
        row_dynamics_seed,
        device_replay
    );
    check_cuda(
        cudaDeviceSynchronize(),
        "quadratic rough-Heston row 77 synchronize"
    );

    float first = 0.0f;
    float replay = 0.0f;
    check_cuda(
        cudaMemcpy(
            &first,
            device_first,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "quadratic rough-Heston first output copy"
    );
    check_cuda(
        cudaMemcpy(
            &replay,
            device_replay,
            sizeof(float),
            cudaMemcpyDeviceToHost
        ),
        "quadratic rough-Heston replay output copy"
    );
    require(
        std::isfinite(first) && first >= 0.0f,
        "Published quadratic rough-Heston sample row 77 is not finite."
    );
    require(
        std::memcmp(&first, &replay, sizeof(float)) == 0,
        "Quadratic rough-Heston row 77 changed with launch geometry."
    );

    exercise_domain_corners<2U>();
    exercise_domain_corners<3U>();
    exercise_domain_corners<7U>();
    check_generated_shared_fit();

    check_cuda(
        cudaFree(device_replay),
        "quadratic rough-Heston replay output free"
    );
    check_cuda(
        cudaFree(device_first),
        "quadratic rough-Heston first output free"
    );
    check_cuda(
        cudaFree(device_prepared),
        "quadratic rough-Heston prepared free"
    );
}
