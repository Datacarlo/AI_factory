// Affine observation cache: pathwise parity, strided reuse.
#include "product/zero_coupon_bond_up_and_out/prepared_bond_policy.cuh"
#include "monte_carlo/block_preparation_workspace.cuh"
#include "model/fixed_income/cir/analytics_impl.cuh"
#include "model/fixed_income/cir/dynamics_impl.cuh"
#include "model/fixed_income/g2/analytics_impl.cuh"
#include "model/fixed_income/g2/dynamics_impl.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace wb = ai_factory::workbench;
namespace fi = wb::model::fixed_income;

template<class Base, class Cached>
__global__ void cache_parity_kernel(
    typename Base::ModelParameters model,
    wb::monte_carlo::BlockPreparationWorkspace workspace,
    unsigned steps, unsigned* mismatches
) {
    __shared__ typename Base::PreparedRow reference;
    __shared__ typename Cached::PreparedRow cached;
    // Reuse each physical block's table for several different contracts.
    for (unsigned row = blockIdx.x + 3U; row < 10U; row += gridDim.x) {
        if (threadIdx.x == 0U) {
            const wb::product::ZeroCouponBondUpAndOutParameters product{
                1.0f, 0.8f, row % 2U ? 0.98f : 1.5f, 21U * row, 252U * 2U};
            const typename Base::TimeConfiguration time{1.0f / (252.0f * steps), steps};
            reference = Base::prepare_row(model, product, time);
            cached = Cached::prepare_row(model, product, time);
        }
        __syncthreads();
        Cached::prepare_block(cached,
            workspace.block(blockIdx.x), workspace.stride);
        const auto key = wb::philox::make_key(20261119ULL + row);
        for (unsigned path = threadIdx.x; path < 1024U; path += blockDim.x) {
            const float a = Base::evaluate_path(reference, key, path);
            const float b = Cached::evaluate_path(cached, key, path);
            if (__float_as_uint(a) != __float_as_uint(b)) atomicAdd(mismatches, 1U);
        }
        __syncthreads();
    }
}

template<class Dynamics, class Analytics>
void check(const typename Dynamics::Parameters& model) {
    using Base = wb::product::ZeroCouponBondUpAndOutPricingPolicy<
        wb::simulation::FixedStepDenseSchedule<Dynamics>>;
    using Cached = wb::product::PreparedBondBarrierPricingPolicy<Base, Analytics>;
    const wb::product::ZeroCouponBondUpAndOutParameters longest{1.f, .8f, 1.5f, 189U, 504U};
    unsigned* differences = nullptr;
    wb::check_cuda(cudaMalloc(reinterpret_cast<void**>(&differences), sizeof(unsigned)), "allocate differences");
    for (unsigned steps : {1U, 2U, 4U}) {
        const auto bytes = Cached::block_workspace_bytes(&longest, 1U, steps);
        wb::monte_carlo::BlockPreparationStorage storage(2U, bytes);
        for (unsigned threads : {128U, 256U}) {
            wb::check_cuda(cudaMemset(differences, 0, sizeof(unsigned)), "reset differences");
            cache_parity_kernel<Base, Cached><<<2U, threads>>>(model, storage.view(), steps, differences);
            wb::check_cuda(cudaGetLastError(), "cache parity launch");
            unsigned count = 0U;
            wb::check_cuda(cudaMemcpy(&count, differences, sizeof(count), cudaMemcpyDeviceToHost), "read differences");
            if (count != 0U) throw std::runtime_error("Bond cache changes path payoff bits.");
        }
    }
    wb::check_cuda(cudaFree(differences), "free differences");
}

int main() {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) return 77;
    try {
        check<fi::cir::joint::DynamicsPolicy, fi::cir::BermudanSwaptionAnalyticsPolicy>(
            fi::cir::ModelParameters{{0.6f, 0.04f, 0.3f}, 0.03f});
        check<fi::g2::joint::DynamicsPolicy, fi::g2::BermudanSwaptionAnalyticsPolicy>(
            fi::g2::ModelParameters{{0.1f, 0.01f, 0.3f, 0.015f, -0.5f}, {0.02f, 0.01f}});
        std::cout << "Bond cache pathwise parity passed.\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
