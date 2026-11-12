// Kernel-dependent time data cached once per grid, never computed per path.
#pragma once
#include <cuda_runtime.h>
namespace ai_factory::workbench::volterra {
// Existing per-time buffers carry Var(Y) for log-volatility models and the
// deterministic mean response for the Gaussian resolvent. Kernel keys suffice
// for both, so X0/theta/nu bumps share the same cached array.
template<typename Kernel, bool UsesVariance = true>
__host__ __device__ inline float path_time_coefficient(
    const typename Kernel::PreparedKernel& kernel, float time
) {
    if constexpr (requires { Kernel::mean_response(kernel, time); })
        return Kernel::mean_response(kernel, time);
    else if constexpr (UsesVariance)
        return Kernel::volterra_variance(kernel, time);
    else
        return 0.0f;
}
} // namespace ai_factory::workbench::volterra
