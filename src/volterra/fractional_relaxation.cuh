// Deterministic fractional mean relaxation, evaluated during grid preparation.
#pragma once

#include <cuda_runtime.h>
#include <cmath>

namespace ai_factory::workbench::volterra {

// E_alpha(-x), 0 < alpha <= 1 and x >= 0. The small-x series uses double
// precision. For larger x, Pollard's positive spectral representation uses
// its inverse CDF z(u)=sin(pi*alpha*u)/sin(pi*alpha*(1-u)). Substituting u=t^4
// resolves the endpoint contribution; composite Gauss quadrature is refined
// before the final FP32 cast. This function must not run inside path loops.
__host__ __device__ inline float fractional_relaxation(double alpha, double x) {
    if (x == 0.0) return 1.0f;
    if (alpha == 1.0) return static_cast<float>(exp(-x));
    if (x <= 2.0) {
        double sum = 1.0;
        double power = 1.0;
        for (int order = 1; order < 128; ++order) {
            power *= -x;
            const double term = power / tgamma(alpha * order + 1.0);
            sum += term;
            if (order > 12 && fabs(term) < 1.0e-14 * fabs(sum)) break;
        }
        return static_cast<float>(sum);
    }
    constexpr double nodes[16] = {
        -0.98940093499164994, -0.9445750230732326,
        -0.86563120238783176, -0.755404408355003,
        -0.61787624440264377, -0.45801677765722737,
        -0.28160355077925892, -0.095012509837637441,
        0.095012509837637441, 0.28160355077925892,
        0.45801677765722737, 0.61787624440264377,
        0.755404408355003, 0.86563120238783176,
        0.9445750230732326, 0.98940093499164994,
    };
    constexpr double weights[16] = {
        0.027152459411754176, 0.062253523938647456,
        0.095158511682492605, 0.12462897125553407,
        0.14959598881657671, 0.16915651939500265,
        0.18260341504492364, 0.18945061045506864,
        0.18945061045506864, 0.18260341504492364,
        0.16915651939500265, 0.14959598881657671,
        0.12462897125553407, 0.095158511682492605,
        0.062253523938647456, 0.027152459411754176,
    };

    constexpr double pi = 3.1415926535897932384626433832795;
    double previous = -1.0;
    for (int panels = 1; panels <= 512; panels *= 2) {
        double sum = 0.0;
        for (int panel = 0; panel < panels; ++panel) {
            for (int point = 0; point < 16; ++point) {
                const double t = (panel + 0.5 * (nodes[point] + 1.0)) / panels;
                const double u = t * t * t * t;
                const double z = sin(pi * alpha * u) / sin(pi * alpha * (1.0-u));
                sum += weights[point] * 4.0 * t * t * t
                    * exp(-pow(x*z, 1.0/alpha)) / (2.0 * panels);
            }
        }
        if (previous >= 0.0 && fabs(sum-previous) < 1.0e-10 * fmax(sum,1.0e-8))
            return static_cast<float>(sum);
        previous = sum;
    }
    return static_cast<float>(previous);
}

}  // namespace ai_factory::workbench::volterra
