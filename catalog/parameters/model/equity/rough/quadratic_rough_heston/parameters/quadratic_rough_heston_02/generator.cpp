#include "datasets/validation.hpp"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <stdexcept>

#include <array>
#include <bit>
#include <cmath>
#include <numeric>
#include <vector>

// Match the two NumPy PCG64 streams used to qualify this Latin hypercube.
using Word = unsigned __int128;
constexpr Word word_pair(std::uint64_t hi, std::uint64_t lo) { return (Word(hi) << 64) | lo; }
struct Stream {
    Word state, increment;
    bool buffered = false;
    std::uint32_t upper = 0;
    std::uint64_t draw() {
        state = state * word_pair(0x2360ed051fc65da4ULL, 0x4385df649fccf645ULL) + increment;
        return std::rotr(std::uint64_t(state >> 64) ^ std::uint64_t(state), int(state >> 122));
    }
    std::uint32_t next32() {
        if (buffered) { buffered = false; return upper; }
        const auto value = draw();
        upper = value >> 32; buffered = true;
        return std::uint32_t(value);
    }
    unsigned index(unsigned hi) {
        const unsigned mask = std::bit_ceil(hi + 1U) - 1U;
        unsigned v;
        do { v = next32() & mask; } while (v > hi);
        return v;
    }
};

int main() {
    using namespace ai_factory::workbench;
    using namespace datasets;
    GeneratedRows rows;
    Stream streams[] = {{word_pair(0x1c57873afbc98472ULL, 0x7277fbf1226aca3bULL), word_pair(0x1bced29ea57a49a2ULL, 0xe74a40f4210aafe1ULL)},
        {word_pair(0xbe7a9091049a65dULL, 0x1d0f0839fd1a40bULL), word_pair(0x6bcc11f4c7dc37cbULL, 0x6e263d695f69371bULL)}};
    for (int regime = 0; regime < 2; ++regime) {
        const bool stress = regime == 1;
        const unsigned count = stress ? 100 : 900;
        auto& rng = streams[regime];
        std::vector<std::array<double,9>> u(count);
        for (auto& row : u) for (double& v : row) v = (rng.draw() >> 11) * 0x1.0p-53;
        for (unsigned j=0; j<9; ++j) {
            std::vector<unsigned> perm(count);
            std::iota(perm.begin(),perm.end(),1U);
            for (unsigned i=count-1; i>0; --i) std::swap(perm[i],perm[rng.index(i)]);
            for (unsigned i=0; i<count; ++i) u[i][j]=(perm[i]-u[i][j])/count;
        }
        for (unsigned i=0; i<count; ++i) {
            const auto [v,u0,scale,h,rate,eta,pos,r,q]=u[i];
            const double sigma=stress ? .09+.31*v : .103+.275*v;
            const double fraction=stress ? .12+.63*u0 : .15+.50*u0;
            const double a=stress ? .12+.30*scale : .14+.22*scale;
            const double offset=sigma*std::sqrt(fraction/a);
            const bool upper=stress && i%3==0;
            const double b=offset*((upper?.35:.85)+.25*pos);
            rows.rows.push_back({{"spot",1.f},{"risk_free_rate",float(stress?-.03+.15*r:.001+.079*r)},
                {"dividend_yield",float(stress?.1*q:.06*q)},{"initial_feedback",float(upper?b+offset:b-offset)},
                {"quadratic_scale",float(a)},{"quadratic_shift",float(b)},{"variance_floor",float(sigma*sigma*(1-fraction))},
                {"feedback_rate",float(stress?.32+.63*rate:.42+.45*rate)},
                {"feedback_volatility",float(stress?.55+.53*eta:.65+.37*eta)},
                {"hurst_exponent",float(stress?.04+.20*h:.06+.14*h)}});
        }
    }
    rows.construction = {{"method","Latin hypercube in initial volatility, feedback fraction, scale, H, rate, vol of feedback, center, risk-free rate and dividend yield"},
                         {"core_rows",900},{"stress_rows",100},{"seeds",{261126,261127}}};
    const std::filesystem::path dataset =
        "datasets/parameters/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_02.json";
    write_model_dataset(
        "quadratic_rough_heston_02", "quadratic rough Heston", dataset,
        "catalog/parameters/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_02/generation.yaml",
        "https://datasets.ai-factory.example/v1/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_02.json",
        {
            {"spot", "Initial spot; fixed to 1."},
            {"risk_free_rate", "Continuously compounded risk-free rate."},
            {"dividend_yield", "Continuously compounded dividend yield."},
            {"initial_feedback", "Initial feedback factor Z0."},
            {"quadratic_scale", "Positive coefficient a."},
            {"quadratic_shift", "Quadratic center b."},
            {"variance_floor", "Strictly positive floor c."},
            {"feedback_rate", "Feedback rate lambda."},
            {"feedback_volatility", "Feedback volatility eta."},
            {"hurst_exponent", "Fractional roughness H."},
        },
        {
            {"variance", "V=a(Z-b)^2+c."},
            {"feedback", "Z=Z0-lambda K*Z dt+lambda eta K*sqrt(V)dW."},
            {"spot", "dS/S=(r-q)dt+sqrt(V)dW with the same W."},
            {"simulation", "Positive exponential N-factor lift."},
        },
        rows
    );
    validate_model_dataset_file(dataset);
    if (model::equity::quadratic_rough_heston::load_models(dataset).size()
        != 1000U) {
        throw std::runtime_error("Quadratic rough-Heston reload failed.");
    }
}
