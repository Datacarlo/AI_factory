#include "correlation/dataset.hpp"
#include "multi_asset/rate_adjusted_model_group.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"
#include "model/fixed_income/vasicek/correlated_dynamics.cuh"
#include "model/fixed_income/constant_rate/correlated_dynamics.cuh"
#include "product/basket_down_and_out_call/pricing_policy.cuh"
#include "product/worst_of_call/pricing_policy.cuh"
#include "product/best_of_call/pricing_policy.cuh"
#include "product/basket_call/pricing_policy.cuh"
#include "tools/pricing/cross_asset_price_generation.cuh"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace wb=ai_factory::workbench;
using BS=wb::model::equity::black_scholes::DynamicsPolicy;
using Rate=wb::model::fixed_income::vasicek::CorrelatedRatePolicy;
using Rough=wb::model::equity::quadratic_rough_heston::CausalFftPathPolicy;
using Worst=wb::product::WorstOfCallPathPolicy;
using Best=wb::product::BestOfCallPathPolicy;
using Basket=wb::product::BasketCallPathPolicy;
using Barrier=wb::product::BasketDownAndOutCallPathPolicy;
using Expression=wb::RowIndexExpression;
using TwoBS=wb::simulation::MultiAssetPack<
    wb::simulation::RateAdjustedModelGroup<BS,2U>>;
using OneBS=wb::simulation::MultiAssetPack<
    wb::simulation::RateAdjustedModelGroup<BS,1U>>;
using NoRough=wb::simulation::MultiRoughPack<>;
using OneRough=wb::simulation::MultiRoughPack<Rough>;

void require(bool ok,const char* message) {
    if(!ok) throw std::runtime_error(message);
}

wb::CompiledRowIndexPlan single_plan(unsigned slots) {
    std::vector<Expression> sources;
    for(unsigned i=0U;i<slots;++i)
        sources.push_back(Expression::source(i,1U));
    return wb::compile_row_index_plan(Expression::aligned(sources),slots);
}

void markov_rate() {
    wb::offline::pricing::MultiAssetHostModels<TwoBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.0f}};
    wb::offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<Rate::Parameters> rates{{{0.7f,0.025f,0.035f},0.02f}};
    const std::vector<std::string> factors{
        "a.spot","b.spot","rate.brownian"};
    const std::vector<wb::correlation::MatrixRow> matrices{{
        "correlated",factors,wb::correlation::packed_cholesky({
            1.0,0.2,0.4,0.2,1.0,-0.3,0.4,-0.3,1.0},3U)}};
    const auto plan=single_plan(5U);
    const auto martingale=wb::offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Worst>(equities,rough,rates,
            {{0.0f,32U}},matrices,factors,factors,plan,
            2048U,2048U,1.0f/252.0f,1U,32U,47511U);
    require(std::abs(martingale.prices[0]-1.0f)<4e-5f,
        "discounted zero-volatility basket is not a martingale");
    equities.models[0].volatility=0.25f;
    const auto one=wb::offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Barrier,1U>(equities,rough,rates,
            {{0.8f,0.72f,32U,4U}},matrices,factors,factors,plan,
            1024U,1024U,1.0f/252.0f,1U,32U,4511U);
    const auto four=wb::offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Barrier,4U>(equities,rough,rates,
            {{0.8f,0.72f,32U,4U}},matrices,factors,factors,plan,
            1024U,1024U,1.0f/252.0f,1U,32U,4511U);
    require(std::abs(one.prices[0]-four.prices[0])<2e-6f
        &&std::abs(one.standard_errors[0]-four.standard_errors[0])<2e-6f,
        "rate basket depends on path team size");
    const auto initially_out=wb::offline::pricing::execute_cross_asset_prices<
        TwoBS,NoRough,Rate,Barrier>(equities,rough,rates,
            {{0.0f,1.0f,32U,4U}},matrices,factors,factors,plan,
            1024U,1024U,1.0f/252.0f,1U,32U,4511U);
    require(initially_out.prices[0]==0.0f
        &&initially_out.standard_errors[0]==0.0f,
        "basket barrier touched at inception must knock out");
}

void rough_rate() {
    wb::offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.0f,0.0f,0.0f,0.2f}};
    wb::offline::pricing::MultiRoughHostModels<OneRough> rough;
    rough.models={{1.0f,0.0f,0.0f,0.05f,0.2f,0.0f,
                   0.04f,1.0f,0.3f,0.12f}};
    const std::vector<Rate::Parameters> rates{{{0.7f,0.025f,0.035f},0.02f}};
    const std::vector<std::string> external{
        "bs.spot","quadratic_rough_heston.spot","rate.brownian"};
    const std::vector<std::string> internal{
        "quadratic_rough_heston.spot","bs.spot","rate.brownian"};
    const wb::correlation::MatrixRow raw{
        "quadratic_rough_rate",external,wb::correlation::packed_cholesky({
            1.0,0.2,0.0,
            0.2,1.0,-0.1,
            0.0,-0.1,1.0},3U)};
    const auto plan=single_plan(5U);
    const auto full=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Barrier,1U>(equities,rough,rates,
            {{0.85f,0.75f,32U,4U}},{raw},external,internal,plan,
            512U,512U,1.0f/252.0f,1U,32U,4511U);
    const auto chunked=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Barrier,4U>(equities,rough,rates,
            {{0.85f,0.75f,32U,4U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    require(std::isfinite(full.prices[0]) && full.prices[0]>=0.0f,
        "rough and rate basket returned invalid price");
    require(std::abs(full.prices[0]-chunked.prices[0])<2e-6f
        &&std::abs(full.standard_errors[0]-chunked.standard_errors[0])<2e-6f,
        "rough and rate basket depends on path chunk or team size");

    const auto worst=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Worst,4U>(equities,rough,rates,
            {{1.0f,32U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    const auto basket=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Basket,4U>(equities,rough,rates,
            {{1.0f,32U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    const auto best=wb::offline::pricing::execute_cross_asset_prices<
        OneBS,OneRough,Rate,Best,4U>(equities,rough,rates,
            {{1.0f,32U}},{raw},external,internal,plan,
            512U,128U,1.0f/252.0f,1U,32U,4511U);
    require(std::isfinite(worst.prices[0]) && worst.prices[0]>=0.0f
        &&worst.prices[0]<=basket.prices[0]
        &&basket.prices[0]<=best.prices[0],
        "rough/rate terminal basket payoffs are misordered");
}

// Independent deterministic price: refining the grid must not lengthen time.
void business_day_clock() {
    using ConstantRate=wb::model::fixed_income::constant_rate::CorrelatedRatePolicy;
    wb::offline::pricing::MultiAssetHostModels<OneBS> equities;
    equities.models={{1.7f,0.0f,0.01f,0.0f}};
    wb::offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> factors{"equity.spot"};
    const std::vector<wb::correlation::MatrixRow> matrices{{"one",factors,{1.0f}}};
    const auto plan=single_plan(4U);
    for (const unsigned days : {42U,252U,1512U}) {
        const double years=double(days)/252.0;
        const double expected=std::exp(-0.01*years)-0.8*std::exp(-0.05*years);
        for (const unsigned steps_per_day : {1U,2U,4U}) {
            const auto run=wb::offline::pricing::execute_cross_asset_prices<
                OneBS,NoRough,ConstantRate,Worst>(equities,rough,{{0.05f}},
                    {{0.8f,days}},matrices,factors,factors,plan,
                    128U,128U,1.0f/(252.0f*steps_per_day),steps_per_day,32U,4511U);
            std::printf("clock days=%u steps_per_day=%u price=%.9g expected=%.12g error=%.9g\n",
                days,steps_per_day,run.prices[0],expected,double(run.prices[0])-expected);
            std::fflush(stdout);
            // Thousands of FP32 spot/rate updates accumulate rounding. This
            // clock check distinguishes horizons (percent-level errors), not
            // exact analytical pricing from a discretized floating-point path.
            const double tolerance=5e-5*std::max(1.0,years);
            require(std::abs(double(run.prices[0])-expected)<tolerance,
                "calendar horizon or discount depends on steps per day");
            require(run.standard_errors[0]<1e-6f,"deterministic path has sampling noise");
        }
    }
    // New contracts can monitor every half-day without changing historical day calendars.
    const Barrier::ProductParameters half_day{0.8f,0.7f,252U,0U,1U};
    const Barrier::ProductParameters daily{0.8f,0.7f,252U,1U,0U};
    require(Barrier::valid_parameters(half_day,2U),"half-day monitoring rejected");
    require(!Barrier::valid_parameters({0.8f,0.7f,252U,1U,1U},2U),"ambiguous monitoring accepted");
    for(unsigned step=0;step<504U;++step) {
        require(Barrier::is_observation_step(step,504U,2U,half_day),"half-day observation missed");
        require(Barrier::is_observation_step(step,504U,2U,daily)==((step+1U)%2U==0U),"daily calendar changed");
    }
    // The relabelled barrier must observe exactly the same simulation steps.
    const Barrier::ProductParameters old_product{0.8f,0.7f,252U,1U};
    const Barrier::ProductParameters new_product{0.8f,0.7f,504U,2U};
    for (unsigned step=0;step<504U;++step)
        require(Barrier::is_observation_step(step,504U,2U,old_product)
            ==Barrier::is_observation_step(step,504U,1U,new_product),
            "barrier relabelling changes a monitoring date");
}

int main() {
    business_day_clock();
    markov_rate();
    rough_rate();
}
