// American compositions: state normalization, CRN replay, contracts and row mapping.
#include "longstaff_schwartz/price_gradients/device_prepared_frozen_exercise_replay.cuh"
#include <iostream>
#include <cmath>
#include <algorithm>
#include "model/equity/markovian/sabr/product/american_option.cuh"
#include "model/equity/markovian/sabr/product/american_option_price_delta.cuh"
#include "model/equity/markovian/sabr/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/sabr/product/american_option_continuation.cuh"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"
#include "model/equity/markovian/sabr/price_gradients/coupled_dynamics_impl.cuh"
#include "model/equity/markovian/heston_3_2/product/american_option.cuh"
#include "model/equity/markovian/heston_3_2/product/american_option_price_delta.cuh"
#include "model/equity/markovian/heston_3_2/product/american_option_price_gradients.cuh"
#include "model/equity/markovian/heston_3_2/product/american_option_continuation.cuh"
#include "model/equity/markovian/heston_3_2/dynamics_impl.cuh"
#include "model/equity/markovian/heston_3_2/price_gradients/coupled_dynamics_impl.cuh"

#include "tests/price_gradients/cuda_test_support.cuh"

using namespace ai_factory::workbench;
namespace pg = price_gradients;
namespace ls = longstaff_schwartz::price_gradients;
using price_gradient_test::DeviceArray;
using price_gradient_test::require;
using price_gradient_test::same;
constexpr std::size_t paths = 8192;
constexpr std::uint64_t seed = 73319;
constexpr pg::TimeConfiguration time_grid{1.0f/504.0f, 2U};

struct sabr_contract {
    using Model = model::equity::sabr::ModelParameters;
    using Dynamics = model::equity::sabr::DynamicsPolicy;
    using Coupled = model::equity::sabr::price_gradients::CoupledDynamics;
    using Continuation = model::equity::sabr::AmericanOptionContinuationState;
    using Plan = model::equity::sabr::AmericanOptionPriceGradientPlan;
    static auto prepare(const std::vector<Model>& m, const std::vector<product::AmericanOptionParameters>& p,
                        PriceConstruction layout, const pg::PriceGradientConfiguration& c) {
        return model::equity::sabr::prepare_sabr_american_option_price_gradients(m,p,layout,time_grid,c);
    }
    template<OptionSide Side, class I, class S>
    static auto gradients(const Plan& p, I i, S s, const pg::LaunchConfiguration& l,
                          pg::Outputs o, ls::ExerciseReplayStrategy replay) {
        return model::equity::sabr::launch_sabr_american_option_price_gradients_with_replay_cuda<Side>(p,i,s,l,o,replay);
    }
    template<OptionSide Side, class I, class S>
    static auto graph(const Plan& p, I i, S s, const pg::LaunchConfiguration& l,
                      pg::Outputs o, ls::ExerciseReplayStrategy replay) {
        auto gp=pg::prepare_device_sensitivities_absolute<Plan>(
            std::span<const Model>(p.models), std::span<const product::AmericanOptionParameters>(p.products),
            p.construction,p.time,p.configuration,{pg::SensitivityOrders::first_and_second},p.relative_quotes);
        DeviceArray<pg::SensitivityStencil<4U>> stencils(p.result_count*p.sensitivity_count());
        typename Plan::DiagonalStencilOutputs ds{stencils.data,stencils.count,s.error};
        DeviceArray<float> hessian(stencils.count),hessian_se(stencils.count);
        const auto bytes=model::equity::sabr::sabr_american_option_node_graph_workspace_bytes_with_replay<Side,pg::SensitivityOrders::first_and_second>(gp,l,replay);
        DeviceArray<std::uint8_t> workspace(bytes);
        const pg::SensitivityOutputs full{o.prices,o.price_standard_errors,o.gradients,o.gradient_standard_errors,hessian.data,hessian_se.data,o.price_capacity,o.gradient_capacity};
        return model::equity::sabr::launch_sabr_american_option_node_graph_sensitivities_with_replay_cuda<Side,pg::SensitivityOrders::first_and_second>(gp,i,ds,l,full,workspace.data,bytes,replay);
    }
    template<OptionSide Side>
    static auto price(const Model* m, std::size_t nm, const std::vector<product::AmericanOptionParameters>& hp,
                      const product::AmericanOptionParameters* dp, PriceConstruction layout, std::size_t rows,
                      float* prices, float* errors) {
        return model::equity::sabr::launch_sabr_american_option_cuda<Side>(m,nm,hp.data(),dp,hp.size(),layout,rows,paths,time_grid.dt,2U,128U,4U,seed,prices,errors);
    }
};

struct heston_3_2_contract {
    using Model = model::equity::heston_3_2::ModelParameters;
    using Dynamics = model::equity::heston_3_2::DynamicsPolicy;
    using Coupled = model::equity::heston_3_2::price_gradients::CoupledDynamics;
    using Continuation = model::equity::heston_3_2::AmericanOptionContinuationState;
    using Plan = model::equity::heston_3_2::AmericanOptionPriceGradientPlan;
    static auto prepare(const std::vector<Model>& m, const std::vector<product::AmericanOptionParameters>& p,
                        PriceConstruction layout, const pg::PriceGradientConfiguration& c) {
        return model::equity::heston_3_2::prepare_heston_3_2_american_option_price_gradients(m,p,layout,time_grid,c);
    }
    template<OptionSide Side, class I, class S>
    static auto gradients(const Plan& p, I i, S s, const pg::LaunchConfiguration& l,
                          pg::Outputs o, ls::ExerciseReplayStrategy replay) {
        return model::equity::heston_3_2::launch_heston_3_2_american_option_price_gradients_with_replay_cuda<Side>(p,i,s,l,o,replay);
    }
    template<OptionSide Side, class I, class S>
    static auto graph(const Plan& p, I i, S s, const pg::LaunchConfiguration& l,
                      pg::Outputs o, ls::ExerciseReplayStrategy replay) {
        auto gp=pg::prepare_device_sensitivities_absolute<Plan>(
            std::span<const Model>(p.models), std::span<const product::AmericanOptionParameters>(p.products),
            p.construction,p.time,p.configuration,{pg::SensitivityOrders::first_and_second},p.relative_quotes);
        DeviceArray<pg::SensitivityStencil<4U>> stencils(p.result_count*p.sensitivity_count());
        typename Plan::DiagonalStencilOutputs ds{stencils.data,stencils.count,s.error};
        DeviceArray<float> hessian(stencils.count),hessian_se(stencils.count);
        const auto bytes=model::equity::heston_3_2::heston_3_2_american_option_node_graph_workspace_bytes_with_replay<Side,pg::SensitivityOrders::first_and_second>(gp,l,replay);
        DeviceArray<std::uint8_t> workspace(bytes);
        const pg::SensitivityOutputs full{o.prices,o.price_standard_errors,o.gradients,o.gradient_standard_errors,hessian.data,hessian_se.data,o.price_capacity,o.gradient_capacity};
        return model::equity::heston_3_2::launch_heston_3_2_american_option_node_graph_sensitivities_with_replay_cuda<Side,pg::SensitivityOrders::first_and_second>(gp,i,ds,l,full,workspace.data,bytes,replay);
    }
    template<OptionSide Side>
    static auto price(const Model* m, std::size_t nm, const std::vector<product::AmericanOptionParameters>& hp,
                      const product::AmericanOptionParameters* dp, PriceConstruction layout, std::size_t rows,
                      float* prices, float* errors) {
        return model::equity::heston_3_2::launch_heston_3_2_american_option_cuda<Side>(m,nm,hp.data(),dp,hp.size(),layout,rows,paths,time_grid.dt,2U,128U,4U,seed,prices,errors);
    }
};

struct Results { std::vector<float> price, se, gradient, gradient_se; };
template<class C, OptionSide Side>
Results run(const typename C::Plan& p, ls::ExerciseReplayStrategy replay,
            std::size_t offset=0, std::size_t count=0, bool graph=false) {
    if (!count) count=p.result_count;
    const auto k=p.sensitivity_count();
    DeviceArray<typename C::Model> models(p.models);
    DeviceArray<product::AmericanOptionParameters> products(p.products);
    DeviceArray<typename C::Plan::SensitivitySpec> specs(p.sensitivities);
    DeviceArray<pg::SensitivityStencil<3U>> stencils(p.result_count*k);
    DeviceArray<equity::price_gradients::device_preparation::Error> error(1);
    DeviceArray<float> prices(p.result_count),se(p.result_count),grad(p.result_count*k),gse(p.result_count*k);
    const typename C::Plan::DeviceInputs inputs{models.data,models.count,products.data,products.count,specs.data,specs.count};
    const typename C::Plan::StencilOutputs outputs{stencils.data,stencils.count,error.data};
    const pg::LaunchConfiguration launch{pg::PricingMethod::monte_carlo,offset,count,paths,128U,4U,seed,1U};
    const pg::Outputs out{prices.data,se.data,grad.data,gse.data,p.result_count,p.result_count*k};
    const auto diag=graph ? C::template graph<Side>(p,inputs,outputs,launch,out,replay)
                          : C::template gradients<Side>(p,inputs,outputs,launch,out,replay);
    longstaff_schwartz::validate_regression_diagnostics(diag,"SABR/Heston3/2 American");
    require(error.read()[0].code==0,"stencil preparation failed");
    const auto slice=[&](const auto& values, std::size_t stride) {
        return std::vector<float>(values.begin()+offset*stride,values.begin()+(offset+count)*stride);
    };
    Results result{slice(prices.read(),1),slice(se.read(),1),slice(grad.read(),k),slice(gse.read(),k)};
    for (auto* v : {&result.price,&result.se,&result.gradient,&result.gradient_se})
        for (float x:*v) require(std::isfinite(x),"nonfinite output");
    for (float x:result.se) require(x>=0,"negative price SE");
    for (float x:result.gradient_se) require(x>=0,"negative gradient SE");
    return result;
}

// Compare direct dynamics and repeated identical CRN nodes, including a non-unit spot.
template<class C>
__global__ void replay_kernel(typename C::Model model, float* out) {
    using D=typename C::Dynamics;
    using R=ls::DevicePreparedFrozenExerciseReplay<typename C::Coupled,3U>;
    using P=typename C::Plan::Preparation;
    pg::SensitivityNodes<typename P::Scenario,3U> nodes{};
    P::make_central(model,{.9f,63U,7U,true},time_grid,nodes[0]);
    nodes[1]=nodes[0];nodes[2]=nodes[0];nodes[1].reuse_central=false;nodes[2].reuse_central=false;
    const auto prepared=R::prepare(nodes,3U,{time_grid.dt,14*time_grid.dt,14*time_grid.dt});
    const auto d=D::prepare_dynamics(model,time_grid.dt);auto state=D::initial_state(d);
    const auto key=philox::make_key(seed);typename D::RandomContext random(key,0U);
    D::advance(d,14U,random,state);for(int i=0;i<3;++i)D::advance(d,14U,random,state);
    const float spot=D::spot(state);auto values=R::evaluate(prepared,{3U,spot},key,0U,14U,14U);
    out[0]=spot;out[1]=values[0];out[2]=values[1];out[3]=values[2];
    auto cp=C::Continuation::prepare(model);
    out[4]=C::Continuation::regression_input(cp,D::initial_state(d),1.0f/model.spot).secondary;
}

template<class C, OptionSide Side>
void check(const std::vector<typename C::Model>& models) {
    const std::vector<product::AmericanOptionParameters> products{{.9f,63U,7U,true},{1.1f,42U,14U,true}};
    pg::PriceGradientConfiguration cfg;
    for (auto n:C::Plan::Preparation::ModelAdapter::parameter_names) {
        cfg.sensitivities.push_back({std::string(n),{.001f,pg::BumpScale::absolute}});
    }
    cfg.sensitivities.push_back({"product.strike",{.005f,pg::BumpScale::relative}});
    const auto plan=C::prepare(models,products,PriceConstruction::CartesianProduct,cfg);
    require(plan.result_count==4,"Cartesian count");
    DeviceArray<typename C::Model> dm(models);DeviceArray<product::AmericanOptionParameters> dp(products);
    DeviceArray<float> price(4),se(4);
    C::template price<Side>(dm.data,models.size(),products,dp.data,PriceConstruction::CartesianProduct,4,price.data,se.data);
    const auto central=price.read(),central_se=se.read();
    for(auto replay:{ls::ExerciseReplayStrategy::frozen_exercise_time,ls::ExerciseReplayStrategy::frozen_regression_policy}) {
        const auto full=run<C,Side>(plan,replay);
        require(full.price==central && full.se==central_se,"price/gradient central parity");
        const auto graph=run<C,Side>(plan,replay,0,0,true);
        require(graph.price==full.price && graph.se==full.se && graph.gradient==full.gradient && graph.gradient_se==full.gradient_se,"mono/graph parity");
        for(std::size_t row=0;row<4;++row) {
            const auto chunk=run<C,Side>(plan,replay,row,1);
            same(chunk.price[0],full.price[row],"offset row price");
            same(chunk.se[0],full.se[row],"offset row SE");
            for(std::size_t j=0;j<plan.sensitivity_count();++j){
                same(chunk.gradient[j],full.gradient[row*plan.sensitivity_count()+j],"offset gradient");
                same(chunk.gradient_se[j],full.gradient_se[row*plan.sensitivity_count()+j],"offset gradient SE");
            }
        }
        // Quote contract must equal the corresponding absolute contract at S0 != 1.
        auto absolute=products;for(auto& x:absolute){x.strike*=models[0].spot;x.quote_is_ratio=false;}
        std::vector<typename C::Model> one{models[0]};
        pg::PriceGradientConfiguration delta{{{"model.spot",{.001f,pg::BumpScale::absolute}}}};
        const auto quoted=C::prepare(one,products,PriceConstruction::CartesianProduct,delta);
        const auto resolved=pg::prepare_device_sensitivities_absolute<typename C::Plan>(
            std::span<const typename C::Model>(one),std::span<const product::AmericanOptionParameters>(absolute),
            PriceConstruction::CartesianProduct,time_grid,delta,{pg::SensitivityOrders::first},false);
        const auto q=run<C,Side>(quoted,replay),a=run<C,Side>(resolved,replay);
        require(q.price==a.price && q.gradient==a.gradient && q.gradient_se==a.gradient_se,"fixed absolute strike under spot bump");
        for(std::size_t row=0;row<2;++row)
            same(q.gradient[row],full.gradient[row*plan.sensitivity_count()],"gradient selection preserves CRN");
    }
    std::cout<<"PASS "<<typeid(C).name()<<" side="<<int(Side)<<" prices, both replays, graph, offsets, quote, CRN\n";
}
// Independent CPU CRR reference, allowing exercise only on the contract's dates.
double tree(bool call, double spot, double strike, double r, double q, double sigma) {
    constexpr int n=504, daily=8, interval=7*daily;
    const double dt=.25/n, u=std::exp(sigma*std::sqrt(dt)), d=1/u;
    const double probability=(std::exp((r-q)*dt)-d)/(u-d), discount=std::exp(-r*dt);
    auto payoff=[&](double x){return std::max(0.,call?x-strike:strike-x);};
    std::vector<double> v(n+1);
    for(int j=0;j<=n;++j)v[j]=payoff(spot*std::pow(u,2*j-n));
    for(int i=n-1;i>=0;--i)for(int j=0;j<=i;++j){
        v[j]=discount*((1-probability)*v[j]+probability*v[j+1]);
        if(i && i%interval==0)v[j]=std::max(v[j],payoff(spot*std::pow(u,2*j-i)));
    }
    return v[0];
}
template<class C>
void canonical(const typename C::Model& model, float expected_secondary) {
    DeviceArray<float> v(5);replay_kernel<C><<<1,1>>>(model,v.data);
    check_cuda(cudaGetLastError(),"canonical replay kernel");const auto x=v.read();
    for(int i=1;i<4;++i)same(x[i],x[0],"direct dynamics versus CRN nodes");
    require(std::abs(x[4]-expected_secondary)<2e-6f,"continuation normalization");
}
template<class C, OptionSide Side>
void constant_vol_reference(const typename C::Model& model) {
    const std::vector<typename C::Model> models{model};
    const std::vector<product::AmericanOptionParameters> products{{1.1f,63U,7U,false}};
    DeviceArray<typename C::Model> dm(models);DeviceArray<product::AmericanOptionParameters> dp(products);
    DeviceArray<float> price(1),se(1);
    C::template price<Side>(dm.data,1,products,dp.data,PriceConstruction::Aligned,1,price.data,se.data);
    const auto p=price.read()[0], e=se.read()[0];
    const double ref=tree(Side==OptionSide::call,model.spot,1.1,model.risk_free_rate,model.dividend_yield,.2);
    std::cout<<"constant-vol CPU tree: "<<p<<" +/- "<<e<<" ref="<<ref<<'\n';
    require(std::abs(p-ref)<6*e+.003,"independent constant-vol American price");
}
int main() {
 try {
    const std::vector<sabr_contract::Model> sabr{{1.7f,.03f,.01f,.22f,.4f,-.6f,.7f},{.8f,.04f,.06f,.3f,.6f,-.4f,.5f}};
    const std::vector<heston_3_2_contract::Model> h32{{1.7f,.03f,.01f,.04f,2.f,.04f,.5f,-.6f},{.8f,.04f,.06f,.09f,1.5f,.06f,.8f,-.4f}};
    canonical<sabr_contract>(sabr[0],1.f);canonical<heston_3_2_contract>(h32[0],1.f);
    const sabr_contract::Model slimit{1.f,.03f,.06f,.2f,1e-7f,-.6f,1.f};
    const heston_3_2_contract::Model hlimit{1.f,.03f,.06f,.04f,2.f,.04f,1e-7f,-.6f};
    constant_vol_reference<sabr_contract,OptionSide::call>(slimit);
    constant_vol_reference<sabr_contract,OptionSide::put>(slimit);
    constant_vol_reference<heston_3_2_contract,OptionSide::call>(hlimit);
    constant_vol_reference<heston_3_2_contract,OptionSide::put>(hlimit);
    check<sabr_contract,OptionSide::call>(sabr);check<sabr_contract,OptionSide::put>(sabr);
    check<heston_3_2_contract,OptionSide::call>(h32);check<heston_3_2_contract,OptionSide::put>(h32);
    return 0;
 } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
