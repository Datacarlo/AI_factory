// Independent cashflow oracles plus integration with correlated rates and rough paths.
#include "product/multi_asset_autocall/pricing_policy.cuh"
#include "product/athena_autocall/pricing_policy.cuh"
#include "product/phoenix_autocall/pricing_policy.cuh"
#include "product/phoenix_memory_autocall/pricing_policy.cuh"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "model/fixed_income/constant_rate/correlated_dynamics.cuh"
#include "model/fixed_income/vasicek/correlated_dynamics.cuh"
#include "multi_asset/rate_adjusted_model_group.cuh"
#include "tools/pricing/cross_asset_price_generation.cuh"
#include <array>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace w=ai_factory::workbench;
namespace p=w::product;
using Params=p::MultiAssetAutocallParameters;
using I=p::AutocallIndicator;
using Athena=p::MultiAssetAthenaAutocallPathPolicy;
using Phoenix=p::MultiAssetPhoenixAutocallPathPolicy;
using Memory=p::MultiAssetPhoenixMemoryAutocallPathPolicy;
using BS=w::model::equity::black_scholes::DynamicsPolicy;
using TwoBS=w::simulation::MultiAssetPack<w::simulation::RateAdjustedModelGroup<BS,2U>>;
using OneBS=w::simulation::MultiAssetPack<w::simulation::RateAdjustedModelGroup<BS,1U>>;
using NoRough=w::simulation::MultiRoughPack<>;
using Rate=w::model::fixed_income::vasicek::CorrelatedRatePolicy;
using Fixed=w::model::fixed_income::constant_rate::CorrelatedRatePolicy;
using E=w::RowIndexExpression;
void require(bool ok,const char* message) { if(!ok)throw std::runtime_error(message); }
void close(double a,double b,double tol,const char* what) {
    if(std::abs(a-b)>tol) { std::cerr<<what<<": "<<a<<" vs "<<b<<'\n';throw std::runtime_error(what); }
}
w::CompiledRowIndexPlan one_plan(unsigned slots) {
    std::vector<E> v;for(unsigned i=0;i<slots;++i)v.push_back(E::source(i,1));
    return w::compile_row_index_plan(E::aligned(v),slots);
}
// Deliberately double precision, with an explicit cashflow list and no product handler.
double reference(int kind,const Params& p,const std::vector<std::array<double,3>>& obs,
                 const std::vector<double>& discounts) {
    const double c=double(p.annual_coupon_rate)*p.observation_interval_days/252.;
    double result=0;unsigned missed=0;
    for(unsigned i=0;i<obs.size();++i) {
        const auto at=[&](I k){return obs[i][unsigned(k)];};
        const bool last=i+1==obs.size(),call=at(p.autocall_indicator)>=p.autocall_barrier;
        if(kind==0) {
            if(call) return discounts[i]*(1+(i+1)*c);
            if(last)return discounts[i]*(at(p.protection_indicator)>=p.protection_barrier?1:at(p.loss_indicator));
        } else {
            ++missed;double coupon=0;
            if(at(p.coupon_indicator)>=p.coupon_barrier){coupon=c*(kind==2?missed:1);missed=0;}
            result+=discounts[i]*coupon;
            if(last || call) {
                result+=discounts[i]*(!last || at(p.protection_indicator)>=p.protection_barrier?1:at(p.loss_indicator));
                return result;
            }
        }
    }
    return result;
}
template<class Product> void scripted(int kind) {
    const std::vector<std::array<double,3>> v{{.7,1.2,.95},{.8,1.3,1.05},{.6,1.1,.85}};
    const std::vector<double> ds{.99,.97,.94};
    for(unsigned a=0;a<3;++a)for(unsigned b=0;b<3;++b)for(unsigned c=0;c<3;++c)for(unsigned d=0;d<3;++d){
        Params q{189,63,1.15f,.75f,.08f,I(a),I(b),I(c),1.f,I(d)};
        auto prepared=Product::prepare(q,1);auto h=Product::make_handler(prepared);
        for(unsigned n=0;n<v.size();++n){p::AutocallObservation o;for(unsigned k=0;k<3;++k)o.values[k]=float(v[n][k]);
            if(!h.on_observation(n,o,float(ds[n])))break;}
        close(Product::finalize(prepared,h),reference(kind,q,v,ds),3e-7,"cashflow oracle");
    }
    // Exact threshold comparisons, coupon memory recovered, lost at maturity, and no call at t=0.
    Params q{189,63,2.f,.5f,.08f,I::worst_of,I::worst_of,I::worst_of,1.f,I::worst_of};
    require(!Product::is_observation_step(0,378,2,q),"observed at inception");
    for(unsigned spd:{1U,2U,4U}){unsigned count=0;for(unsigned k=0;k<189*spd;++k)if(Product::is_observation_step(k,189*spd,spd,q)){
        require(Product::observation_index(k,spd,q)==count++,"fixing index");}require(count==3,"fixing count");}
    for(const auto& path:std::vector<std::vector<std::array<double,3>>>{{{.9,.9,.9},{1,1,1},{.4,.4,.4}},{{.9,.9,.9},{.9,.9,.9},{.4,.4,.4}}}){
        auto prepared=Product::prepare(q,1);auto h=Product::make_handler(prepared);
        for(unsigned n=0;n<3;++n){p::AutocallObservation o;for(unsigned k=0;k<3;++k)o.values[k]=float(path[n][k]);if(!h.on_observation(n,o,float(ds[n])))break;}
        close(Product::finalize(prepared,h),reference(kind,q,path,ds),3e-7,"coupon memory/terminal loss");
    }
}
// A normalized one-name contract must agree with the existing single-asset handler.
__global__ void single_asset_parity(float* errors) {
    w::model::equity::black_scholes::ModelParameters model{2.5f,.07f,0,.2f};
    const w::equity::ProductPreparationContext ctx{1.f/252.f,1.f};
    auto sa=p::AthenaAutocallPathPolicy::make_handler(p::AthenaAutocallPathPolicy::prepare_product(model,{252,63,3.f,1.75f,.08f,2.5f},ctx));
    auto sp=p::PhoenixAutocallPathPolicy::make_handler(p::PhoenixAutocallPathPolicy::prepare_product(model,{252,63,3.f,2.5f,1.75f,.08f,2.5f},ctx));
    auto sm=p::PhoenixMemoryAutocallPathPolicy::make_handler(p::PhoenixMemoryAutocallPathPolicy::prepare_product(model,{252,63,3.f,2.5f,1.75f,.08f,2.5f},ctx));
    Params q{252,63,1.2f,.7f,.08f,I::worst_of,I::worst_of,I::worst_of,1.f,I::worst_of};
    auto a=Athena::make_handler(Athena::prepare(q,1));auto b=Phoenix::make_handler(Phoenix::prepare(q,1));auto c=Memory::make_handler(Memory::prepare(q,1));
    float discount=1,per=expf(-.07f*.25f);const float ratios[]{.9f,1.05f,.95f,.6f};
    for(unsigned k=0;k<4;++k){discount*=per;p::AutocallObservation v{{ratios[k],ratios[k],ratios[k]}};
        sa.on_observation(k,2.5f*ratios[k]);sp.on_observation(k,2.5f*ratios[k]);sm.on_observation(k,2.5f*ratios[k]);
        a.on_observation(k,v,discount);b.on_observation(k,v,discount);c.on_observation(k,v,discount);}
    errors[0]=a.present_value-sa.discounted_payoff;errors[1]=b.present_value-sp.present_value;errors[2]=c.present_value-sm.present_value;
}
template<class Product,unsigned Team> void deterministic(int kind) {
    w::offline::pricing::MultiAssetHostModels<TwoBS> equity;
    equity.models={{2.f,0.f,.10f,0.f},{7.f,0.f,-.10f,0.f}};
    w::offline::pricing::MultiRoughHostModels<NoRough> rough;
    std::vector<Params> products;
    for(unsigned i=0;i<3;++i){products.push_back({189,63,1.03f,.99f,.08f,I(i),I(i),I(i),.99f,I(i)});products.push_back({189,63,2.f,.99f,.08f,I(i),I(i),I(i),1.f,I(i)});}
    auto plan=w::compile_row_index_plan(E::cartesian({E::source(3,products.size()),E::source(2,2),E::source(0,2),E::source(1,2),E::source(4,1)}),5);
    const std::vector<std::string> f{"a","b"};const std::vector<w::correlation::MatrixRow> corr{{"id",f,{1,0,1}}};
    const std::vector<Fixed::Parameters> rates{{.02f},{.08f}};
    for(unsigned spd:{1U,2U,4U}){
        auto run=w::offline::pricing::execute_cross_asset_prices<TwoBS,NoRough,Fixed,Product,Team>(equity,rough,rates,products,corr,f,f,plan,256,256,1.f/(252*spd),spd,32,1234,{}, {5,2,128,1,1,2});
        for(unsigned row=0;row<plan.result_count;++row){const auto mi=plan.selectors[0].select(row),pi=plan.selectors[3].select(row),ri=plan.selectors[2].select(row);
            const auto& q=products[pi];const double rate=rates[ri].rate,div=equity.models[mi].dividend_yield,div2=equity.models[plan.selectors[1].select(row)].dividend_yield;
            std::vector<std::array<double,3>> obs;std::vector<double> ds;
            for(unsigned k=1;k<=3;++k){const double t=k*.25;const double ratio=std::exp((rate-div)*t),other=std::exp((rate-div2)*t);obs.push_back({std::min(ratio,other),std::max(ratio,other),(ratio+other)/2});ds.push_back(std::exp(-rate*t));}
            close(run.prices[row],reference(kind,q,obs,ds),4e-5,"deterministic rates, row mapping or fixing");require(run.standard_errors[row]<1e-6,"deterministic standard error");}
    }
    // Different initial levels alone cannot change a normalized contract.
    equity.models.resize(1);
    auto go=[&](){return w::offline::pricing::execute_cross_asset_prices<TwoBS,NoRough,Fixed,Product,Team>(equity,rough,{{.02f}},{products[0]},corr,f,f,one_plan(5),256,256,1.f/504,2,32,1234);};
    auto x=go();equity.models[0].spot=13.f;auto y=go();close(x.prices[0],y.prices[0],4e-5,"spot normalization");
}
void stochastic_discount(){
    w::offline::pricing::MultiAssetHostModels<OneBS> eq;eq.models={{2.f,0.f,0.f,0.f}};
    w::offline::pricing::MultiRoughHostModels<NoRough> rough;
    const std::vector<std::string> f{"spot","rate"};
    const std::vector<w::correlation::MatrixRow> corr{{"c",f,w::correlation::packed_cholesky({1.,.3,.3,1.},2)}};
    // With sigma_spot=0 and q=0, discounted terminal loss S(T)/S(0) is exactly one.
    Params q{189,63,100.f,100.f,.12f,I::worst_of,I::worst_of,I::worst_of,.001f,I::worst_of};
    auto run=w::offline::pricing::execute_cross_asset_prices<OneBS,NoRough,Rate,Phoenix,4>(eq,rough,{{{.7f,.025f,.035f},.02f}},{q},corr,f,f,one_plan(4),8192,8192,1.f/504,2,32,5511);
    double expected=1;for(unsigned k=1;k<=3;++k){double t=k*.25,a=.7,theta=.025,r0=.02,sigma=.035;double B=(1-std::exp(-a*t))/a;double variance=sigma*sigma/(a*a)*(t-2*B+(1-std::exp(-2*a*t))/(2*a));expected+=.03*std::exp(-theta*t-(r0-theta)*B+.5*variance);}
    close(run.prices[0],expected,6*run.standard_errors[0]+3e-5,"coupons must use their own stochastic discounts");
}
template<class Product> void rough_geometry(){
    using QR=w::model::equity::quadratic_rough_heston::CausalFftPathPolicy;
    using RB=w::model::equity::rough_multi_asset::HybridPolicy<w::model::equity::rough_bergomi::PathPolicy>;
    using RP=w::simulation::MultiRoughPack<QR,RB>;
    w::offline::pricing::MultiAssetHostModels<OneBS> eq;eq.models={{2.f,0.f,0.f,.2f}};
    w::offline::pricing::MultiRoughHostModels<RP> rough;rough.models={{1.f,0.f,0.f,.08f,.55f,1.f,.04f,1.5f,.55f,.12f}};rough.tail.models={{3.f,0.f,.01f,.04f,1.f,.12f,-.7f}};
    const std::vector<std::string> external{"bs","qr","rb","rbvol","rate"},internal{"qr","rbvol","bs","rate","rb"};
    std::vector<double> matrix(25,0);for(unsigned i=0;i<5;++i)matrix[5*i+i]=1;matrix[2*5+3]=matrix[3*5+2]=-.7;
    const std::vector<w::correlation::MatrixRow> corr{{"c",external,w::correlation::packed_cholesky(matrix,5)}};
    std::vector<Params> products{{32,8,1.02f,.8f,.08f,I::best_of,I::worst_of,I::basket,.98f,I::basket},{32,8,10.f,1.1f,.12f,I::worst_of,I::worst_of,I::best_of,1.f,I::best_of},{32,8,.1f,.8f,.12f,I::basket,I::worst_of,I::worst_of,.1f,I::worst_of}};
    auto plan=w::compile_row_index_plan(E::cartesian({E::source(4,products.size()),E::aligned({E::source(0,1),E::source(1,1),E::source(2,1),E::source(3,1),E::source(5,1)})}),6);
    auto a=w::offline::pricing::execute_cross_asset_prices<OneBS,RP,Rate,Product,1>(eq,rough,{{{.7f,.025f,.035f},.02f}},products,corr,external,internal,plan,512,512,1.f/504,2,32,7799,{}, {3,0,128,1,1,1});
    auto b=w::offline::pricing::execute_cross_asset_prices<OneBS,RP,Rate,Product,4>(eq,rough,{{{.7f,.025f,.035f},.02f}},products,corr,external,internal,plan,512,256,1.f/504,2,32,7799,{}, {3,0,256,2,2,8});
    for(unsigned i=0;i<products.size();++i){require(std::isfinite(a.prices[i]) && a.prices[i]>0,"rough autocall invalid");close(a.prices[i],b.prices[i],3e-6,"rough L/P/S/team invariance");close(a.standard_errors[i],b.standard_errors[i],3e-6,"rough standard error invariance");}
}
int main(){
    scripted<Athena>(0);scripted<Phoenix>(1);scripted<Memory>(2);
    std::cout<<"Scripted contracts passed"<<std::endl;
    bool rejected=false;try{Athena::validate_time_configuration(1.f/252,2);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"inconsistent time grid accepted");
    Params bad{32,7,1.f,.8f,.1f};require(!Athena::valid_parameters(bad,2),"incomplete calendar accepted");
    float* errors;w::check_cuda(cudaMallocManaged(&errors,3*sizeof(float)),"parity allocation");single_asset_parity<<<1,1>>>(errors);w::check_cuda(cudaDeviceSynchronize(),"parity");for(unsigned i=0;i<3;++i)close(errors[i],0,2e-6,"single/multi contract parity");w::check_cuda(cudaFree(errors),"parity free");
    deterministic<Athena,1>(0);deterministic<Athena,4>(0);deterministic<Phoenix,1>(1);deterministic<Memory,4>(2);
    std::cout<<"Deterministic engine/mapping passed"<<std::endl;
    stochastic_discount();
    std::cout<<"Stochastic discount passed"<<std::endl;
    rough_geometry<Athena>();rough_geometry<Phoenix>();rough_geometry<Memory>();
    std::cout<<"Autocalls: cashflows, selectors, time, mapping, rates and mixed FFT geometry passed\n";
}
