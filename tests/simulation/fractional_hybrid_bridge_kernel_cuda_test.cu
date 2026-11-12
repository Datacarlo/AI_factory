// Independent first-cell covariance oracle for correlated fractional integrals.
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "tools/pricing/heterogeneous_rough_price_generation.cuh"
#include <cmath>
#include <cstdio>
#include <stdexcept>

namespace w=ai_factory::workbench;
using Policy=w::model::equity::rough_multi_asset::HybridPolicy<w::model::equity::rough_bergomi::PathPolicy>;
using Pack=w::simulation::MultiRoughPack<Policy,Policy>;
constexpr unsigned paths=1U<<18U,steps=2U;
constexpr float dt=1.f/504.f;
void run(float h0,float h1,double rho) {
    w::offline::pricing::MultiRoughHostModels<Pack> h;
    h.models={{1.f,0.f,0.f,.04f,1.f,h0,0.f}};
    h.tail.models={{1.f,0.f,0.f,.04f,1.f,h1,0.f}};
    w::offline::pricing::MultiRoughDeviceModels<Pack> models(h);
    const w::RowIndexSelector selectors[]={{1,1,w::RowIndexMode::Strided},
        {1,1,w::RowIndexMode::Strided},{1,1,w::RowIndexMode::Strided},{1,1,w::RowIndexMode::Strided}};
    w::offline::cuda::DeviceBuffer<w::RowIndexSelector> ds(4);ds.copy_from(selectors);
    const auto lower=w::correlation::packed_cholesky({1,rho,0,0,rho,1,0,0,0,0,1,0,0,0,0,1},4);
    w::offline::cuda::DeviceBuffer<float> dl(lower.size());dl.copy_from(lower.data());
    const auto bytes=models.workspace_bytes(steps,paths);
    w::offline::cuda::DeviceBuffer<unsigned char> workspace(bytes);
    w::offline::cuda::DeviceBuffer<float> out(2*paths*steps);
#ifdef AI_FACTORY_TEST_LEGACY_BRIDGE
    // Reproduce the old source routing, deliberately omitting the joint bridge.
    for(unsigned i=0;i<2;++i)
        w::volterra::hybrid_fft::launch_correlated_hybrid_volatility_cuda<Policy,4>(
            i?models.tail.models.data():models.models.data(),dl.data(),0,0,paths,dt,steps,193475,
            workspace.data(),bytes,out.data()+i*paths*steps,i,2+i,nullptr,ds.data(),i,3);
#else
    models.launch_volatilities<4,0>(dl.data(),0,0,paths,dt,steps,32,193475,
        workspace.data(),bytes,out.data(),nullptr,ds.data(),3);
#endif
    std::vector<float> v(2*paths*steps);out.copy_to(v.data());
    double sx=0,sy=0,sxx=0,syy=0,sxy=0;
    for(unsigned p=0;p<paths;++p) {
        const auto k=w::volterra::causal_fft::history_index(p,1,steps);
        const double x=(std::log(double(v[k])/.04)+.5*std::pow(double(dt),2*h0))/std::pow(double(dt),h0);
        const double y=(std::log(double(v[paths*steps+k])/.04)+.5*std::pow(double(dt),2*h1))/std::pow(double(dt),h1);
        sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;
    }
    const double expected=rho*2*std::sqrt(double(h0)*h1)/(h0+h1);
    const double empirical=sxy/paths-sx*sy/(double(paths)*paths);
    const double se=std::sqrt((1+expected*expected)/paths);
    std::printf("H=(%.4g,%.4g) rho=%.4g covariance=%.9g exact=%.9g error/SE=%.5g var=(%.7g,%.7g)\n",h0,h1,rho,empirical,expected,(empirical-expected)/se,sxx/paths,syy/paths);
#ifndef AI_FACTORY_TEST_LEGACY_BRIDGE
    if(std::abs(empirical-expected)>6*se+5e-5 || std::abs(sxx/paths-1)>.02 || std::abs(syy/paths-1)>.02)
        throw std::runtime_error("First-cell cross covariance differs from integral of the two kernels");
#endif
}
int main() {
    for(double rho:{-.9,0.,.9}){run(.1f,.1f,rho);run(.08f,.4f,rho);}
    run(.4f,.08f,.8);run(.5f,.5f,.9);
}
