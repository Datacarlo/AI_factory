// Conditional singular-cell innovations for several fractional Gaussian factors.
#pragma once
#include "multi_asset/rough_pack.cuh"
#include "volterra/fractional_hybrid_kernel.cuh"
#include <type_traits>

namespace ai_factory::workbench::simulation {

template<typename Pack> struct HybridBridgeTraits;
template<> struct HybridBridgeTraits<MultiRoughPack<>> {
    static constexpr unsigned count=0;
    static constexpr bool supported=true;
    __device__ static void collect(const MultiRoughPack<>::Inputs&, unsigned,
        double*, unsigned*, unsigned&, unsigned=0) {}
};
template<typename Head,typename... Tail>
struct HybridBridgeTraits<MultiRoughPack<Head,Tail...>> {
    using Pack=MultiRoughPack<Head,Tail...>;
    using Rest=HybridBridgeTraits<MultiRoughPack<Tail...>>;
    static constexpr bool gaussian=!(requires { typename Head::VolatilityState; });
    static constexpr unsigned count=unsigned(gaussian)+Rest::count;
    static constexpr bool supported=(!gaussian || std::is_same_v<
        typename Head::KernelPolicy,volterra::FractionalHybridKernelPolicy>) && Rest::supported;
    __device__ static void collect(const typename Pack::Inputs& input,unsigned row,
        double* hursts,unsigned* factors,unsigned& next,unsigned offset=0) {
        if constexpr(gaussian) {
            hursts[next]=input.models[input.row_selectors[0].select(row)].hurst_exponent;
            factors[next++]=offset;
        }
        Rest::collect(input.tail,row,hursts,factors,next,offset+1);
    }
};
template<typename Policy,std::size_t Count>
struct HybridBridgeTraits<HomogeneousRoughPack<Policy,Count>> {
    using Pack=HomogeneousRoughPack<Policy,Count>;
    static constexpr unsigned count=Pack::kAllCausal?0:Count;
    static constexpr bool supported=Pack::kAllCausal || std::is_same_v<
        typename Policy::KernelPolicy,volterra::FractionalHybridKernelPolicy>;
    __device__ static void collect(const typename Pack::Inputs& input,unsigned row,
        double* hursts,unsigned* factors,unsigned& next,unsigned offset=0) {
        if constexpr(!Pack::kAllCausal) for(unsigned i=0;i<Count;++i) {
            hursts[next]=input.models[input.row_selectors[i].select(row)].hurst_exponent;
            factors[next++]=offset+i;
        }
    }
};

// Given Brownian cell averages, Cov(I_i,I_j) is rho_ij times
// dt^(H_i+H_j) [1/(H_i+H_j)-1/((H_i+.5)(H_j+.5))].
// After normalization the residual correlation is
// rho_ij * 2 sqrt(H_i H_j)/(H_i+H_j) (for H_i,H_j<.5).
// The dt and kernel normalization cancel. Prepare one small Cholesky row,
// in double, per model row; path kernels only multiply its float coefficients.
template<typename Pack> struct FractionalHybridBridge {
    using Traits=HybridBridgeTraits<Pack>;
    typename Pack::Inputs models;
    __device__ unsigned prepare(unsigned row,unsigned factor,const float* lower,
                                float* coefficients) const {
        if constexpr(Traits::count<=1 || !Traits::supported) return 0;
        else {
            constexpr unsigned N=Traits::count;
            static_assert(N<=32);
            double h[N],triangle[N*(N+1)/2];unsigned f[N],n=0,target=0;
            Traits::collect(models,row,h,f,n);
            for(unsigned i=0;i<N;++i) if(f[i]==factor) target=i;
            for(unsigned i=0;i<=target;++i) for(unsigned j=0;j<=i;++j) {
                double rho=0;
                for(unsigned k=0;k<=f[j];++k)
                    rho+=double(lower[f[i]*(f[i]+1)/2+k])*lower[f[j]*(f[j]+1)/2+k];
                const double sign=(h[i]>.5)==(h[j]>.5)?1.:-1.;
                double v=i==j?1.:rho*sign*2.*sqrt(h[i]*h[j])/(h[i]+h[j]);
                for(unsigned k=0;k<j;++k) v-=triangle[i*(i+1)/2+k]*triangle[j*(j+1)/2+k];
                triangle[i*(i+1)/2+j]=i==j?sqrt(fmax(v,0.)):
                    (triangle[j*(j+1)/2+j]>0?v/triangle[j*(j+1)/2+j]:0.);
            }
            for(unsigned j=0;j<=target;++j) coefficients[j]=float(triangle[target*(target+1)/2+j]);
            return target+1;
        }
    }
};
} // namespace ai_factory::workbench::simulation
