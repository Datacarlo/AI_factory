// Five independent mother slots of one equity family, with a correlated CIR rate.
#pragma once
#include "multi_asset/rate_adjusted_model_group.cuh"
#include "model/equity/markovian/black_scholes/dataset.hpp"
#include "model/equity/markovian/black_scholes/dynamics_impl.cuh"
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/heston/dynamics_impl.cuh"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "model/equity/markovian/sabr/dynamics_impl.cuh"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "model/equity/rough/quadratic_rough_heston/causal_fft_dynamics.cuh"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "model/equity/rough/hybrid_multi_asset.cuh"
#include "model/fixed_income/cir/dataset.hpp"
#include "model/fixed_income/cir/correlated_dynamics.cuh"
#include "tools/pricing/cross_asset_price_generation.cuh"
#include "tools/datasets/multi_asset_price_dataset.hpp"
#include <array>
#include <iostream>

namespace ai_factory::workbench::offline::pricing {
// Separate one-asset groups keep all five independent source selectors intact.
template<typename Dynamics> using FiveMarkov=simulation::MultiAssetPack<
    simulation::RateAdjustedModelGroup<Dynamics,1>,simulation::RateAdjustedModelGroup<Dynamics,1>,
    simulation::RateAdjustedModelGroup<Dynamics,1>,simulation::RateAdjustedModelGroup<Dynamics,1>,
    simulation::RateAdjustedModelGroup<Dynamics,1>>;
template<typename Policy> using FiveRough=simulation::MultiRoughPack<Policy,Policy,Policy,Policy,Policy>;

template<typename Host,typename Loader>
void load_homogeneous_slots(Host& host, Loader load, const std::vector<std::filesystem::path>& paths, unsigned slot=0) {
    if constexpr (requires { host.models; }) {
        host.models=load(paths.at(slot));
        if(host.models.size()!=100U) throw std::invalid_argument("Expected 100 equity rows per slot");
        load_homogeneous_slots(host.tail,load,paths,slot+1U);
    }
}

template<typename EquityPack,typename RoughPack,typename Product,unsigned TeamSize=1,typename LoadModel,typename LoadProduct>
int generate_homogeneous_equity_cir_prices(
    int argc,char** argv,LoadModel load_model,LoadProduct load_product,
    const std::string& model_path,const std::string& model_name,
    const std::vector<std::string>& factor_suffixes,
    const std::string& product_path,const std::string& output_path,
    const std::string& generation_path,const std::string& option_side,
    std::uint64_t seed,CrossAssetBatchGeometry geometry={128U,0U,256U,1U,1U,1U},
    std::size_t path_chunk=8192U
) {
    using E=RowIndexExpression;
    using Rate=model::fixed_income::cir::CorrelatedRatePolicy;
    std::size_t paths=1U<<20U,limit=100000U,offset=0,product_index=0,correlation_index=0;
    bool pilot=false;
    if(argc==6 && std::string_view(argv[1])=="--pilot") {
        pilot=true; product_index=std::stoull(argv[2]); correlation_index=std::stoull(argv[3]);
        paths=std::stoull(argv[4]); limit=std::stoull(argv[5]);
        if(product_index>=100U || correlation_index>=10U || !limit || limit>100U
            || paths<256U || paths%256U)
            throw std::invalid_argument("Invalid pilot selection or path count");
        offset=product_index*1000U+correlation_index*100U;
    } else if(argc!=1) throw std::invalid_argument("Usage: generator [--pilot product_index correlation_index paths rows]");
    std::vector<std::filesystem::path> models;
    for(unsigned i=0;i<5;++i) models.emplace_back("datasets/parameters/model/"+model_path+
        "/factor_parameters/"+model_name+"_multi100_core_slot"+std::to_string(i)+"_01.json");
    const std::string rate_path="datasets/parameters/model/fixed_income/cir/factor_parameters/cir_multi100_core_01.json";
    const std::string corr_path="datasets/parameters/correlation/five_"+model_name+"_cir_01.json";
    MultiAssetHostModels<EquityPack> equity;
    MultiRoughHostModels<RoughPack> rough;
    if constexpr(EquityPack::kAssetCount) load_homogeneous_slots(equity,load_model,models);
    else load_homogeneous_slots(rough,load_model,models);
    auto rates=model::fixed_income::cir::load_models(rate_path);
    auto products=load_product(product_path);
    auto correlations=correlation::load_matrices(corr_path);
    if(rates.size()!=100U || products.size()!=100U || correlations.size()!=10U)
        throw std::invalid_argument("Expected 100 rates, 100 products and 10 correlations");
    std::vector<std::string> external,internal;
    for(unsigned i=0;i<5;++i) for(const auto& f:factor_suffixes)
        external.push_back(model_name+"_"+std::to_string(i)+"."+f);
    external.push_back("cir.rate");
    if constexpr(RoughPack::kAssetCount) {
        std::vector<std::string> variance,spot;
        RoughPack::append_factor_names(external,0,variance,spot);
        internal=variance; internal.push_back("cir.rate");
        internal.insert(internal.end(),spot.begin(),spot.end());
    } else internal=external;
    if(pilot) { products={products.at(product_index)}; correlations={correlations.at(correlation_index)}; }
    auto plan=compile_row_index_plan(E::cartesian({E::source(6,pilot?1U:100U),E::source(7,pilot?1U:10U),
        E::aligned({E::source(0,100),E::source(1,100),E::source(2,100),
                    E::source(3,100),E::source(4,100),E::source(5,100)})}),8);
    if(!pilot && plan.result_count!=100000U) throw std::logic_error("Expected 100000 rows");
    // Pilot selects one product/correlation and retains native model indices and row seeds.
    if(pilot) plan.result_count=static_cast<std::uint32_t>(limit);
    const auto run=execute_cross_asset_prices<EquityPack,RoughPack,Rate,Product,TeamSize>(
        equity,rough,rates,products,correlations,external,internal,plan,paths,
        std::min(paths,path_chunk),1.0f/504.0f,2U,32U,seed+offset,{},geometry);
    if(pilot) {
        std::cout.precision(9);
        std::cout<<"first_row="<<offset<<" rows="<<limit<<" paths="<<paths
                 <<" wall_seconds="<<run.wall_seconds<<" kernel_seconds="<<run.kernel_seconds<<'\n';
        for(std::size_t i=0;i<run.prices.size();++i)
            std::cout<<offset+i<<' '<<run.prices[i]<<' '<<run.standard_errors[i]<<'\n';
        return 0;
    }
    datasets::MultiAssetPriceDatasetRecipe recipe{
        models,product_path,corr_path,std::nullopt,output_path,generation_path,"",rate_path,option_side};
    datasets::write_multi_asset_price_dataset(recipe,plan,run.prices,run.standard_errors,paths,seed,
        {{"engine","homogeneous_five_equity_cir_mc"},{"dt",1.0/504.0},{"steps_per_business_day",2},
         {"team_size",TeamSize},{"rows_per_launch",geometry.rough_rows_per_launch},
         {"streams",geometry.streams},{"threads_per_block",geometry.threads_per_block},
         {"paths_per_team",geometry.paths_per_team},{"path_chunk_size",path_chunk},
         {"factor_order",external},{"rate_scheme","CIR full-truncation Euler, trapezoid discount integral"}},
        run.wall_seconds,run.kernel_seconds);
    return 0;
}
}
