// Deterministic correlation scenarios for five equal-model equities and CIR.
#pragma once
#include "correlation/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace ai_factory::workbench::datasets {
inline void write_homogeneous_correlations(
    const std::string& model, const std::vector<std::string>& asset_factors,
    const std::string& id
) {
    if(asset_factors.size()!=1U && asset_factors.size()!=2U)
        throw std::invalid_argument("Expected one or two factors per equity");
    const std::size_t d=5U*asset_factors.size()+1U;
    std::vector<std::string> names;
    for(unsigned i=0;i<5;++i)
        for(const auto& f:asset_factors)
            names.push_back(model+"_"+std::to_string(i)+"."+f);
    names.push_back("cir.rate");
    constexpr std::array<std::array<double,5>,10> a{{
        {0,0,0,0,0},{.30,.30,.30,.30,.30},{.55,-.55,.55,-.55,.55},
        {.75,.75,.75,.75,.75},{.85,.68,-.55,-.35,.10},
        {.90,.65,.40,-.25,-.80},{.80,.80,.80,-.80,-.80},
        {.35,-.85,.90,.15,-.55},{.95,.95,.95,.95,.95},
        {.92,.92,-.92,-.92,-.92}}};
    constexpr std::array<double,10> rate{0,.20,-.15,-.25,.25,-.30,.35,-.40,.40,-.45};
    nlohmann::ordered_json rows=nlohmann::ordered_json::array();
    for(unsigned s=0;s<10;++s) {
        // Common driver + five spot residuals + five vol residuals + rate residual.
        std::vector<std::array<double,12>> loading(d);
        for(unsigned i=0;i<5;++i) {
            auto& spot=loading[i*asset_factors.size()];
            spot[0]=a[s][i]; spot[1+i]=std::sqrt(1-a[s][i]*a[s][i]);
            if(asset_factors.size()==2U) {
                const double rho=-.55-.4*((s+2*i)%10)/9.;
                auto& vol=loading[i*2+1];
                for(unsigned k=0;k<12;++k) vol[k]=rho*spot[k];
                vol[6+i]=std::sqrt(1-rho*rho);
            }
        }
        loading.back()[0]=rate[s]; loading.back()[11]=std::sqrt(1-rate[s]*rate[s]);
        std::vector<double> dense(d*d);
        nlohmann::ordered_json matrix=nlohmann::ordered_json::array();
        for(std::size_t i=0;i<d;++i) {
            nlohmann::ordered_json row=nlohmann::ordered_json::array();
            for(std::size_t j=0;j<d;++j) {
                double v=0; for(unsigned k=0;k<12;++k) v+=loading[i][k]*loading[j][k];
                dense[i*d+j]=v; row.push_back(v);
            }
            matrix.push_back(row);
        }
        correlation::packed_cholesky(dense,d);
        rows.push_back({{"id",format_row_id(s)},{"factors",names},{"matrix",matrix}});
    }
    const std::string output="datasets/parameters/correlation/"+id+".json";
    const std::string leaf="catalog/parameters/correlation/"+id;
    write_json_file(output,{{"database_id",id},{"row_count",10},{"correlations",rows}});
    correlation::load_matrices(output);
    write_yaml_document(leaf+"/recipe.yaml",{
        {"schema_version",1},{"kind","correlation_parameters"},{"dataset_id",id},
        {"generator",leaf+"/generator.cpp"},{"output",{{"path",output},{"format","json"}}},
        {"row_count",10},{"factor_order",names},{"construction",{
            {"method","Gram matrix of common and independent residual drivers"},
            {"spot_common_factor_loadings",a},{"rate_common_driver_loadings",rate},
            {"own_spot_volatility_correlation_range",asset_factors.size()==2U
                ? nlohmann::ordered_json({-.95,-.55}) : nlohmann::ordered_json::array()},
            {"positive_definite","Cholesky verified"}}}});
    write_generation_receipt(leaf+"/generation.yaml",10,{{"method","latent-factor Gram matrix"}},0,0);
}
}
