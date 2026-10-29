// Ten-factor positive-definite correlations for five equities and CIR.
// Quadratic rough Heston has one shared spot/feedback Brownian factor.
#include "correlation/dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    constexpr std::size_t dimension=10U;
    constexpr std::size_t latent_count=11U;
    constexpr std::array<const char*,dimension> factors{
        "heston.spot","heston.variance",
        "bates.spot","bates.variance",
        "sabr.spot","sabr.volatility",
        "quadratic_rough_heston.spot",
        "rough_bergomi.spot","rough_bergomi.variance",
        "cir.rate"};
    // Each spot loads one common driver plus its own independent residual.
    // The final two scenarios cover near-common movement and opposing groups.
    constexpr std::array<std::array<double,5U>,10U> spot_common_loadings{{
        {0.00, 0.00, 0.00, 0.00, 0.00},
        {0.30, 0.30, 0.30, 0.30, 0.30},
        {0.55,-0.55, 0.55,-0.55, 0.55},
        {0.75, 0.75, 0.75, 0.75, 0.75},
        {0.85, 0.68,-0.55,-0.35, 0.10},
        {0.90, 0.65, 0.40,-0.25,-0.80},
        {0.80, 0.80, 0.80,-0.80,-0.80},
        {0.35,-0.85, 0.90, 0.15,-0.55},
        {0.95, 0.95, 0.95, 0.95, 0.95},
        {0.92, 0.92,-0.92,-0.92,-0.92},
    }};
    constexpr std::array<double,10U> rate_loadings{
        0.0,0.20,-0.15,-0.25,0.25,-0.30,0.35,-0.40,0.40,-0.45};
    nlohmann::ordered_json rows=nlohmann::ordered_json::array();
    for(std::size_t scenario=0U;scenario<10U;++scenario) {
        std::array<std::array<double,latent_count>,dimension> loadings{};
        constexpr std::array<std::size_t,5U> spot_indices{0U,2U,4U,6U,7U};
        constexpr std::array<std::size_t,4U> vol_spot_indices{0U,2U,4U,7U};
        constexpr std::array<std::size_t,4U> volatility_indices{1U,3U,5U,8U};
        for(std::size_t asset=0U;asset<5U;++asset) {
            auto& spot=loadings[spot_indices[asset]];
            const double loading=spot_common_loadings[scenario][asset];
            spot[0U]=loading;
            spot[1U+asset]=std::sqrt(1.0-loading*loading);
        }
        for(std::size_t asset=0U;asset<4U;++asset) {
            auto& spot=loadings[vol_spot_indices[asset]];
            auto& volatility=loadings[volatility_indices[asset]];
            const double own_rho=-0.55-0.4*static_cast<double>(
                (scenario+2U*asset)%10U)/9.0;
            for(std::size_t factor=0U;factor<latent_count;++factor)
                volatility[factor]=own_rho*spot[factor];
            volatility[6U+asset]=std::sqrt(1.0-own_rho*own_rho);
        }
        loadings[9U][0U]=rate_loadings[scenario];
        loadings[9U][10U]=std::sqrt(
            1.0-rate_loadings[scenario]*rate_loadings[scenario]);
        std::vector<double> dense(dimension*dimension,0.0);
        nlohmann::ordered_json matrix=nlohmann::ordered_json::array();
        for(std::size_t row=0U;row<dimension;++row) {
            nlohmann::ordered_json values=nlohmann::ordered_json::array();
            for(std::size_t column=0U;column<dimension;++column) {
                double value=0.0;
                for(std::size_t latent=0U;latent<latent_count;++latent)
                    value+=loadings[row][latent]*loadings[column][latent];
                dense[row*dimension+column]=value;
                values.push_back(value);
            }
            matrix.push_back(std::move(values));
        }
        correlation::packed_cholesky(dense,dimension);
        rows.push_back({{"id",datasets::format_row_id(scenario)},
            {"factors",factors},{"matrix",std::move(matrix)}});
    }
    constexpr auto path=
        "datasets/correlation/five_equity_cir_01.json";
    datasets::write_json_file(path,{
        {"database_id","five_equity_cir_01"},
        {"row_count",rows.size()},
        {"correlations",rows}});
    correlation::load_matrices(path);
    datasets::write_yaml_document(
        "catalog/correlation/five_equity_cir_01/recipe.yaml",{
            {"schema_version",1},
            {"kind","correlation_parameters"},
            {"dataset_id","five_equity_cir_01"},
            {"generator","catalog/correlation/five_equity_cir_01/generator.cpp"},
            {"output",{{"path",path},{"format","json"}}},
            {"row_count",10},
            {"factor_order",factors},
            {"construction",{
                {"method","latent common spot driver and independent residuals"},
                {"quadratic_rough_heston_shared_spot_feedback_factor",true},
                {"spot_common_factor_loadings",spot_common_loadings},
                {"rate_common_driver_loadings",rate_loadings},
                {"own_spot_volatility_correlation_range",{-0.95,-0.55}},
                {"positive_definite","verified by Cholesky per matrix"}}}});
    datasets::write_generation_receipt(
        "catalog/correlation/five_equity_cir_01/generation.yaml",
        rows.size(),{{"method","latent-factor Gram matrix"}},0.0,0.0);
}
