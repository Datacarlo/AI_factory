#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"

#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    const auto directory=std::filesystem::temp_directory_path()
        / "ai_factory_multi_asset_rate_dataset_test";
    std::filesystem::create_directories(directory);
    const auto output=directory/"price.json";
    const auto receipt=directory/"generation.yaml";
    const auto correlation=directory/"correlation.json";
    datasets::write_json_file(correlation, {
        {"database_id", "synthetic_correlation_01"},
        {"correlations", {{{"id", "000001"}}}}
    });
    const auto equity=directory/"equity.json";
    const auto rate=directory/"rate.json";
    const auto product=directory/"product.json";
    const auto curve=directory/"curve.json";
    nlohmann::ordered_json models=nlohmann::ordered_json::array();
    for(std::uint32_t i=1U;i<=1000U;++i) {
        char id[7];
        std::snprintf(id,sizeof(id),"%06u",i);
        models.push_back({{"id",id}});
    }
    datasets::write_json_file(equity,{
        {"database_id","equity_01"},{"model_family","Heston"},{"models",models}});
    datasets::write_json_file(rate,{
        {"database_id","g2_plus_plus_01"},{"model_family","G2++"},{"models",models}});
    datasets::write_json_file(product,{
        {"database_id","product_01"},
        {"product_family","Basket Options"},
        {"quote_convention",{{"strike",{{"mode","ratio_to_initial_basket"}}}}},
        {"time_convention",{{"unit","business_day"},{"days_per_year",252}}},
        {"products",{{{"id","000001"}}}}});
    datasets::write_json_file(curve,{
        {"database_id","curve_01"},{"curve_family","Flat"},{"curves",models}});
    const datasets::MultiAssetPriceDatasetRecipe recipe{
        {equity,equity},product,correlation,curve,output,receipt,"",rate};
    using Expression=RowIndexExpression;
    const auto plan=compile_row_index_plan(
        Expression::cartesian({
            Expression::aligned({Expression::source(0U,1000U),
                                 Expression::source(1U,1000U),
                                 Expression::source(2U,1000U),
                                 Expression::source(5U,1000U)}),
            Expression::source(3U,1U),
            Expression::source(4U,1U)}),6U);
    std::vector<float> prices,errors;
    for(std::uint32_t j=0U;j<plan.result_count;++j) {
        prices.push_back(0.03f+0.00001f*j);
        errors.push_back(0.001f);
    }
    auto compact_recipe=recipe;
    compact_recipe.dataset_path=directory/"price_compact.json";
    compact_recipe.option_side="put";
    datasets::write_multi_asset_price_dataset(
        compact_recipe,plan,prices,errors,128U,42U,
        {{"engine","cross_asset_mc"}},0.01,0.005);
    const auto compact_document=datasets::read_json_file(
        compact_recipe.dataset_path);
    if(compact_document.at("time_convention") != datasets::read_json_file(product).at("time_convention")
        || compact_document.at("option_side")!="put"
        ||compact_document.at("results").size()!=1000U
        ||compact_document.at("results").at(999).at("model_ids").size()!=2U
        ||compact_document.at("rate_model_dataset").at("id")
            !="g2_plus_plus_01"
        ||compact_document.at("results").at(0).at("rate_model_id")
            !="000001"
        ||compact_document.at("results").at(0).at("curve_id")
            !="000001")
        throw std::runtime_error("compact dataset provenance incomplete");
    const auto text = compact_document.dump();
    if (text.find("option_side") < text.find("product_family")
        || text.find("option_side") > text.find("rate_model_family")
        || text.find("quote_convention") < text.find("correlation_dataset"))
        throw std::runtime_error("multi-asset header field order changed");
    if (compact_document.contains("timing") || compact_document.contains("price_construction"))
        throw std::runtime_error("execution metadata leaked into JSON");
    for (std::size_t j=0;j<prices.size();++j) {
        const auto& row=compact_document.at("results").at(j);
        if (row.contains("outputs") || row.contains("seed") || row.contains("id")
            || row.at("price").get<float>() != prices[j]
            || row.at("standard_error").get<float>() != errors[j])
            throw std::runtime_error("compact result serialization changed values");
    }
    bool rejected=false;
    try {
        auto invalid=plan;
        invalid.selectors[2].source_count=1000000U;
        datasets::write_multi_asset_price_dataset(
            recipe,invalid,prices,errors,128U,42U,
            {{"engine","cross_asset_mc"}},0.01,0.005);
    } catch(const std::invalid_argument&) { rejected=true; }
    if(!rejected) throw std::runtime_error("invalid rate selector accepted");
    std::filesystem::remove_all(directory);
}
