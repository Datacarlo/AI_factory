#include "product/multi_asset_autocall/dataset.hpp"
#include "tools/datasets/multi_asset_price_dataset.hpp"
#include "tools/datasets/artifact_io.hpp"
#include <fstream>
#include <stdexcept>
#include <string>
#include <chrono>

namespace w=ai_factory::workbench;
using J=nlohmann::ordered_json;
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){
    const auto dir=std::filesystem::temp_directory_path()/(
        "ai_factory_autocall_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directories(dir);
    const auto source=dir/"products.json";
    const J quote{{"mode","ratio_to_initial_spot"},{"unit","dimensionless"},{"observation","indicator(S_i(t)/S_i(0))"}};
    J base{{"database_id","multi_autocall_test"},{"row_count",1U},
        {"product_family","Multi-asset Athena Autocalls"},
        {"time_convention",{{"unit","business_day"},{"days_per_year",252U}}},
        {"quote_convention",{{"autocall_barrier",quote},{"protection_barrier",quote}}},
        {"products",J::array({{{"id","p1"},{"parameters",{
            {"maturity",252U},{"observation_interval",63U},{"autocall_barrier",1.05},
            {"protection_barrier",.7},{"annual_coupon_rate",.08},
            {"autocall_indicator","worst_of"},{"protection_indicator","basket"},{"loss_indicator","best_of"}}}}})}};
    const auto load=[&](int kind){
        if(kind==0)return w::product::load_multi_asset_athena_autocalls(source);
        if(kind==1)return w::product::load_multi_asset_phoenix_autocalls(source);
        return w::product::load_multi_asset_phoenix_memory_autocalls(source);
    };
    const auto reject=[&](J document,int kind){w::datasets::write_json_file(source,document);bool failed=false;try{load(kind);}catch(const std::exception&){failed=true;}require(failed,"invalid product accepted");};
    for(int kind=0;kind<3;++kind){
        J document=base;
        if(kind){document["product_family"]=kind==1?"Multi-asset Phoenix Autocalls":"Multi-asset Phoenix Memory Autocalls";document["quote_convention"]["coupon_barrier"]=quote;
            document["products"][0]["parameters"]["coupon_barrier"]=.8;document["products"][0]["parameters"]["coupon_indicator"]="basket";}
        w::datasets::write_json_file(source,document);auto values=load(kind);
        require(values.size()==1 && values[0].maturity_days==252 && values[0].autocall_barrier==1.05f,"loader changed values");
        require(values[0].loss_indicator==w::product::AutocallIndicator::best_of,"lost selector");
        auto bad=document;bad["quote_convention"]["autocall_barrier"]["mode"]="absolute";reject(bad,kind);
        bad=document;bad["time_convention"]["days_per_year"]=365;reject(bad,kind);
        bad=document;bad["products"][0]["parameters"]["maturity"]=253;reject(bad,kind);
        bad=document;bad["products"][0]["parameters"]["maturity"]=-252;reject(bad,kind);
        bad=document;bad["products"][0]["parameters"]["maturity"]=252.5;reject(bad,kind);
        bad=document;bad["products"][0]["parameters"]["observation_interval"]=0;reject(bad,kind);
        bad=document;bad["products"][0]["parameters"]["autocall_indicator"]="unknown";reject(bad,kind);
        if(kind){bad=document;bad["products"][0]["parameters"].erase("coupon_indicator");reject(bad,kind);}
        w::datasets::write_json_file(source,document);
        // Full writer chain retains source IDs and normalized conventions, without a fake call side.
        const auto model=dir/"models.json",rate=dir/"rates.json",corr=dir/"correlations.json";
        w::datasets::write_json_file(model,{{"database_id","model"},{"model_family","Black-Scholes"},{"models",J::array({{{"id","m1"}}})}});
        w::datasets::write_json_file(rate,{{"database_id","rate"},{"model_family","Constant rate"},{"models",J::array({{{"id","r1"}}})}});
        w::datasets::write_json_file(corr,{{"database_id","correlation"},{"correlations",J::array({{{"id","c1"}}})}});
        using E=w::RowIndexExpression;
        auto plan=w::compile_row_index_plan(E::aligned({E::source(0,1),E::source(1,1),E::source(2,1),E::source(3,1)}),4);
        w::datasets::MultiAssetPriceDatasetRecipe recipe{{model},source,corr,std::nullopt,dir/"prices.json",dir/"generation.yaml","",rate,""};
        w::datasets::write_multi_asset_price_dataset(recipe,plan,{1.02f},{.001f},128,42,J::object(),.1,.05);
        auto output=w::datasets::read_json_file(recipe.dataset_path);
        require(!output.contains("option_side"),"autocall is not a call/put option");
        require(output["quote_convention"]==document["quote_convention"],"lost normalization convention");
        require(output["results"][0]["product_id"]=="p1" && output["results"][0]["rate_model_id"]=="r1","bad output mapping");
    }
    std::filesystem::remove_all(dir);
}
