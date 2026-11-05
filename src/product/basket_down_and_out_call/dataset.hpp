#pragma once
#include "product/basket_down_and_out_call/parameters.hpp"
#include "datasets/validation.hpp"
#include <cmath>
#include <stdexcept>
namespace ai_factory::workbench::product {
inline std::vector<BasketDownAndOutCallParameters> load_basket_down_and_out_calls(
    const std::filesystem::path& path
) {
    return datasets::load_parameter_rows<BasketDownAndOutCallParameters>(path,
        datasets::ParameterDatasetFamily::Product,"Arithmetic-mean basket down-and-out call",
        [](const nlohmann::json& p,const std::string& prefix) {
            const auto count=[&](const char* key, bool required) -> std::uint32_t {
                if(!p.contains(key) && !required) return 0;
                const auto& v=p.at(key);
                if(!v.is_number_integer() || v.get<double>()<=0 || v.get<double>()>8192)
                    throw std::invalid_argument(prefix+key+" must be a positive integer <= 8192");
                return v.get<std::uint32_t>();
            };
            BasketDownAndOutCallParameters r{p.at("strike").get<float>(),p.at("lower_barrier").get<float>(),
                count("maturity",true),count("monitoring_interval_days",false),count("monitoring_interval_steps",false)};
            if(!std::isfinite(r.strike) || r.strike<0 || !std::isfinite(r.lower_barrier) || r.lower_barrier<=0
                || (r.monitoring_interval_days==0)==(r.monitoring_interval_steps==0))
                throw std::invalid_argument(prefix+"invalid basket barrier contract");
            return r;
        });
}
}
