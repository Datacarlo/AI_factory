// Every factor-only model base loads with neutral legacy correlation fields.
#include "model/equity/markovian/bates/dataset.hpp"
#include "model/equity/markovian/heston/dataset.hpp"
#include "model/equity/markovian/sabr/dataset.hpp"
#include "model/equity/rough/quadratic_rough_heston/dataset.hpp"
#include "model/equity/rough/rough_bergomi/dataset.hpp"
#include "model/fixed_income/cir/dataset.hpp"
#include "datasets/validation.hpp"

#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    using namespace ai_factory::workbench;
    const auto check = [](auto load, const std::filesystem::path& path,
                          std::size_t expected_rows = 1000U) {
        std::ifstream stream(path);
        if (!stream) {
            std::cout << "Skipping external factor dataset test: "
                      << path << '\n';
            std::exit(77);
        }
        nlohmann::json document;
        stream >> document;
        if (document.at("parameterization") != "factorized"
            || document.at("models").size() != expected_rows)
            throw std::runtime_error("Invalid factor dataset: " + path.string());
        for (const auto& row : document.at("models")) {
            const auto& p = row.at("parameters");
            if (p.contains("risk_free_rate") || p.contains("rho")
                || p.contains("correlation"))
                throw std::runtime_error("Exogenous field in " + path.string());
        }
        const auto models = load(path);
        if (models.size() != expected_rows)
            throw std::runtime_error("Wrong factor count: " + path.string());
        for (const auto& model : models) {
            if constexpr (requires { model.spot; })
                if (model.spot != 1.0f)
                    throw std::runtime_error("Multi-asset equity spot must equal one");
            if constexpr (requires { model.risk_free_rate; })
                if (model.risk_free_rate != 0.0f)
                    throw std::runtime_error("Equity rate was not neutral");
            if constexpr (requires { model.rho; })
                if (model.rho != 0.0f)
                    throw std::runtime_error("Internal rho was not neutral");
            if constexpr (requires { model.correlation; })
                if (model.correlation != 0.0f)
                    throw std::runtime_error("Internal correlation was not neutral");
            if constexpr (requires { model.process.correlation; })
                if (model.process.correlation != 0.0f)
                    throw std::runtime_error("Rate correlation was not neutral");
        }
    };
    check(model::equity::bates::load_models, "datasets/model/equity/markovian/bates/factor_parameters/bates_multi_core_01.json", 250U);
    check(model::equity::heston::load_models, "datasets/model/equity/markovian/heston/factor_parameters/heston_multi_core_01.json", 250U);
    check(model::equity::sabr::load_models, "datasets/model/equity/markovian/sabr/factor_parameters/sabr_multi_core_01.json", 250U);
    check(model::equity::quadratic_rough_heston::load_models, "datasets/model/equity/rough/quadratic_rough_heston/factor_parameters/quadratic_rough_heston_multi_core_01.json", 250U);
    check(model::equity::rough_bergomi::load_models, "datasets/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_multi_core_01.json", 250U);
    check(model::fixed_income::cir::load_models, "datasets/model/fixed_income/cir/factor_parameters/cir_multi_core_01.json", 250U);
    std::ifstream invalid_source(
        "datasets/model/equity/markovian/heston/factor_parameters/heston_multi_core_01.json");
    nlohmann::json invalid;
    invalid_source >> invalid;
    invalid.at("models").at(0).at("parameters")["rho"] = 0.3;
    bool rejected = false;
    try { datasets::validate_model_dataset(invalid); }
    catch (const std::invalid_argument&) { rejected = true; }
    if (!rejected)
        throw std::runtime_error("Factorized row accepted embedded rho");

}
