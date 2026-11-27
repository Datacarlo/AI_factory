#pragma once
#include "implied_volatility/european.hpp"
#include <nlohmann/json.hpp>
#include <vector>

namespace ai_factory::workbench::datasets {
inline bool has_european_volatility(const nlohmann::ordered_json& header) {
    return header.value("product_family", "") == "European Options"
        && header.at("model_family") != "Black-Scholes";
}

inline void enrich_european_rows(const nlohmann::ordered_json& header,
                                 std::vector<nlohmann::ordered_json>& rows) {
    using namespace implied_volatility;
    const auto& time = header.at("time_convention");
    const float days = time.at("days_per_year");
    const auto& side = header.at("option_side");
    const auto& mode = header.at("quote_convention").at("strike").at("mode");
    if ((side != "call" && side != "put") || time.at("unit") != "business_day"
        || !(days > 0) || (mode != "ratio_to_central_spot" && mode != "absolute"))
        throw std::invalid_argument("Unsupported European option convention");

    std::vector<Quote> quotes;
    quotes.reserve(rows.size());
    for (const auto& row : rows) {
        const auto& model = row.at("model_parameters");
        const auto& product = row.at("product_parameters");
        const float spot = model.at("spot"), strike = product.at("strike");
        // Match the FP32 contract level resolved by the pricing kernel.
        quotes.push_back({spot, mode == "absolute" ? strike : strike * spot,
            product.at("maturity").get<float>() / days, model.at("risk_free_rate"),
            model.at("dividend_yield"), row.at("price")});
    }
    std::vector<float> results(rows.size());
    invert_batch(quotes, side == "call", results);
    for (size_t i = 0; i < rows.size(); ++i) {
        if (!std::isfinite(results[i]) || results[i] < 0.0f)
            throw std::runtime_error("Cannot compute implied volatility for row " + rows[i].at("id").dump());
        rows[i]["implied_volatility"] = results[i];
    }
}
}
