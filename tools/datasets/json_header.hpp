// Order descriptive fields, source identities, then conventions (header only).
#pragma once

#include <nlohmann/json.hpp>

namespace ai_factory::workbench::datasets {

inline nlohmann::ordered_json ordered_dataset_header(
    const nlohmann::ordered_json& header
) {
    nlohmann::ordered_json ordered = nlohmann::ordered_json::object();
    for (const char* field : {
            "database_id", "row_count", "model_family", "model_families",
            "product_family", "option_side", "rate_model_family", "curve_family",
            "model_dataset", "model_datasets", "product_dataset",
            "rate_model_dataset", "curve_dataset", "correlation_dataset",
            "source_model_dataset", "time_convention", "quote_convention"}) {
        if (header.contains(field)) ordered[field] = header.at(field);
    }
    for (const auto& [field, value] : header.items()) {
        if (!ordered.contains(field)) ordered[field] = value;
    }
    return ordered;
}

} // namespace ai_factory::workbench::datasets
