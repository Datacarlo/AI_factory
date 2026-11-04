// Single-asset publication format. Simulation and row selection live elsewhere.
#pragma once

#include "tools/datasets/json_header.hpp"
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <ostream>
#include <stdexcept>
#include <system_error>

namespace ai_factory::workbench::datasets {

inline nlohmann::ordered_json single_asset_observation(
    std::size_t id,
    const nlohmann::ordered_json& model,
    const nlohmann::ordered_json& product,
    const nlohmann::ordered_json* curve = nullptr
) {
    nlohmann::ordered_json row = {
        {"id", id}, {"model_id", model.at("id")}, {"product_id", product.at("id")},
    };
    if (curve != nullptr) row["curve_id"] = curve->at("id");
    row["model_parameters"] = model.at("parameters");
    row["product_parameters"] = product.at("parameters");
    if (curve != nullptr) row["curve_parameters"] = curve->at("parameters");
    return row;
}

inline void write_single_asset_header(
    std::ostream& stream, const nlohmann::ordered_json& header
) {
    if (!header.is_object() || header.empty() || header.contains("results")) {
        throw std::invalid_argument("Single-asset header must exclude results.");
    }
    auto text = ordered_dataset_header(header).dump(2);
    text.resize(text.size() - 2U); // Replace the final newline/brace with results.
    stream << text << ",\n  \"results\": [\n";
}

inline void write_single_asset_observation(
    std::ostream& stream, const nlohmann::ordered_json& row, bool last
) {
    stream << "    {\n";
    std::size_t field = 0U;
    for (const auto& [name, value] : row.items()) {
        stream << "      " << nlohmann::json(name).dump() << ": " << value.dump();
        if (++field != row.size()) stream << ',';
        stream << '\n';
    }
    stream << "    }" << (last ? "\n" : ",\n");
}

template<class MakeRow>
void write_single_asset_json(
    const std::filesystem::path& path,
    const nlohmann::ordered_json& header,
    std::size_t count,
    MakeRow&& make_row
) {
    std::filesystem::create_directories(path.parent_path());
    const std::filesystem::path temporary = path.string() + ".tmp";
    try {
        std::ofstream stream(temporary);
        if (!stream) throw std::runtime_error("Cannot open " + temporary.string());
        write_single_asset_header(stream, header);
        for (std::size_t index = 0U; index < count; ++index) {
            write_single_asset_observation(stream, make_row(index), index + 1U == count);
        }
        stream << "  ]\n}\n";
        stream.close();
        if (!stream) throw std::runtime_error("Cannot write " + temporary.string());
        std::filesystem::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        throw;
    }
}

} // namespace ai_factory::workbench::datasets
