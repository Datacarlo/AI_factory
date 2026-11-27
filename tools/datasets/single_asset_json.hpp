// Single-asset publication format. Simulation and row selection live elsewhere.
#pragma once

#include "tools/datasets/json_header.hpp"
#include "tools/datasets/european_implied_volatility.hpp"
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
        if (has_european_volatility(header)) {
            constexpr size_t batch_size = 4096;
            for (size_t first = 0; first < count; first += batch_size) {
                std::vector<nlohmann::ordered_json> rows;
                const size_t size = std::min(batch_size, count - first);
                rows.reserve(size);
                for (size_t i = 0; i < size; ++i) rows.push_back(make_row(first + i));
                enrich_european_rows(header, rows);
                for (size_t i = 0; i < size; ++i)
                    write_single_asset_observation(stream, rows[i], first + i + 1 == count);
            }
        } else {
            for (size_t i = 0; i < count; ++i)
                write_single_asset_observation(stream, make_row(i), i + 1 == count);
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
