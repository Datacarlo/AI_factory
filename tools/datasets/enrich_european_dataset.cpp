// Reuse the generator writer on a finished dataset without rerunning pricing.
#include "tools/datasets/single_asset_json.hpp"
#include <iostream>

int main(int argc, char** argv) {
    using namespace ai_factory::workbench::datasets;
    try {
        if (argc != 3) throw std::invalid_argument("Usage: enrich_european_dataset INPUT.json OUTPUT.json");
        std::ifstream input(argv[1]);
        auto document = nlohmann::ordered_json::parse(input);
        if (document.value("product_family", "") != "European Options") throw std::invalid_argument("Expected European Options");
        auto rows = std::move(document.at("results"));
        document.erase("results");
        if (!rows.is_array() || document.at("row_count") != rows.size())
            throw std::invalid_argument("Result count mismatch");
        const bool include_iv = has_european_volatility(document);
        write_single_asset_json(std::filesystem::absolute(argv[2]), document, rows.size(),
            [&](size_t i) {
                auto& row = rows[i];
                row.erase("implied_volatility_status"); // Previous publication format.
                if (!include_iv) row.erase("implied_volatility");
                return std::move(row);
            });
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
