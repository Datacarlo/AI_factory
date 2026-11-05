#pragma once
#include "product/multi_asset_autocall/parameters.hpp"
#include <filesystem>
#include <vector>

namespace ai_factory::workbench::product {
std::vector<MultiAssetAutocallParameters> load_multi_asset_athena_autocalls(
    const std::filesystem::path& path);
std::vector<MultiAssetAutocallParameters> load_multi_asset_phoenix_autocalls(
    const std::filesystem::path& path);
std::vector<MultiAssetAutocallParameters> load_multi_asset_phoenix_memory_autocalls(
    const std::filesystem::path& path);
} // namespace ai_factory::workbench::product
