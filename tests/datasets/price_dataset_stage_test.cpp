// Verify price-dataset row assembly, references and publication metadata.
#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/price_dataset.hpp"
#include "tools/datasets/single_asset_json.hpp"
#include "datasets/validation.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_price_dataset_stage_test";
    std::filesystem::remove_all(directory);
    const auto reference = [](const char* id) {
        return nlohmann::ordered_json{
            {"database_id", id},
        };
    };
    auto model = reference("model_01");
    model["model_family"] = "Test model";
    model["models"] = {{{"id", "000001"}, {"parameters", {{"x", 1.0}}}}};
    auto product = reference("product_01");
    product["product_family"] = "Test product";
    product["time_convention"] = {
        {"unit", "business_day"}, {"days_per_year", 252U}
    };
    product["products"] = {
        {{"id", "000001"}, {"parameters", {{"maturity", 252U}}}}
    };
    write_json_file(directory / "model.json", model);
    write_json_file(directory / "product.json", product);
    write_analytical_price_dataset(
        directory / "model.json",
        directory / "product.json",
        PriceConstruction::Aligned,
        std::vector<float>{1.25f},
        directory / "price.json",
        directory / "generation.yaml",
        "https://datasets.ai-factory.example/test/price.json",
        "test formula",
        {{"block_count", 1U}},
        0.01,
        0.001
    );
    const auto result = read_json_file(directory / "price.json");
    validate_price_dataset(result);
    if (result.at("row_count") != 1U
        || result.at("model_dataset").at("id") != "model_01"
        || result.at("results").at(0).at("price") != 1.25f) {
        throw std::runtime_error("price artifact assembly failed");
    }
    const auto& observation = result.at("results").at(0);
    if (observation.at("id") != 0U
        || observation.at("model_parameters") != model.at("models").at(0).at("parameters")
        || observation.at("product_parameters") != product.at("products").at(0).at("parameters")
        || result.at("time_convention") != product.at("time_convention")) {
        throw std::runtime_error("Inline price inputs or time convention changed");
    }
    std::ifstream yaml(directory / "generation.yaml");
    const std::string receipt{std::istreambuf_iterator<char>(yaml), {}};
    if (receipt.find("status: \"complete\"") == std::string::npos
        || receipt.find("paths_per_price: 0") == std::string::npos
        || receipt.find("validation:") != std::string::npos) {
        throw std::runtime_error("price generation receipt assembly failed");
    }
    bool certification_rejected = false;
    try {
        write_monte_carlo_price_dataset(
            directory / "model.json", directory / "product.json",
            PriceConstruction::Aligned, {1.0f}, {0.01f}, "Philox",
            directory / "price.json", directory / "generation.yaml",
            "https://datasets.ai-factory.example/test/price.json", "test MC",
            32U, "", {{"block_count", 1U}},
            {{"validation", {{"verified", true}}}}, 1U, 0.01, 0.001
        );
    } catch (const std::invalid_argument&) {
        certification_rejected = true;
    }
    if (!certification_rejected || read_json_file(directory / "price.json") != result) {
        throw std::runtime_error("A validation override must fail before replacing an artifact");
    }
    const nlohmann::ordered_json methodology = {
        {"standard_error_scope", "in-sample cashflow dispersion only"}
    };
    write_monte_carlo_price_dataset(
        directory / "model.json", directory / "product.json",
        PriceConstruction::Aligned, {1.0f}, {0.01f}, "Philox",
        directory / "price.json", directory / "generation.yaml",
        "https://datasets.ai-factory.example/test/price.json", "test MC",
        32U, "", {{"block_count", 1U}}, methodology, 1U, 0.01, 0.001
    );
    const auto compact = read_json_file(directory / "price.json");
    validate_price_dataset(compact);
    if (compact.contains("methodology") || compact.contains("timing")
        || compact.at("results").at(0).at("standard_error") != 0.01f) {
        throw std::runtime_error("Monte Carlo price data are not compact");
    }
    // Several distinct inputs: preserve the Cartesian order and quoted levels.
    model["models"].push_back({{"id", "model-second"}, {"parameters", {{"x", 1.7}}}});
    product["products"].push_back({{"id", "product-second"}, {"parameters",
        {{"maturity", 504U}, {"strike", 0.9}, {"observation_days", {126U, 252U, 504U}}}}});
    product["quote_convention"] = {{"strike", {{"mode", "spot_ratio"}}}};
    auto curve = reference("curve_01");
    curve["curve_family"] = "Flat";
    curve["curves"] = {
        {{"id", "curve-first"}, {"parameters", {{"rate", 0.02}}}},
        {{"id", "curve-second"}, {"parameters", {{"rate", 0.05}}}}
    };
    write_json_file(directory / "model.json", model);
    write_json_file(directory / "product.json", product);
    write_json_file(directory / "curve.json", curve);
    const std::vector<float> prices{0.f, 1.f, 2.f, 3.f, 4.f, 5.f, 6.f, 7.f};
    write_analytical_price_dataset(
        directory / "model.json", directory / "curve.json", directory / "product.json",
        PriceConstruction::CartesianProduct, prices, directory / "curve-price.json",
        directory / "curve-generation.yaml", "https://example.org/test.json", "test",
        {{"block_count", 1U}}, 0.01, 0.001);
    const auto cartesian = read_json_file(directory / "curve-price.json");
    validate_price_dataset(cartesian);
    auto malformed = cartesian;
    malformed["results"][0]["model_parameters"] = "not an object";
    bool bad_inputs_rejected = false;
    try { validate_price_dataset(malformed); }
    catch (const std::invalid_argument&) { bad_inputs_rejected = true; }
    if (!bad_inputs_rejected) throw std::runtime_error("Invalid inline inputs accepted");
    for (std::size_t i = 0; i < prices.size(); ++i) {
        const auto& row = cartesian.at("results").at(i);
        if (row.at("id") != i || row.at("price") != prices[i]
            || row.at("model_parameters") != model.at("models").at(i / 4).at("parameters")
            || row.at("curve_parameters") != curve.at("curves").at(i / 2 % 2).at("parameters")
            || row.at("product_parameters") != product.at("products").at(i % 2).at("parameters")) {
            throw std::runtime_error("Cartesian inline parameters do not match the priced row");
        }
    }
    if (cartesian.at("quote_convention") != product.at("quote_convention")) {
        throw std::runtime_error("Quoted parameters lost their convention");
    }
    std::ifstream formatted(directory / "curve-price.json");
    std::string line;
    std::size_t inline_products = 0;
    while (std::getline(formatted, line)) {
        if (line.find("\"product_parameters\":") != std::string::npos) {
            ++inline_products;
            if (line.find('}') == std::string::npos) {
                throw std::runtime_error("Product parameters must fit on one JSON line");
            }
        }
    }
    if (inline_products != prices.size()) throw std::runtime_error("Missing inline rows");

    // A failure after writing a row must leave the previous complete file intact.
    bool write_failed = false;
    try {
        write_single_asset_json(directory / "curve-price.json", {{"row_count", 2U}}, 2U,
            [&](std::size_t i) -> nlohmann::ordered_json {
                if (i == 1U) throw std::runtime_error("injected row failure");
                return cartesian.at("results").at(0);
            });
    } catch (const std::runtime_error&) { write_failed = true; }
    if (!write_failed || read_json_file(directory / "curve-price.json") != cartesian
        || std::filesystem::exists(directory / "curve-price.json.tmp")) {
        throw std::runtime_error("Failed streaming publication replaced complete data");
    }
    write_single_asset_json(directory / "empty.json", {{"row_count", 0U}}, 0U,
        [](std::size_t) { return nlohmann::ordered_json::object(); });
    const auto canonical = ordered_dataset_header({
        {"quote_convention", {{"strike", "relative"}}}, {"curve_family", "Flat"},
        {"product_family", "European Options"}, {"option_side", "call"},
        {"model_dataset", {{"id", "model"}}}, {"time_convention", {{"days_per_year", 252U}}}
    });
    const auto header_text = canonical.dump();
    if (header_text.find("\"product_family\":\"European Options\",\"option_side\":\"call\"")
            == std::string::npos
        || header_text.find("model_dataset") > header_text.find("time_convention")
        || header_text.find("time_convention") > header_text.find("quote_convention")) {
        throw std::runtime_error("Dataset header fields are out of order");
    }
    if (!read_json_file(directory / "empty.json").at("results").empty()) {
        throw std::runtime_error("All-rejected selection must remain valid JSON");
    }
    std::filesystem::remove_all(directory);
}
