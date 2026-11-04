// Verify model-parameter dataset assembly and metadata serialization.
#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/parameter_dataset.hpp"

#include <filesystem>
#include <fstream>
#include <stdexcept>

int main() {
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_parameter_dataset_stage_test";
    std::filesystem::remove_all(directory);
    const GeneratedRows generated{{{{"sigma", 0.2f}}}, {{"method", "test"}}};
    write_model_dataset(
        "test_model_01",
        "Test model",
        directory / "model.json",
        directory / "generation.yaml",
        "https://datasets.ai-factory.example/test/model.json",
        {{"sigma", "volatility"}},
        {{"equation", "dS = sigma S dW"}},
        generated
    );
    const auto document = read_json_file(directory / "model.json");
    if (document.at("row_count") != 1U
        || document.at("models").at(0).at("id") != "000001") {
        throw std::runtime_error("parameter artifact assembly failed");
    }
    std::ifstream yaml(directory / "generation.yaml");
    const std::string receipt{std::istreambuf_iterator<char>(yaml), {}};
    if (receipt.find("status: \"complete\"") == std::string::npos
        || receipt.find("row_count: 1") == std::string::npos) {
        throw std::runtime_error("parameter generation receipt assembly failed");
    }
    const GeneratedRows heston{{{{"spot", 1.0f}, {"risk_free_rate", 0.02f},
        {"dividend_yield", 0.01f}, {"initial_variance", 0.04f}, {"kappa", 1.5f},
        {"theta", 0.04f}, {"gamma", 0.3f}, {"rho", -0.7f}}}, {{"method", "test"}}};
    const auto source = directory / "heston.json";
    write_model_dataset("heston_test", "Heston", source,
        directory / "source/generation.yaml", "https://example.test/heston.json",
        {}, {}, heston);
    const auto factor = directory / "factors.json";
    write_core_factorized_model_dataset(source, "factors_test", factor,
        directory / "factor/generation.yaml", "https://example.test/factors.json",
        1U, 1U, 42U, std::nullopt);
    const auto factors = read_json_file(factor);
    if (factors.at("source_model_dataset") != "heston_test"
        || factors.at("models").at(0).at("parameters").contains("rho")
        || factors.at("models").at(0).at("parameters").contains("risk_free_rate")) {
        throw std::runtime_error("Factor JSON must reference the source ID, not its path");
    }
    std::ifstream factor_recipe(directory / "factor/recipe.yaml");
    const std::string factor_yaml{std::istreambuf_iterator<char>(factor_recipe), {}};
    if (factor_yaml.find(source.generic_string()) == std::string::npos) {
        throw std::runtime_error("Factor recipe must retain the actual input path");
    }
    write_spot_relative_equity_product_dataset(
        "test_call", "European Options", directory / "product.json",
        directory / "product/generation.yaml", "https://example.test/product.json",
        {{"strike", "quoted strike"}}, {{"expression", "max(S-K,0)"}},
        {{{{"strike", 0.9f}, {"maturity", 252U}}}, {{"method", "test"}}});
    const auto quotes = read_json_file(directory / "product.json").at("quote_convention");
    if (quotes.contains("reference_policy") || quotes.size() != 1U
        || quotes.at("strike").at("mode") != "ratio_to_central_spot"
        || quotes.at("strike").at("effective_level") != "strike * S0") {
        throw std::runtime_error("Product conventions must retain units without reference prose");
    }
    std::filesystem::remove_all(directory);
}
