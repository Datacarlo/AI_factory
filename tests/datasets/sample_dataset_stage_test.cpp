// Verify model-sample dataset assembly and observable metadata serialization.
#include "tools/datasets/artifact_io.hpp"
#include "tools/datasets/sample_dataset.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
    using namespace ai_factory::workbench;
    using namespace ai_factory::workbench::datasets;
    const std::filesystem::path directory =
        "/tmp/ai_factory_sample_dataset_stage_test";
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);
    const std::string progress_path = (directory / "progress.json").string();
    const std::string journal_path = (directory / "progress.jsonl").string();
    setenv("AI_FACTORY_GENERATION_PROGRESS", progress_path.c_str(), 1);
    setenv("AI_FACTORY_GENERATION_PROGRESS_LOG", journal_path.c_str(), 1);

    const std::vector<nlohmann::ordered_json> parameters = {
        {{"sigma", 0.2f}},
        {{"sigma", 0.3f}},
    };
    const std::vector<std::uint32_t> maturity_days = {63U, 64U, 503U, 504U};
    const std::vector<float> spots = {1.01f, 0.99f, 1.20f, 0.80f};
    const ModelSampleRecipe recipe{
        "samples_01",
        "Test model",
        directory / "samples_01.json",
        directory / "generation.yaml",
        2U,
        2U,
        63U,
        504U,
        {101U, 102U, 103U},
        "test transition",
        {
            {"regime", "core"},
            {"bounds", {{"sigma", {0.1f, 0.4f}}}},
        },
        {{"spot", {{"description", "terminal spot"}}}},
        {{"transition", "exact"}},
    };
    write_model_sample_dataset(
        recipe,
        {
            2U,
            2U,
            true,
            0.01,
            0.001,
            {
                {"threads_per_block", 128U},
                {"block_count", 2U},
            },
        },
        [&](std::size_t index) { return parameters.at(index); },
        maturity_days,
        {{"spot", &spots}}
    );
    validate_model_sample_dataset_file(
        recipe.dataset_path,
        4U,
        63U,
        504U
    );
    const auto progress = read_json_file(progress_path);
    if (progress.at("state") != "complete"
        || progress.at("completed_samples") != 4U
        || progress.at("total_samples") != 4U
        || progress.at("phase") != "writing") {
        throw std::runtime_error("sample write progress did not finish");
    }
    std::ifstream journal(journal_path);
    std::size_t progress_records = 0U;
    for (std::string line; std::getline(journal, line);) {
        ++progress_records;
    }
    if (progress_records != 2U) {
        throw std::runtime_error("sample progress journal is incomplete");
    }
    const auto document = read_json_file(recipe.dataset_path);
    if (document.at("database_id") != "samples_01"
        || document.at("model_family") != "Test model"
        || document.at("row_count") != 4U
        || document.contains("parameters")
        || document.at("time_convention").at("days_per_year") != 252
        || document.at("results").at(0).at("parameters").at("sigma") != 0.2f
        || document.at("results").at(1).at("parameters").at("sigma") != 0.2f
        || document.at("results").at(2).at("parameters").at("sigma") != 0.3f
        || document.at("results").at(0).at("id") != 0U
        || document.at("results").at(2).at("id") != 2U) {
        throw std::runtime_error(
            "sample parameter packages are not parameter-major"
        );
    }
    // One compact observation per line, distinct IDs even within a parameter group.
    std::ifstream json_lines(recipe.dataset_path);
    std::size_t observations = 0U;
    for (std::string line; std::getline(json_lines, line);) {
        if (line.find("\"id\":") == std::string::npos) continue;
        if (line.back() == ',') line.pop_back();
        const auto row = nlohmann::ordered_json::parse(line);
        if (row.size() != 4U || row.at("id") != observations
            || row.at("maturity_days") != maturity_days.at(observations)
            || row.at("values").at("spot") != spots.at(observations)) {
            throw std::runtime_error("sample observation changed");
        }
        ++observations;
    }
    if (observations != 4U) throw std::runtime_error("not one sample per line");
    // The same writer covers one parameter draw per observation and multiple outputs.
    auto iid_recipe = recipe;
    iid_recipe.database_id = "samples_02";
    iid_recipe.dataset_path = directory / "samples_02.json";
    iid_recipe.catalog_path = directory / "iid_generation.yaml";
    iid_recipe.production_parameter_count = 4U;
    iid_recipe.production_paths_per_parameter = 1U;
    const std::vector<float> variances = {0.01f, 0.04f, 0.09f, 0.16f};
    std::size_t parameter_calls = 0U;
    write_model_sample_dataset(
        iid_recipe, {4U, 1U, true, 0.01, 0.001, {{"threads_per_block", 128U}}},
        [&](std::size_t index) {
            ++parameter_calls;
            return nlohmann::ordered_json{{"sigma", variances.at(index)}};
        }, maturity_days, {{"spot", &spots}, {"variance", &variances}}
    );
    validate_model_sample_dataset_file(iid_recipe.dataset_path, 4U, 63U, 504U);
    const auto iid = read_json_file(iid_recipe.dataset_path);
    if (parameter_calls != 4U) throw std::runtime_error("wrong parameter call count");
    for (std::size_t i = 0; i < 4U; ++i) {
        const auto& row = iid.at("results").at(i);
        if (row.at("parameters").at("sigma") != variances[i]
            || row.at("values").at("spot") != spots[i]
            || row.at("values").at("variance") != variances[i]) {
            throw std::runtime_error("iid sample association changed");
        }
    }
    auto corrupted = iid;
    corrupted["results"][1]["id"] = 0U;
    { std::ofstream file(iid_recipe.dataset_path); file << corrupted.dump(); }
    bool rejected = false;
    try { validate_model_sample_dataset_file(iid_recipe.dataset_path, 4U, 63U, 504U); }
    catch (const std::runtime_error&) { rejected = true; }
    if (!rejected) throw std::runtime_error("duplicate observation ID accepted");

    std::ifstream yaml(recipe.catalog_path);
    const std::string yaml_text{
        std::istreambuf_iterator<char>(yaml),
        std::istreambuf_iterator<char>()
    };
    if (yaml_text.find("schema_version: 1") == std::string::npos
        || yaml_text.find("status: \"complete\"") == std::string::npos
        || yaml_text.find("row_count: 4") == std::string::npos
        || yaml_text.find("paths_per_parameter: 2") == std::string::npos) {
        throw std::runtime_error("sample generation receipt assembly failed");
    }
    std::filesystem::remove_all(directory);
    unsetenv("AI_FACTORY_GENERATION_PROGRESS");
    unsetenv("AI_FACTORY_GENERATION_PROGRESS_LOG");
}
