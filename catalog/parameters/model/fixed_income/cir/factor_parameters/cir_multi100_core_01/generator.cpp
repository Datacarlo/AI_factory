#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/fixed_income/cir/parameters/cir_01.json", "cir_multi100_core_01", "datasets/parameters/model/fixed_income/cir/factor_parameters/cir_multi100_core_01.json", "catalog/parameters/model/fixed_income/cir/factor_parameters/cir_multi100_core_01/generation.yaml", "",
        900U, 100U, 202611040105ULL, std::nullopt, 0U);
}
