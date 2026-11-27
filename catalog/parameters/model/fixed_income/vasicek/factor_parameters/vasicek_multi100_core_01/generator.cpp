#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/fixed_income/vasicek/parameters/vasicek_01.json", "vasicek_multi100_core_01", "datasets/parameters/model/fixed_income/vasicek/factor_parameters/vasicek_multi100_core_01.json",
        "catalog/parameters/model/fixed_income/vasicek/factor_parameters/vasicek_multi100_core_01/generation.yaml", "", 900U, 100U, 202611270106ULL,
        std::nullopt, 0U);
}
