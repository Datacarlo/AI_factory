#include "tools/datasets/parameter_dataset.hpp"

int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json",
        "quadratic_rough_heston_multi_core_01",
        "datasets/model/equity/rough/quadratic_rough_heston/factor_parameters/quadratic_rough_heston_multi_core_01.json",
        "catalog/model/equity/rough/quadratic_rough_heston/factor_parameters/quadratic_rough_heston_multi_core_01/generation.yaml",
        "https://datasets.ai-factory.example/v2/model/equity/rough/quadratic_rough_heston/factor_parameters/quadratic_rough_heston_multi_core_01.json",
        900U, 250U, 2026102701ULL);
}
