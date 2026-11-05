#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/rough/quadratic_rough_heston/parameters/quadratic_rough_heston_01.json", "quadratic_rough_heston_multi100_core_slot1_01", "datasets/parameters/model/equity/rough/quadratic_rough_heston/factor_parameters/quadratic_rough_heston_multi100_core_slot1_01.json", "catalog/parameters/model/equity/rough/quadratic_rough_heston/factor_parameters/quadratic_rough_heston_multi100_core_slot1_01/generation.yaml", "",
        900U, 100U, 202611040104ULL, 1.0f, 100U);
}
