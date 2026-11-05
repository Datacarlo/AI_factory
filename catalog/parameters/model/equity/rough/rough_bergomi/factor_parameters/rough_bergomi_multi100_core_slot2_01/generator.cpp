#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/rough/rough_bergomi/parameters/rough_bergomi_01.json", "rough_bergomi_multi100_core_slot2_01", "datasets/parameters/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_multi100_core_slot2_01.json", "catalog/parameters/model/equity/rough/rough_bergomi/factor_parameters/rough_bergomi_multi100_core_slot2_01/generation.yaml", "",
        900U, 100U, 202611040103ULL, 1.0f, 200U);
}
