#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/sabr/parameters/sabr_01.json", "sabr_multi100_core_slot2_01", "datasets/parameters/model/equity/markovian/sabr/factor_parameters/sabr_multi100_core_slot2_01.json", "catalog/parameters/model/equity/markovian/sabr/factor_parameters/sabr_multi100_core_slot2_01/generation.yaml", "",
        900U, 100U, 202611040102ULL, 1.0f, 200U);
}
