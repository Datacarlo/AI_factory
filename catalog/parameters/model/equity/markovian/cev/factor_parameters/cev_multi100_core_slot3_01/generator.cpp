#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/cev/parameters/cev_01.json", "cev_multi100_core_slot3_01", "datasets/parameters/model/equity/markovian/cev/factor_parameters/cev_multi100_core_slot3_01.json",
        "catalog/parameters/model/equity/markovian/cev/factor_parameters/cev_multi100_core_slot3_01/generation.yaml", "", 900U, 100U, 202611270104ULL,
        1.0f, 300U);
}
