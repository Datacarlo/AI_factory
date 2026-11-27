#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/bates/parameters/bates_01.json", "bates_multi100_core_slot0_01", "datasets/parameters/model/equity/markovian/bates/factor_parameters/bates_multi100_core_slot0_01.json",
        "catalog/parameters/model/equity/markovian/bates/factor_parameters/bates_multi100_core_slot0_01/generation.yaml", "", 900U, 100U, 202611270105ULL,
        1.0f, 0U);
}
