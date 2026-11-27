#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/merton/parameters/merton_01.json", "merton_multi100_core_slot0_01", "datasets/parameters/model/equity/markovian/merton/factor_parameters/merton_multi100_core_slot0_01.json",
        "catalog/parameters/model/equity/markovian/merton/factor_parameters/merton_multi100_core_slot0_01/generation.yaml", "", 900U, 100U, 202611270102ULL,
        1.0f, 0U);
}
