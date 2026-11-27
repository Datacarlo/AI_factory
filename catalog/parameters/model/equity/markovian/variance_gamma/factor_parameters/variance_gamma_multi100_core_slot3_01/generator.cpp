#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/variance_gamma/parameters/variance_gamma_01.json", "variance_gamma_multi100_core_slot3_01", "datasets/parameters/model/equity/markovian/variance_gamma/factor_parameters/variance_gamma_multi100_core_slot3_01.json",
        "catalog/parameters/model/equity/markovian/variance_gamma/factor_parameters/variance_gamma_multi100_core_slot3_01/generation.yaml", "", 900U, 100U, 202611270103ULL,
        1.0f, 300U);
}
