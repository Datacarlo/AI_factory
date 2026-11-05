#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/black_scholes/parameters/black_scholes_01.json", "black_scholes_multi100_core_slot1_01", "datasets/parameters/model/equity/markovian/black_scholes/factor_parameters/black_scholes_multi100_core_slot1_01.json", "catalog/parameters/model/equity/markovian/black_scholes/factor_parameters/black_scholes_multi100_core_slot1_01/generation.yaml", "",
        900U, 100U, 202611040100ULL, 1.0f, 100U);
}
