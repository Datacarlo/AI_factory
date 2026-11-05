#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/heston/parameters/heston_01.json", "heston_multi100_core_slot2_01", "datasets/parameters/model/equity/markovian/heston/factor_parameters/heston_multi100_core_slot2_01.json", "catalog/parameters/model/equity/markovian/heston/factor_parameters/heston_multi100_core_slot2_01/generation.yaml", "",
        900U, 100U, 202611040101ULL, 1.0f, 200U);
}
