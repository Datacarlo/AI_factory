#include "tools/datasets/parameter_dataset.hpp"
int main() {
    ai_factory::workbench::datasets::write_core_factorized_model_dataset(
        "datasets/parameters/model/equity/markovian/schobel_zhu/parameters/schobel_zhu_01.json", "schobel_zhu_multi100_core_slot4_01", "datasets/parameters/model/equity/markovian/schobel_zhu/factor_parameters/schobel_zhu_multi100_core_slot4_01.json",
        "catalog/parameters/model/equity/markovian/schobel_zhu/factor_parameters/schobel_zhu_multi100_core_slot4_01/generation.yaml", "", 900U, 100U, 202611270101ULL,
        1.0f, 400U);
}
