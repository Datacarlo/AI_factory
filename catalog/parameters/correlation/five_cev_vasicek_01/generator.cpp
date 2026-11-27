#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations(
        "cev", {"spot"}, "five_cev_vasicek_01", "vasicek");
}
