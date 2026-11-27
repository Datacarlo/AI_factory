#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations(
        "merton", {"spot"}, "five_merton_vasicek_01", "vasicek");
}
