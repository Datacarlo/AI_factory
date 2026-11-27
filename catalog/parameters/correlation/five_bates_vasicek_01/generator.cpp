#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations(
        "bates", {"spot", "variance"}, "five_bates_vasicek_01", "vasicek");
}
