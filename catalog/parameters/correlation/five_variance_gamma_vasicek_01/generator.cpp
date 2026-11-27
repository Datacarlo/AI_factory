#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations(
        "variance_gamma", {"spot"}, "five_variance_gamma_vasicek_01", "vasicek");
}
