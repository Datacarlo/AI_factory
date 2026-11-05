#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations("quadratic_rough_heston", {"spot"}, "five_quadratic_rough_heston_cir_01");
}
