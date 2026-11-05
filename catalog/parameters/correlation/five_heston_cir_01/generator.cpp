#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations("heston", {"spot", "variance"}, "five_heston_cir_01");
}
