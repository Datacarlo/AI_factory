#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations("rough_bergomi", {"spot", "variance"}, "five_rough_bergomi_cir_01");
}
