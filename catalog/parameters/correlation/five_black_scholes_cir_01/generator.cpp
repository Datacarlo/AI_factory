#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations("black_scholes", {"spot"}, "five_black_scholes_cir_01");
}
