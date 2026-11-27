#include "tools/datasets/homogeneous_correlation.hpp"
int main() {
    ai_factory::workbench::datasets::write_homogeneous_correlations(
        "schobel_zhu", {"spot", "volatility"}, "five_schobel_zhu_vasicek_01", "vasicek");
}
