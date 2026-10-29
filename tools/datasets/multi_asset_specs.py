"""Selected product, factor and rate datasets for multi-asset composition."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ReferenceDatasetSpec:
    cmake_target: str
    dataset_kind: str
    generator_path: str
    dataset_path: str

    @property
    def recipe_yaml_path(self) -> str:
        return str(Path(self.generator_path).with_name("recipe.yaml"))

    @property
    def generation_yaml_path(self) -> str:
        return str(Path(self.generator_path).with_name("generation.yaml"))


_FACTOR_MODELS = {
    "equity/markovian": ("bates", "heston", "sabr"),
    "equity/rough": ("quadratic_rough_heston", "rough_bergomi"),
    "fixed_income": ("cir",),
}

REFERENCE_DATASET_SPECS = (
    *(
        ReferenceDatasetSpec(
            f"generate_{model}_multi_core_01", "model_parameters",
            f"catalog/model/{market}/{model}/factor_parameters/"
            f"{model}_multi_core_01/generator.cpp",
            f"datasets/model/{market}/{model}/factor_parameters/"
            f"{model}_multi_core_01.json",
        )
        for market, models in _FACTOR_MODELS.items()
        for model in models
    ),
    *(
        ReferenceDatasetSpec(
            f"generate_{dataset_id}", "product_parameters",
            f"catalog/product/{family}/{dataset_id}/generator.cpp",
            f"datasets/product/{family}/{dataset_id}.json",
        )
        for family, dataset_id in (
            ("worst_of_option", "worst_of_options_multi_01"),
            ("best_of_option", "best_of_options_multi_01"),
            ("basket_option", "basket_options_multi_01"),
            ("basket_down_and_out_call", "basket_down_and_out_calls_multi_01"),
        )
    ),
    ReferenceDatasetSpec(
        "generate_five_equity_cir_correlations_01", "correlation_parameters",
        "catalog/correlation/five_equity_cir_01/generator.cpp",
        "datasets/correlation/five_equity_cir_01.json",
    ),
    ReferenceDatasetSpec(
        "generate_constant_rates_01", "model_parameters",
        "catalog/model/fixed_income/constant_rate/parameters/"
        "constant_rates_01/generator.cpp",
        "datasets/model/fixed_income/constant_rate/parameters/"
        "constant_rates_01.json",
    ),
)
