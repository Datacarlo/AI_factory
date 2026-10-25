"""Regression coverage for compact price-gradient publications."""

import json
from pathlib import Path

import numpy as np

from learning.common.data.cache import _infer_gradients
from learning.common.data.conventions import (
    business_days_to_years,
    ratio_strike_gradient_to_absolute,
    ratio_strike_to_absolute,
)


def test_compact_gradient_mapping_preserves_published_order():
    names = ("model.spot", "product.strike", "product.maturity_years")
    row = {
        "price": 3.0,
        "gradients": {
            "model.spot": 0.4,
            "product.strike": -0.2,
            "product.maturity_years": 1.7,
        },
        "standard_error": 0.01,
        "gradient_standard_errors": {
            "model.spot": 0.03,
            "product.strike": 0.04,
            "product.maturity_years": 0.05,
        },
    }
    targets = _infer_gradients({}, row, names)
    assert [target.wrt_name for target in targets] == list(row["gradients"])
    assert [target.wrt_index for target in targets] == [0, 1, 2]
    assert [target.source_name for target in targets] == [
        "gradients:model.spot",
        "gradients:product.strike",
        "gradients:product.maturity_years",
    ]


def test_strike_chain_rule_and_year_fraction_numerically():
    q, s0 = 1.13, 2.4
    k = ratio_strike_to_absolute(q, s0)
    assert np.isclose(k, q * s0)

    # C(K)=K^2: dC/dq = dC/dK * dK/dq = 2K*S0.
    derivative_by_ratio = 2.0 * k * s0
    assert np.isclose(
        ratio_strike_gradient_to_absolute(derivative_by_ratio, s0), 2.0 * k
    )

    # Product maturity is converted before learning; its derivative remains per year.
    days, basis = 63.0, 252.0
    years = business_days_to_years(days, basis)
    assert np.isclose(years, 0.25)
    annual_derivative = 2.0 * years  # d(T^2)/dT
    daily_derivative = annual_derivative / basis
    assert np.isclose(annual_derivative, daily_derivative * basis)


def test_compact_publication_joins_recipe_inputs_and_converts_coordinates(tmp_path):
    from learning.common.data.cache import prepare_pricing_dataset

    root = tmp_path
    dataset = root / "datasets/model/example/prices/example.json"
    recipe = root / "catalog/model/example/prices/example/recipe.yaml"
    model_path = root / "datasets/model/example/models.json"
    product_path = root / "datasets/product/example/products.json"
    for path in (dataset, recipe, model_path, product_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps({"database_id": "models", "models": [
        {"id": "m1", "parameters": {"spot": 2.0}}
    ]}))
    product_path.write_text(json.dumps({"database_id": "products", "time_convention": {
        "unit": "business_day", "days_per_year": 252
    }, "quote_convention": {"strike": {"mode": "ratio_to_central_spot"}}, "products": [
        {"id": "p1", "parameters": {"strike": 1.25, "maturity": 63}}
    ]}))
    recipe.write_text("""inputs:\n  model: datasets/model/example/models.json\n  product: datasets/product/example/products.json\nsensitivity:\n  parameters:\n  - {parameter: model.spot}\n  - {parameter: product.strike}\n  - {parameter: product.maturity_years}\n""")
    row = {"model_id": "m1", "product_id": "p1", "price": 1.0,
           "gradients": {"model.spot": 0.5, "product.strike": -0.8,
                         "product.maturity_years": 0.2},
           "standard_error": 0.01,
           "gradient_standard_errors": {"model.spot": 0.03,
               "product.strike": 0.04, "product.maturity_years": 0.05}}
    dataset.write_text(json.dumps({"database_id": "prices", "row_count": 1,
        "model_dataset": {"id": "models"}, "product_dataset": {"id": "products"},
        "results": [row]}))
    prepared = prepare_pricing_dataset(dataset, cache_root=root / "cache")
    assert prepared.schema.row_count == 1
    assert prepared.schema.feature_names == (
        "model.spot", "product.maturity_years", "product.strike"
    )
    np.testing.assert_allclose(prepared.features[0], [2.0, 0.25, 2.5])
    np.testing.assert_allclose(prepared.gradients[0], [0.5, -0.4, 0.2])
    np.testing.assert_allclose(prepared.gradient_standard_errors[0], [0.03, 0.02, 0.05])
    assert [g.wrt_name for g in prepared.schema.gradients] == [
        "model.spot", "product.strike", "product.maturity_years"
    ]


def test_group_selection_keeps_complete_surfaces_and_core_stress():
    from learning.common.data.selection import make_dataset_selection

    group_ids = np.repeat(np.arange(20), 4)
    selected = make_dataset_selection(
        80, train_size=40, train_group_count=10, split_seed=10,
        selection_seed=20, group_ids=group_ids, validation_group_count=4,
        test_group_count=4, core_group_count=18,
    )
    train_groups = np.unique(group_ids[selected.train_indices])
    assert train_groups.size == 10
    assert selected.train_indices.size == 40
    assert set(np.unique(group_ids[selected.train_indices])).isdisjoint(
        np.unique(group_ids[selected.validation_indices])
    )
    assert set(np.unique(group_ids[selected.train_indices])).isdisjoint(
        np.unique(group_ids[selected.test_indices])
    )
    assert sum(train_groups < 18) == 8
    assert sum(train_groups >= 18) == 2
    for group in train_groups:
        assert np.count_nonzero(group_ids[selected.train_indices] == group) == 4
