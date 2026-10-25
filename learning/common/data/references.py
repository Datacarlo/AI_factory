"""Resolve and join model, curve and product rows referenced by price data."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Mapping

import yaml


_ROLE_ORDER = {"model": 0, "curve": 1, "product": 2}


@dataclass(frozen=True)
class EntityRow:
    values: tuple[float, ...]
    ordinal: int


@dataclass(frozen=True)
class EntityTable:
    role: str
    dataset_id: str
    path: Path
    feature_names: tuple[str, ...]
    rows: Mapping[str, EntityRow]
    conventions: Mapping[str, object]


def find_repository_root(path: Path) -> Path:
    for candidate in (path.resolve().parent, *path.resolve().parents):
        if (candidate / "catalog").is_dir() and (candidate / "datasets").is_dir():
            return candidate
    raise ValueError(f"Cannot locate repository root from {path}")



def load_dataset_recipe(dataset_path: Path) -> dict:
    root = find_repository_root(dataset_path)
    try:
        relative = dataset_path.resolve().relative_to(root / "datasets")
    except ValueError as error:
        raise ValueError("Cannot resolve dataset recipe outside datasets/") from error
    recipe_path = root / "catalog" / relative.with_suffix("") / "recipe.yaml"
    if not recipe_path.is_file():
        raise FileNotFoundError(f"Missing adjacent dataset recipe: {recipe_path}")
    value = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid dataset recipe: {recipe_path}")
    return value

def resolve_catalog_reference(dataset_path: Path, catalog: str) -> Path:
    catalog_path = Path(catalog)
    if catalog_path.is_absolute():
        candidate = catalog_path
    else:
        parts = catalog_path.parts
        if not parts or parts[0] != "catalog":
            raise ValueError(f"Unsupported dataset catalog reference: {catalog}")
        candidate = find_repository_root(dataset_path) / "datasets" / Path(*parts[1:])
    return candidate.with_suffix(".json")


def _flatten_scalars(value: object, prefix: str = "") -> dict[str, float]:
    if isinstance(value, bool):
        raise ValueError(f"Boolean feature {prefix!r} needs an explicit encoder")
    if isinstance(value, (int, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"Non-finite feature {prefix!r}")
        return {prefix: number}
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for name, child in value.items():
            child_prefix = f"{prefix}.{name}" if prefix else str(name)
            result.update(_flatten_scalars(child, child_prefix))
        return result
    raise ValueError(
        f"Feature {prefix!r} has type {type(value).__name__}; "
        "register a categorical or sequence encoder before using it"
    )


def _row_array(document: dict, role: str) -> list[dict]:
    expected = {"model": "models", "curve": "curves", "product": "products"}.get(role)
    if expected is not None and isinstance(document.get(expected), list):
        return document[expected]
    candidates = [
        value
        for value in document.values()
        if isinstance(value, list)
        and (not value or isinstance(value[0], dict))
        and (not value or "id" in value[0])
        and (not value or "parameters" in value[0])
    ]
    if len(candidates) != 1:
        raise ValueError(f"Cannot identify parameter rows for role {role!r}")
    return candidates[0]


def load_entity_tables(
    dataset_path: Path,
    header: dict,
    reference_paths: Mapping[str, str | Path] | None = None,
) -> tuple[EntityTable, ...]:
    references: list[tuple[str, dict]] = []
    for name, value in header.items():
        if name.endswith("_dataset") and isinstance(value, dict):
            references.append((name.removesuffix("_dataset"), value))
    references.sort(key=lambda item: (_ROLE_ORDER.get(item[0], 100), item[0]))
    if not references:
        raise ValueError("Price dataset does not reference any parameter dataset")

    # Compact publications retain input IDs; resolve paths through the adjacent recipe.
    recipe = None
    if any("catalog" not in reference for _, reference in references):
        recipe = load_dataset_recipe(dataset_path)
        if not isinstance(recipe.get("inputs"), dict):
            raise ValueError("Dataset recipe has no inputs mapping")

    tables: list[EntityTable] = []
    for role, reference in references:
        override = None if reference_paths is None else reference_paths.get(role)
        if override is not None:
            path = Path(override).resolve()
        elif "catalog" in reference:
            path = resolve_catalog_reference(dataset_path, str(reference["catalog"]))
        else:
            recipe_input = recipe["inputs"].get(role) if recipe is not None else None
            if not isinstance(recipe_input, str):
                raise ValueError(f"Recipe has no input path for role {role!r}")
            path = (find_repository_root(dataset_path) / recipe_input).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Missing {role} parameter dataset: {path}")
        document = json.loads(path.read_text(encoding="utf-8"))
        dataset_id = str(document.get("database_id", ""))
        expected_id = reference.get("id", reference.get("database_id", ""))
        if dataset_id != str(expected_id):
            raise ValueError(
                f"{role} dataset id mismatch: expected {expected_id!r}, "
                f"found {dataset_id!r}"
            )
        rows = _row_array(document, role)
        if not rows:
            raise ValueError(f"Empty {role} parameter dataset")
        first = _flatten_scalars(rows[0].get("parameters"), role)
        names = tuple(sorted(first))
        mapped: dict[str, EntityRow] = {}
        for ordinal, row in enumerate(rows):
            row_id = str(row.get("id", ""))
            if not row_id or row_id in mapped:
                raise ValueError(f"Missing or duplicate {role} row id {row_id!r}")
            flattened = _flatten_scalars(row.get("parameters"), role)
            if tuple(sorted(flattened)) != names:
                raise ValueError(f"Inconsistent features in {role} row {row_id}")
            mapped[row_id] = EntityRow(
                values=tuple(flattened[name] for name in names), ordinal=ordinal
            )
        feature_names = names
        if role == "product":
            time = document.get("time_convention")
            if isinstance(time, dict) and time.get("unit") == "business_day":
                feature_names = tuple(
                    name.removesuffix("maturity") + "maturity_years"
                    if name.endswith("maturity") else name
                    for name in names
                )
        tables.append(
            EntityTable(
                role=role,
                dataset_id=dataset_id,
                path=path,
                feature_names=feature_names,
                rows=mapped,
                conventions={
                    name: document[name]
                    for name in ("quote_convention", "time_convention")
                    if name in document
                },
            )
        )
    return tuple(tables)
