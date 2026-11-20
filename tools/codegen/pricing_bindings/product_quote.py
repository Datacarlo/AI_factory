"""Read quote conventions from the product recipe used by a pricing dataset."""

from functools import lru_cache
from pathlib import Path
import re

import yaml


@lru_cache(maxsize=None)
def quote_convention(dataset_path: str) -> dict:
    root = Path(__file__).resolve().parents[3]
    product = Path(dataset_path).relative_to("datasets/parameters/product")
    # Shards retain the parent product's quote convention.
    dataset_id = re.sub(r"_shard[0-9]+$", "", product.stem)
    recipe = (root / "catalog/parameters/product" / product.parent
              / dataset_id / "recipe.yaml")
    return yaml.safe_load(recipe.read_text())["quote_convention"]
