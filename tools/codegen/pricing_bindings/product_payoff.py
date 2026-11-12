"""Contract edge cases shared by price and sensitivity recipes."""
from functools import lru_cache
from pathlib import Path
import yaml

@lru_cache(maxsize=None)
def payoff_metadata(product: str) -> dict:
    if product != "cliquet":
        return {}
    recipe = Path(__file__).resolve().parents[3] / "catalog/parameters/product/cliquet/cliquets_01/recipe.yaml"
    return {"payoff": {"periodic_return": yaml.safe_load(recipe.read_text())["payoff"]["periodic_return"]}}
