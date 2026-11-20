"""Small structural checks for the public catalogue."""

import json
from pathlib import Path
import unittest

import yaml


ROOT = Path(__file__).resolve().parents[2]


class PublicCatalogInventoryTest(unittest.TestCase):
    def test_manifest_entries_point_to_generators_and_recipes(self):
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        rows = manifest["datasets"]
        self.assertEqual(len(rows), manifest["count"])
        self.assertEqual(len({row["source"] for row in rows}), len(rows))
        for row in rows:
            with self.subTest(source=row["source"]):
                leaf = ROOT / row["catalogPath"]
                generator = leaf / "generator.cpp"
                recipe_path = leaf / "recipe.yaml"
                self.assertTrue(generator.is_file())
                self.assertTrue(recipe_path.is_file())
                recipe = yaml.safe_load(recipe_path.read_text())
                self.assertEqual(recipe["generator"], generator.relative_to(ROOT).as_posix())
                self.assertEqual(recipe["output"]["path"], row["output"])

    def test_bermudan_gradients_declare_their_replay_variant(self):
        manifest = json.loads((ROOT / "catalog/manifest.json").read_text())
        for row in manifest["datasets"]:
            if (row["kind"] == "price_gradients"
                    and "/fixed_income/" in row["source"]
                    and "/bermudan_" in row["source"]):
                with self.subTest(source=row["source"]):
                    recipe = yaml.safe_load(
                        (ROOT / row["catalogPath"] / "recipe.yaml").read_text()
                    )
                    variants = {
                        "_frozen_policy": "frozen_regression_policy",
                        "_frozen_exercise": "frozen_exercise_time",
                    }
                    suffix = next((s for s in variants if recipe["dataset_id"].endswith(s)), None)
                    self.assertIsNotNone(suffix)
                    strategy = variants[suffix]
                    self.assertEqual(recipe["exercise_replay"], strategy)
                    generator = ROOT / row["catalogPath"] / "generator.cpp"
                    self.assertIn("ExerciseReplayStrategy::" + strategy, generator.read_text())

    def test_pricing_quotes_match_their_product_recipe(self):
        from tools.codegen.pricing_bindings.product_quote import quote_convention

        for path in (ROOT / "catalog/single_asset").rglob("recipe.yaml"):
            recipe = yaml.safe_load(path.read_text())
            if recipe.get("kind") not in {"prices", "price_gradients"}:
                continue
            with self.subTest(recipe=str(path.relative_to(ROOT))):
                self.assertEqual(recipe["quote_convention"],
                                 quote_convention(recipe["inputs"]["product"]))
                self.assertNotIn("source_price_recipe", recipe.get("sensitivity", {}))

    def test_fixed_income_prices_request_no_bumps(self):
        for generator in (ROOT / "catalog/single_asset/fixed_income").rglob("generator.cpp"):
            if "prices" not in generator.parts:
                continue
            with self.subTest(generator=str(generator.relative_to(ROOT))):
                source = generator.read_text()
                self.assertNotIn("pg::BumpScale::", source)
                recipe = yaml.safe_load(generator.with_name("recipe.yaml").read_text())
                self.assertNotIn("sensitivity", recipe)
                self.assertNotIn("internal_sensitivity_plan", recipe)
                self.assertNotIn("exercise_replay", recipe)


if __name__ == "__main__":
    unittest.main()
