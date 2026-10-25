# Pricing and model-sample code generation

This directory generates thin model-product pricing bindings, model-only sampling bindings, catalogue recipes, and their CMake registration. Numerical algorithms, schedules and payoffs remain in [`src/`](../../../src/README.md).

## Generate and verify

```sh
python3 tools/codegen/pricing_bindings/generate.py --family all --output .
python3 tools/codegen/pricing_bindings/generate.py \
  --family all --output /tmp/ai_factory-pricing-bindings --compare-root .
```

The comparison checks generated files in the repository. Handwritten model, curve, product and correlation parameter generators are outside this codegen inventory. The generator preserves timestamps for unchanged files.

## Sources of truth

| Responsibility | File |
| --- | --- |
| Render artifacts and compare outputs | [`generate.py`](generate.py) |
| Declare models and sample generators | [`sample_manifest.py`](sample_manifest.py) |
| Declare equity products and bindings | [`manifest.py`](manifest.py) |
| Resolve model, curve, product, engine and recipe capabilities | [`capability_manifest.py`](capability_manifest.py) |
| Declare first-order sensitivities and selected strategies | [`price_gradients/manifest.py`](price_gradients/manifest.py) |
| Supply C++ and recipe bodies | [`templates/`](templates/) |
| Record generated-file hashes | [`cmake/generated/PricingCapabilityManifest.json`](../../../cmake/generated/PricingCapabilityManifest.json) |

The capability manifest rejects missing and ambiguous compositions. Generated C++ and recipe bodies live in templates; renderers supply bounded substitutions. Parameter and product quote conventions are resolved before a price or gradient calculation. A `recipe.yaml` describes the intended computation and bump coordinates; a `generation.yaml` receipt appears only after a real execution.

The public catalogue includes aligned and selected Cartesian recipes. Price-only and first-order price-gradient generators are selected for publication; second-order bindings may be present in source without a published dataset. Monte Carlo recipes use 2^20 paths per price. Fixed-income Bermudan sensitivities use `frozen_regression_policy`.
