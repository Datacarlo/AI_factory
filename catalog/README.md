# Dataset catalogue

`catalog/` contains native `generator.cpp` programs and their `recipe.yaml` specifications. A `generation.yaml` is present only for a completed generation. Generated JSON datasets are distributed separately and are not committed to this repository.

[`manifest.json`](manifest.json) lists the selected catalogue inventory, including each output path, kind, row count and recipe readiness. It is the source of truth for recipe counts: `ready` does not mean the JSON exists. Check for the JSON and its `generation.yaml` receipt before treating a dataset as published. The selected set includes generated price and price-gradient recipes alongside handwritten parameter generators. Additional source-only generators can be present without a manifest entry.

A price-gradient recipe records its sensitivity coordinates and bump rules. Fixed-income Bermudan swaptions use `frozen_regression_policy`; production Monte Carlo recipes request 2^20 trajectories per price. The recipe defines the planned calculation, while the receipt records its actual hardware, software and input/output hashes.

To inspect a selection without executing it:

```sh
python3 tools/datasets/generate_catalog.py --build build --kind all --catalog-only
```

Use `--compile --execute --run-dir /path/to/new/run` for an isolated campaign. Publication is a separate `--publish` action and refuses to overwrite different existing data. For one generator, see [`tools/run_generator.py`](../tools/run_generator.py).
