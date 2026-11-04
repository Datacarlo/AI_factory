# Dataset catalogue

`catalog/` contains native `generator.cpp` programs and their `recipe.yaml` specifications. A `generation.yaml` is present only for a completed generation. Generated JSON datasets are distributed separately and are not committed to this repository.

[`manifest.json`](manifest.json) lists the selected catalogue inventory, including each output path, kind, row count and recipe readiness. It is the source of truth for recipe counts: `ready` does not mean the JSON exists. Check for the JSON and its `generation.yaml` receipt before treating a dataset as published. The selected set includes generated price and price-gradient recipes alongside handwritten parameter generators. Additional source-only generators can be present without a manifest entry.

A price-gradient recipe records its sensitivity coordinates and bump rules. Fixed-income Bermudan swaptions use `frozen_regression_policy`; production Monte Carlo recipes request 2^20 trajectories per price. The recipe defines the planned calculation, while the receipt records its actual hardware, software and input/output hashes.

To inspect a selection without executing it:

```sh
python3 tools/datasets/generate_catalog.py --build build --kind all --catalog-only
```

Use `--compile --execute --run-dir /path/to/new/run` for an isolated campaign. Publication is a separate `--publish` action and refuses to overwrite different existing data. For one generator, see [`tools/run_generator.py`](../tools/run_generator.py).

## Layout

The catalogue and downloadable collection share the same relative paths:

```text
catalog/
  parameters/{model,product,curve,correlation}/
  samples/{equity,fixed_income}/
  single_asset/{equity,fixed_income}/
  multi_asset/
```

Each dataset `datasets/<path>/<id>.json` has its generator and recipe under
`catalog/<path>/<id>/`. A completed dataset also has its generation receipt
there. Parameter datasets are linked from the pricing recipes that use them;
sample recipes declare their own parameter-generation bounds.

### Sample observations

Sample JSON files contain `database_id`, `model_family`, `row_count`,
`time_convention` (`days_per_year: 252`) and a `results` array. Each observation
occupies one compact line with `id` (zero-based observation index), `parameters`,
`maturity_days` and named terminal `values`. Parameters are repeated in each
observation; no parameter-ID lookup is needed. Recipes retain the parameter
bounds, seeds and parameter-major/path-major sampling shape.

### Single-asset observations

Price and price-gradient recipes use `row_layout: inline_parameters`. Each
result embeds the original `model_parameters`, `product_parameters`, and, when
applicable, `curve_parameters`. Source row IDs and dataset IDs are retained,
but reading a pricing observation no longer requires joining parameter files.
Quoted strikes/barriers stay in their original units; `quote_convention` and
`time_convention` describe their interpretation. Resolved absolute levels and
bumped inputs are not substituted for the original parameters.

Each result has a zero-based `id` within the published selection. Rejected
candidates do not occupy an observation ID; this does not change simulation
seeds or source-row selection. Every result field occupies one line, with
parameter objects, gradient objects and calendar arrays written compactly.
`price`, `standard_error`, `gradients` and `gradient_standard_errors` retain
their existing meanings. The writer streams observations to a temporary file
and replaces the destination only after the complete JSON has been written.

Multi-asset datasets retain their separate format.

### JSON header order

Descriptive fields come first. When present, `option_side` immediately follows
`product_family`. References to mother datasets follow the descriptive fields;
all conventions follow those references, before the observations. This rule
applies to single-asset, multi-asset, sample and parameter datasets wherever the
corresponding fields exist.
