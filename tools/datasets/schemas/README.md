# Dataset metadata

`recipe.yaml` exists before a run. It names the generator, the output JSON and,
for pricing, the exact input dataset paths and numerical plan. Parameter
recipes keep their sampling bounds; product recipes keep the parameter units,
calendar and payoff. The generation receipt is the sibling `generation.yaml`,
so the recipe does not name it.

The JSON holds data and stable identifiers: dataset ID, families, optional
option side, row count, input dataset IDs, and rows. It contains no URL or
catalogue path. The recipe resolves input IDs to paths. Moving the catalogue
therefore does not change the JSON checksum.

Published `generation.yaml` version 4 records input and output paths with
SHA-256, plus known CPU, GPU and software versions. No generation receipt
publishes execution durations: shared hardware and implementation changes make
them unsuitable as dataset performance promises. Operational durations remain
in campaign logs; native generators emit a `generation_timing` JSON log event.
The receipt does not fingerprint the recipe, generator, executable or build. If a published JSON is edited coherently, update its output SHA-256;
if a published input JSON is normalized or replaced, update the dependent
input SHA-256 only after checking that the intended numerical input rows are
unchanged. See `tools/datasets/refresh_generation.py`.

The top-level presentation order is `schema_version`, `status`, `inputs`,
`output`, `hardware`, then `software`. Omit a field when its evidence
is unavailable or inapplicable; keep the relative order of fields that are
present. In particular, do not invent software versions for historical
receipts. Legacy receipts with timing remain readable for collection; refreshing
or publishing them drops that field.

Version 1–3 receipts remain readable. `campaign.schema.yaml` governs
separately staged campaign checkpoints.

The publication checks require coherent IDs, row counts, finite prices and
standard errors, and the sensitivity columns named by the recipe. They do not
require an exact set or order of presentation fields in the JSON. The tests
exercise these data checks without depending on hard-coded catalogue counts
or on datasets from a separate generation campaign.

## Pricing recipe presentation

Single-asset prices, price gradients and multi-asset prices share the formatter
in `tools/datasets/metadata_format.py`. Their fields appear in this order:

1. Identity: version, kind, dataset ID, optional option side, generator.
2. Data: inputs, output, row layout and output channels when specified.
3. Construction: row count (or candidate row count), mapping, shard, curve
   assignment and rejection rules.
4. Conventions and calculation: quote convention, numerical method, central
   evaluation details when applicable, time grid or exact time representation.
5. Sampling: paths per price, random number generator and seeds.
6. Sensitivities: bump rules and coordinates, reconstruction formulas,
   sensitivity strategy, exercise replay and error scope.
7. Execution settings when specified: team size, rows per launch, streams,
   threads per block, paths per team, path chunk size and checkpoint rows.

A fixed simulation grid uses the same `time_grid` fields in every engine:

```yaml
time_grid:
  steps_per_year: 504
  simulation_steps_per_day: 2
  delta_t: 1 / 504
```

The historical seven heterogeneous multi-asset datasets instead use `252`,
`1`, and `1 / 252`. Their barrier monitoring frequency is recorded separately
inside `time_grid.monitoring`; monitoring and simulation need not have the same
frequency. Exact transitions and closed-form methods must not acquire a
fictitious fixed simulation grid for presentation consistency. Keep their
`time_representation` where applicable. Omit inapplicable fields, preserve
unknown extension fields, and never reorder sensitivity coordinates or rows.

Generation receipts use the common top-level order above. Within hardware,
use device, CPU model, GPU name, UUID, compute capability and driver version;
within software, use CMake, C++ compiler, CUDA compiler, CUDA target
architectures, nlohmann-json and MathDx. Only known values are recorded.
