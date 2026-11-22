# GAN catalogue

Each leaf contains a short `training.py`, a `recipe.yaml`, and a
`generation.yaml` only when its trained model has been exported. All leaves call
[`learning/gan`](../../learning/gan/README.md); there is no per-model training loop.
[`manifest.json`](manifest.json) lists the network recipes, independently of the
dataset and fast-pricer inventories.

Leaves follow `equity/{markovian,rough}/<model>/samples_01/` or `samples_02/`,
and the corresponding `fixed_income/<model>/` paths. `samples_01` train on
grouped observations; `samples_02` train on IID observations. The manifest lists all scheduled models.
Heston, Bates, rough Heston and CIR have completed the initial evaluation. Other
leaves specify the common training protocol; they receive `generation.yaml`
only after successful training and export.
```sh
python catalog/gan/equity/markovian/heston/samples_02/training.py
```

The source sample JSON and the supplied `reference.npz` are required for exact
training reproduction. Logical paths start with `datasets/` or `gan/`; the loader
checks `unpublished/`, then `published/`, then the direct collection path.
The logical `gan/` namespace resolves to the physical `GANS/` export folder;
public download URLs use `GANS/`.
The source JSON's parameter and maturity values are read without changing the
simulation data. The reference archive contains independent repeated samples at
fixed conditions, used for validation, test and a simulator-versus-simulator check.
It is an input to training, not a learned model.

Completed files are local and distributed separately from Git:

```text
GANS/
  published/
  unpublished/
    <same relative leaf>/
      model.pt
      metrics.json
      reference.npz  # may be shared with the paired leaf; see recipe.reference.path
```

`model.pt` includes the best EMA generator, coordinate order and fitted
normalizations. `metrics.json` reports validation, test and reference simulation
noise. Neither is automatically published to an external service. Recipes contain
no execution receipt; `generation.yaml` records input/output paths and SHA256 plus
software/hardware, without hostnames, GPU indices or timings.

Cached arrays, logs and resumable optimizer checkpoints remain under
`work/learning/gan/`. Deleting these does not remove the trained GAN collection or
its reference archive. Downloading only `model.pt` is sufficient for inference.

Grouped/IID comparisons share the exact reference archive when available. Grouped
training excludes complete parameter packs appearing in validation or test,
ignoring maturity. The recipe describes this exclusion and the exported metrics
record the effective training size. Architecture and optimizer are unchanged.
