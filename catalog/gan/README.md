# GAN catalogue

Each leaf contains a short `training.py`, a `recipe.yaml`, and a
`generation.yaml` only when its trained model has been exported. All leaves call
[`learning/gan`](../../learning/gan/README.md); there is no per-model training loop.
[`manifest.json`](manifest.json) lists the network recipes, independently of the
dataset and fast-pricer inventories.

The public offering has one trained model per financial model, using all three
million IID sample rows and the same training procedure. Leaves follow
`equity/{markovian,rough}/<model>/samples_02/` and
`fixed_income/<model>/samples_02/`. The existing dataset identifiers and RNG
domains are retained. The manifest lists these IID recipes only.
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
      reference.npz  # self-contained validation/test conditions and native simulations
```

`model.pt` includes the best EMA generator, coordinate order and fitted
normalizations. `metrics.json` reports validation, test and reference simulation
noise. Neither is automatically published to an external service. Recipes contain
no execution receipt; `generation.yaml` records input/output paths and SHA256 plus
software/hardware, without hostnames, GPU indices or timings.

Cached arrays, logs and resumable optimizer checkpoints remain under
`work/learning/gan/`. Deleting these does not remove the trained GAN collection or
its reference archive. Downloading only `model.pt` is sufficient for inference.

Reference archives are self-contained and supplied with each trained model.
Training does not require any other sample dataset. The recipe declares
`selection: stored_in_archive`: if the archive is missing, download it alongside
the model before reproducing training. This preserves the original holdouts.
