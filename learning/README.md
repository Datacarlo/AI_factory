# Learning

`learning/` contains Python readers, GAN training and fast-pricing training. It consumes generated datasets without changing the pricing kernels or their recipes.

## Terminal samples

[`common/terminal_samples.py`](common/terminal_samples.py) streams the large sample JSON and preserves groups of paths sharing one parameter row when assigning training, validation and test splits. `maturity_days` is an explicit condition. Model-specific observable names and units come from the sample recipe.

`gan/` provides conditional terminal generators, critics, preprocessing, monitoring and evaluation. A configured run can be started and inspected with:

```sh
python3 -m learning.gan.train --config /path/to/gan-config.yaml
python3 -m learning.gan.watch /path/to/gan-run
python3 -m learning.gan.evaluate /path/to/gan-run /path/to/samples.json
```

Terminal samples support terminal observables. A payoff requiring the full path needs a path dataset.

## Fast pricing

[`fast_pricing/`](fast_pricing/README.md) provides the shared price/Sobolev training
engine and its catalogue of network recipes. Catalogue runs share data preparation,
train-only normalization, grouped partitions, scheduling and checkpointing. Equity
Sobolev recipes use RMS gradient balancing and Rj ≤ 0.04, without cell masks.

```sh
python3 -m learning.fast_pricing.train --recipe /path/to/recipe.yaml
```

Each leaf contains `training.py`, `recipe.yaml`, and, after successful training,
`generation.yaml`. Generated weights, metrics, caches and checkpoints are stored
under `work/learning/` by default. The datasets and pricing generators are unchanged.
