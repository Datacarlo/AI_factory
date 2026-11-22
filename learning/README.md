# Learning

`learning/` contains Python readers, GAN training and fast-pricing training. It consumes generated datasets without changing the pricing kernels or their recipes.

## Terminal samples

[`common/terminal_samples.py`](common/terminal_samples.py) streams the large sample JSON and preserves groups of paths sharing one parameter row when assigning training, validation and test splits. `maturity_days` is an explicit condition. Model-specific observable names and units come from the sample recipe.

[`gan/`](gan/README.md) provides one conditional terminal-GAN pipeline. Network
leaves live in [`catalog/gan/`](../catalog/gan/README.md), with `training.py`,
`recipe.yaml` and a `generation.yaml` after successful export. Weights, metrics and
fixed-condition reference samples live in the local `GANS/unpublished/` collection.

```sh
python3 -m learning.gan.train --dataset datasets/samples/equity/markovian/heston/samples_02.json
python3 -m learning.gan.train --recipe catalog/gan/equity/markovian/heston/samples_02/recipe.yaml
```

The common 3M protocol uses a logistic GAN, three hidden layers of 256 neurons,
LR 1e-4, and validation-driven patiences 2/6. Heston, Bates, rough Heston and CIR
have completed the initial evaluation; the registry also supports the remaining
available curve-free sample families. Independent repeated samples
at fixed parameters and maturity are required for distributional validation.

Terminal samples support terminal observables. A payoff requiring the full path needs a path dataset.

## Fast pricing

[`fast_pricing/`](fast_pricing/README.md) provides the shared price/Sobolev training
engine. Network recipes live in [`catalog/fast_pricers/`](../catalog/fast_pricers/README.md). Catalogue runs share data preparation,
train-only normalization, grouped partitions, scheduling and checkpointing. Equity
Sobolev recipes use RMS gradient balancing and Rj ≤ 0.04, without cell masks.

```sh
python3 -m learning.fast_pricing.train --recipe /path/to/recipe.yaml
```

Each leaf contains `training.py`, `recipe.yaml`, and, after successful training,
`generation.yaml`. Completed weights and metrics are stored in `fast_pricers/unpublished/`; caches
and checkpoints remain under `work/learning/fast_pricing/`. The datasets and pricing generators are unchanged.
