# Learning

`learning/` contains Python readers, GAN training and deep-pricing training. It consumes generated datasets without changing the pricing kernels or their recipes.

## Terminal samples

[`common/terminal_samples.py`](common/terminal_samples.py) streams the large sample JSON and preserves groups of paths sharing one parameter row when assigning training, validation and test splits. `maturity_days` is an explicit condition. Model-specific observable names and units come from the sample recipe.

`gan/` provides conditional terminal generators, critics, preprocessing, monitoring and evaluation. A configured run can be started and inspected with:

```sh
python3 -m learning.gan.train --config /path/to/gan-config.yaml
python3 -m learning.gan.watch /path/to/gan-run
python3 -m learning.gan.evaluate /path/to/gan-run /path/to/samples.json
```

Terminal samples support terminal observables. A payoff requiring the full path needs a path dataset.

## Deep pricing

`deep_pricing/` joins price rows to model, product and optional curve parameters. Its reader supports price-only and sensitivity datasets, content-addressed caches, fixed held-out splits, and uncertainty-weighted losses. The training engine supports supervised and Sobolev methods; the dataset decides which gradient coordinates are available.

```sh
python3 -m learning.deep_pricing.train \
  --dataset /path/to/price_gradients.json \
  --method sobolev --train-size 30000
python3 -m learning.deep_pricing.train --config /path/to/training-config.yaml
python3 -m learning.deep_pricing.compare /path/to/run-a /path/to/run-b
```

By default, training runs and prepared caches are written under the Git-ignored `work/learning/` directory. Explicit run and cache paths can still be supplied. The source code under `learning/` contains no bundled generated datasets.
