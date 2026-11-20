# Fast pricing

A single training engine handles price supervision and Sobolev supervision. Each
network has a short `training.py` and an explicit `recipe.yaml` in [`catalog/fast_pricers/`](../../catalog/fast_pricers/README.md).
Successful training adds `generation.yaml`, with input/output checksums and the
software and hardware used. Weights and metrics are written under
`fast_pricers/unpublished/` by default. Caches and resumable runs remain under
`work/learning/fast_pricing/`.

## Train one network

Run a catalogue leaf from the repository containing its datasets:

```sh
python catalog/fast_pricers/<dataset>/<architecture>/<objective>/training.py
```

Alternatively:

```sh
python -m learning.fast_pricing.train --recipe path/to/recipe.yaml
```

`--output-root /path/to/campaign` retains the isolated campaign layout: exports
under `/path/to/campaign/fast_pricers/` and runs under `/path/to/campaign/runs/`.
Recipe output paths remain logical `fast_pricers/...` paths, independent of publication status.

`--resume` restores weights, optimizer, scheduler, best validation checkpoint and
batch shuffle state. It requires the same configuration and prepared data.
Checkpoints are saved every ten epochs; an interruption may replay the unsaved
epochs. Numerical reproducibility requires the same software and hardware.

The generic configuration interface is retained for experiments. Catalogue
recipes select the equity production protocol described below.

## Shared components

- `data.py`: read JSON once, verify inline parameters against parent datasets,
  encode contractual inputs and create a checksummed numerical cache.
- `supervision.py`: select gradient coordinates using train-only Rj and compute
  their RMS normalization.
- `engine.py`: the common training loop, learning-rate schedule and checkpointing.
- `batches.py`: keep one dataset in GPU memory across its training variants.
- `recipes.py`: translate a recipe into the common engine configuration and write
  the completed model, metrics and generation receipt.
- `run_catalog.py`: run datasets sequentially, one worker per dataset; release GPU
  allocations between datasets. A worker shares tensors across its 3 or 12 runs.
- `catalogue.py`: generate leaves for additional equity datasets.
- `predict.py`: load the best checkpoint with its normalization and feature order.

Networks, losses, calendar handling and numerical transforms reuse `learning/common`.
Pricing kernels and source datasets are never modified by this pipeline.

## Equity protocol

| Size | Hidden layers | Activation |
| --- | --- | --- |
| Small | 3 × 64 | SiLU |
| Medium | 4 × 128 | SiLU |
| Large | 4 × 512 | SiLU |

Each price dataset has three price-only runs. Each price-gradient dataset has
those three runs and nine Sobolev runs: three sizes × lambda in {0.1, 1, 10}.
The only gradient filter is **Rj ≤ 0.04**, including Variance Gamma. There is no
cell mask, sign mask or concentration weighting.

The training budget is at most 100,000 rows. Larger datasets use a reproducible
sample stratified by model ID and product core/stress, without replacement.
Model groups are separated into train/validation/test; the same partition,
normalizations, seed and initialization are used for each paired price/Sobolev
comparison. Feature and price standardizations, Rj and gradient RMS values are
fitted on training rows only. Calendar intervals without gradients remain inputs.

For each retained coordinate j:

```
Rj = sum_train(standard_error_ij²) / sum_train(gradient_ij²)
g_norm_ij = gradient_ij * std_train(input_j) / std_train(price)
a_j = max(RMS_train(g_norm_j), 0.1 * median_positive_RMS, 1e-6)
loss = MSE(price_normalized) + lambda/J * sum_j kept_j * MSE(g_norm_j)/a_j²
```

J is the eligible coordinate count before Rj filtering. Closed-form derivatives
without Monte Carlo standard errors are retained. A zero target with zero error
is retained; a zero target with positive error is excluded.

Adam starts at 0.001, batch size 1024, maximum 500 epochs. Validation price RMSE
controls ReduceLROnPlateau (factor 0.5, patience 10, minimum 1e-6) and early stopping
(patience 40 after at least 50 epochs). The best validation checkpoint is exported.
Test metrics are computed only after training, with additional core/stress price
metrics. Test results never control training or checkpoint selection.

## Input conventions

Each recipe lists its ordered inputs and fixed parameters. Times are in years
(days/252). Constant model inputs are removed; varying SABR, rough SABR and CEV
spots are retained. Numerical American exercise intervals are not network inputs.
Contractual calendar variables are retained.

When spot is an input, quoted contract levels are converted to absolute levels
and their gradient labels are divided by the reference spot. Autocalls additionally
retain the contract reference spot separately from the variable model spot. No
output homogeneity transform is applied: the model predicts the dataset price.
The network is learned on the dataset domain; no claim of extrapolation outside
that domain is made.

`FastPricer` accepts tensors in its `feature_names` order and in these financial
coordinates. It applies statistical normalization internally. Checkpoints also
contain the fixed input values and the Rj supervision decisions.

## Campaigns

`run_catalog` consumes a manifest and validated cache delivery markers. Each
cache records the source dataset and parent SHA256, its coordinate policy, row
selection and array SHA256. Workers verify these before training. These caches
allow remote training without another copy of all source JSON files; an ordinary
catalogue leaf can rebuild the same cache from those JSON files.

Training progress, elapsed times and resumable state live under `work/learning/`.
Public `generation.yaml` files contain input/output paths and SHA256, hardware
and software; they contain no hostnames, GPU indices or timers.
