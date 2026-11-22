# Conditional terminal GANs

One training engine consumes terminal samples. Network recipes live in
[`catalog/gan/`](../../catalog/gan/README.md). Completed models are exported under
`gan/unpublished/`; the sibling `gan/published/` is for externally distributed
models. These collections are local, like `datasets/` and `fast_pricers/`.

## One dataset, one pipeline

```sh
python -m learning.gan.train --dataset datasets/samples/equity/markovian/heston/samples_02.json
# Or use the fixed recipe:
python catalog/gan/equity/markovian/heston/samples_02/training.py
```

The dataset command creates a missing catalogue leaf and native reference samples,
then calls the same preparation, training and export routine as the leaf.
Existing recipes and supplied reference archives are reused. `--prepare-only`
checks and caches inputs without training; `--resume` restores the exact saved
training state. `--output-root PATH` puts exports under `PATH/gan/` and resumable
runs under `PATH/runs/`. Logical output paths are unchanged.

Training requires Python, NumPy, PyYAML and PyTorch with the appropriate CUDA
runtime. Creating new native reference samples additionally requires a compatible
CUDA compiler and the repository's `src/` headers. The small
[`reference_kernel.cu`](reference_kernel.cu) driver uses the existing model
simulation and preparation functions; CUDA fast math is not enabled. `--nvcc` and
`--architecture` select the compiler and CUDA SM. No compilation is needed when
the reference archive is supplied.

The model registry covers the available curve-free sample families. The four
initially evaluated GANs are Heston, Bates, rough Heston and CIR; adding a recipe
for another family does not establish its trained accuracy. New reference drivers
call the existing public sample launchers, including the FFT and prepared lift
pipelines. FFT references also require MathDx (set `AI_FACTORY_MATHDX_ROOT` if it
is not found in the local toolchain directories).

Fixed inputs are declared and checked, never silently discarded. Unit spot is
removed only where fixed in the source; CEV, SABR and rough SABR retain their
varying spot input. Equity outputs remain terminal spot, not the full state or a
trajectory. OU and Vasicek use their signed terminal state; G2 generates both
signed factors jointly. Signed outputs use the identity transform and training
standardization, without a logarithm or clipping negative rates.

Hull–White, CIR++ and G2++ have no sample or GAN offering. Rough
Stein–Stein uses the corrected sample model with independent initial volatility and long-run level. Terminal-only GANs cannot
price path-dependent payoffs from their terminal output alone.

## Common 3M protocol

| Setting | Value |
| --- | --- |
| Objective | Classical logistic GAN |
| Generator / discriminator | Three hidden layers of 256 neurons; embeddings of 150 |
| Activation | Softplus, beta 10 |
| Latent noise | 5 independent uniform [0,1] coordinates |
| Optimizer | Adam, betas (0.5, 0.9), initial LR 1e-4 for both networks |
| Batch | 2,000; 5 discriminator updates per generator update |
| EMA | 0.99 |
| Validation | Every epoch |
| LR schedule | Halve after 2 non-improving validations; minimum 1e-6 |
| Early stopping | 6 non-improving validations, after at least 9 epochs |
| Budget | At most 53 epochs |
| Significant improvement | Absolute score decrease greater than 0.001 |

All IID training rows are used. Grouped `samples_01` use the same engine and
hyperparameters, with row-wise shuffling of their packs of 250. When paired with
an existing IID GAN, they reuse its exact validation/test archive. Entire packs
whose model tuple occurs in that archive are excluded from training, even if
maturity differs. This is declared as `training_rows: exclude_reference_parameters`.
Source JSON files are unchanged. Counts and exclusions are recorded in the
prepared manifest and exported selection metadata. Typically 256 reserved packs
leave 2,936,000 training rows versus 3,000,000 IID rows; this small budget difference
must be reported when interpreting comparisons. Holdouts are never training data.

The qualified production datasets have 3 million
rows, giving 1,500 generator updates per epoch. Smaller/larger datasets can use the
same routine but the effective budget must be reviewed. Conditions are standardized
using training rows only. Validation/test model tuples are distinct from training
(and from each other), ignoring maturity; coincident float32 draws are excluded
before choosing conditions. Nonnegative outputs use log(value + 1e-12), train-only
standardization and a nonnegative inverse transform. The treatment near zero is
part of model-specific qualification.

Patience counts validation evaluations. A small improvement can set a new best
checkpoint without resetting significant-improvement patience. Both schedulers,
optimizers, RNG and shuffle state are checkpointed every epoch for resumption.
The exact lowest validation score selects the best EMA weights. Test results never
control scheduling, stopping or selection. Hardware/software differences can
change numerical results.

## Independent validation and test

An IID sample is insufficient to estimate a full distribution at fixed parameters.
Each reference archive therefore contains **128 validation configurations and 128
test configurations**, with fixed parameters **and maturity**. Each configuration
has two independent sets of 8,192 native samples. The first set is compared with
the GAN; the second measures simulator-versus-simulator variation.

All model parameter tuples are disjoint between train, validation and test, even
ignoring maturity. The preparation checks this across the entire training set.
Validation references are reused across candidate architectures and recipes. The
score is:

```text
mean_over_conditions(Wasserstein1 / reference_std
                     + 0.1 * abs(log(generated_std / reference_std)))
```

The score is not a relative pricing error. The noise reference is not subtracted
from it. Calls, quantiles, moments and collapse diagnostics accompany it.
The best checkpoint is evaluated once on test after training. The four initial
published GANs use the same test previously inspected in the architecture study;
these are not claims of a never-before-inspected benchmark.

`reference.npz` is required to reproduce the selected training conditions.
For the initial four GANs, those exact conditions were retained from exploration;
the archive is authoritative (`selection: stored_in_archive`). If it is missing,
the routine does not silently choose new conditions. New datasets can create new
references from the sibling grouped sample dataset using a recorded seed, then
reuse the archive. Reference sampling conventions are explicit in each recipe.

## Use an exported GAN

```sh
python -m learning.gan.sample gan/unpublished/equity/markovian/heston/samples_02/model.pt \
  conditions.json --samples-per-condition 8192 --output samples.json
```

`conditions.json` is a list of dictionaries with exactly the coordinate names in
the recipe (`T` is in years, days/252). Normalization and inverse transforms are
applied internally; no source dataset or reference archive is needed for inference.
The previous run-directory interface to `learning.gan.sample` remains available.

## Shared implementation

- `catalogue.py`, `recipes.py`: compact leaves, common configuration and exports.
- `models.py`, `references.py`: coordinate declarations and native references.
- `data.py`: mmap preparation, disjoint groups and source identity checks.
- `engine.py`: the existing training, scheduling and checkpoint loop.
- `networks.py`, `metrics.py`, `sample.py`: networks, diagnostics and inference.
- `learning/common`: dataset readers, normalization and collection/receipt utilities.

Explicit experimental configurations remain supported through
`python -m learning.gan.train --config CONFIG.yaml`. They do not replace the common
catalogue protocol.
