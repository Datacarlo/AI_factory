# Fast-pricer catalogue

Each leaf contains `training.py` and `recipe.yaml`. The script calls the shared
[`learning/fast_pricing`](../../learning/fast_pricing/README.md) engine. Successful
training adds `generation.yaml` with input/output SHA256 and software/hardware.

[`manifest.json`](manifest.json) lists source datasets and their network recipes.
A listed recipe is a training specification, not evidence of a completed network.
Each price dataset has three architectures; each price-gradient dataset adds
three Sobolev weights per architecture. Training settings and supervision rules
are described in the shared engine README.

```sh
python catalog/fast_pricers/<dataset>/<architecture>/<objective>/training.py
```

Leaves follow the source dataset path, then architecture and objective. Logical
output paths start with `fast_pricers/`. By default, completed weights and metrics
are stored under the local `fast_pricers/unpublished/` collection. The sibling
`fast_pricers/published/` is reserved for models already distributed externally.
Neither collection is part of the public source catalogue. No publication is
performed automatically. The model checkpoint includes weights, feature order,
fixed inputs and normalization for inference without the training dataset.
Caches, logs and resumable training checkpoints remain in `work/learning/`.
An explicit `--output-root` retains the isolated campaign layout documented by
the training engine.

To add leaves for another dataset:

```sh
python -m learning.fast_pricing.catalogue datasets/single_asset/equity/.../dataset.json \
  --manifest /path/to/new-selection.json
```

The command writes leaves in `catalog/fast_pricers/` and a manifest for the supplied
selection. Use a separate selection manifest when extending an existing campaign.
