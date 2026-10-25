# AI Factory

AI Factory is a C++20/CUDA library for model simulation, financial-product pricing, sensitivity calculation, and dataset generation. It provides shared engines for Markovian and rough equity models, fixed-income models, early exercise, and correlated multi-asset products.

This repository contains the code and the [catalogue](catalog/README.md) of generators and recipes. Large generated JSON datasets are distributed separately. A `generation.yaml` in the catalogue records a completed generation; a recipe alone does not imply that its dataset has been generated.

## Repository map

| Directory | Contents |
| --- | --- |
| [`src/`](src/README.md) | C++/CUDA models, products, curves, simulation and pricing engines |
| [`catalog/`](catalog/README.md) | Dataset generators, recipes, completed-generation receipts and inventory |
| [`tools/`](tools/README.md) | Dataset orchestration, sampling and code generation |
| [`tests/`](tests/README.md) | Host and CUDA contract tests |
| [`cmake/`](cmake/README.md) | Build and test registration |
| [`learning/`](learning/README.md) | Python readers, GANs and deep-pricing training |

The [capability manifest](tools/codegen/pricing_bindings/capability_manifest.py) defines supported model-product compositions. The [catalogue manifest](catalog/manifest.json) lists selected dataset recipes and their readiness. A `ready` entry does not prove that its JSON has been generated; check for the JSON and its `generation.yaml` receipt.

## Build

Requirements: CMake 3.20+, a C++20 compiler compatible with the chosen CUDA Toolkit, NVIDIA GPU for CUDA tests, nlohmann-json, Python 3, PyYAML and `jsonschema`. Volterra FFT targets also require a compatible mathDx/cuFFTDx installation.

Configure with your own compiler, GPU architecture and mathDx paths. For example:

```sh
cmake -S . -B build -G Ninja \
  -DCMAKE_CUDA_COMPILER=/path/to/cuda/bin/nvcc \
  -DCUDA_WORKBENCH_ARCHITECTURES=89 \
  -DAI_FACTORY_MATHDX_ROOT=/path/to/mathdx
cmake --build build --target ai_factory_host_tests
ctest --test-dir build -R '^dataset_catalog$' --output-on-failure
```

[`CMakePresets.json`](CMakePresets.json) also contains machine-specific examples. Adapt compiler, dependency and architecture paths to the target machine before using a preset.

## Generate a dataset

Every catalogue leaf has a `generator.cpp` and `recipe.yaml`. Build the selected generator target, then run it through the [campaign controller](tools/datasets/generate_catalog.py). For example, to inspect the available selections:

```sh
python3 tools/datasets/generate_catalog.py --help
python3 tools/datasets/generate_catalog.py --build build --kind all --catalog-only
```

Add `--compile --execute --run-dir /path/to/new/run` to build and execute a selected campaign. The controller stages inputs and results in that run directory. `--publish` explicitly publishes complete dataset and receipt pairs; it rejects conflicting existing files. The [generator launcher](tools/run_generator.py) offers a shorter interface for one target when one of its configured machine presets applies.

A completed dataset has a JSON output and a `generation.yaml` receipt. The recipe specifies the numerical method, input datasets and sensitivity coordinates; the receipt records the actual execution and input/output hashes. Use the [catalogue inventory test](tests/datasets/test_public_catalog_inventory.py) and the [metadata compatibility tool](tools/datasets/check_dataset_compatibility.py) to inspect a publication.

Generated bindings must match their templates:

```sh
python3 tools/codegen/pricing_bindings/generate.py \
  --family all --output /tmp/ai_factory-pricing-bindings --compare-root .
```

CUDA fast math is disabled by default. Philox streams and contractual time grids are part of the numerical contract; numerical comparisons across GPU architectures still require qualification.
