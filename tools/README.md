# Offline tools

`tools/` contains reusable dataset and code-generation utilities. Runtime code under [`src/`](../src/README.md) does not depend on this directory.

| Task | Entry point |
| --- | --- |
| Generate and check model-product bindings | [Pricing and sampling codegen](codegen/pricing_bindings/README.md) |
| Select, compile and execute a dataset campaign | [`datasets/generate_catalog.py`](datasets/generate_catalog.py) |
| Run one configured generator | [`run_generator.py`](run_generator.py) |
| Check a dataset and its receipt | [`datasets/check_dataset_compatibility.py`](datasets/check_dataset_compatibility.py) |
| Read metadata schemas | [Dataset schemas](datasets/schemas/README.md) |

`datasets/` owns row construction, JSON/YAML I/O, provenance and publication. `sampling/` owns parameter and sample generation; `pricing/` owns offline price-generation orchestration. `cuda/` holds reusable CUDA launch support. `codegen/` owns generated bindings, recipes and manifest drift checks. Tool tests belong under [`tests/`](../tests/README.md).

Catalogue `generator.cpp` files select inputs and launch settings. Shared serialization and numerical algorithms belong in `tools/` and `src/`, respectively.
