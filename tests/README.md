# Tests

`tests/` contains host and CUDA checks for the library and dataset tools. CMake registers executable tests in [`cmake/AIFactoryTests.cmake`](../cmake/AIFactoryTests.cmake); the configured CTest inventory is authoritative.

| Directory | Focus |
| --- | --- |
| `datasets/` | Loaders, catalogue structure, provenance and publication |
| `sampling/` | Model sample contracts and replay |
| `model/` | Model dynamics and analytics |
| `product/` | Payoff and observation policies |
| `price_gradients/`, `price_delta/` | Sensitivity plans and pricing parity |
| `longstaff_schwartz/` | Early exercise and regression |
| `volterra/`, `numerical/`, `cuda/` | Shared numerical and CUDA behavior |
| `simulation/` | Schedule and cross-asset simulation |
| `learning/` | Python dataset-reader and training contracts |

Keep fixtures with their owning test domain and name tests by subject.
The factorized-model loader integration test is skipped when its separately
distributed factor datasets are unavailable. To inspect and run a configured build:

```sh
ctest --test-dir build -N
ctest --test-dir build --output-on-failure
```
