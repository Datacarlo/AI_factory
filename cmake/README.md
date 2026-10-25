# CMake target ownership

The root [`CMakeLists.txt`](../CMakeLists.txt) owns project-wide configuration and includes the domain modules below. Product target registration and source inventories belong to the narrowest applicable module.

| Module | Responsibility |
| --- | --- |
| `AIFactoryRuntime.cmake` | CUDA tuning, dataset validation and offline libraries |
| `AIFactoryTargets.cmake` | Runtime, launchers and construction helpers |
| `AIFactoryCatalog.cmake` | Catalogue generators and aggregate targets |
| `AIFactoryTests.cmake` | Host and CUDA tests |
| `generated/` | Code-generated bindings and capability metadata |

The configured build graph is authoritative. Inspect it after configuring a machine-specific build directory:

```sh
cmake --build build --target help
ctest --test-dir build -N
```

Generated CMake files are updated through the [pricing and sampling code generator](../tools/codegen/pricing_bindings/README.md). Sources used to infer link dependencies or test labels must be registered in `CMAKE_CONFIGURE_DEPENDS` so edits update the graph on the next build.
