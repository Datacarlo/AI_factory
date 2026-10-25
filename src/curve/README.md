# Curve implementations

`src/curve/<curve>` owns a compact parameter row, its host loader and reusable
term-structure analytics. A curve does not own stochastic-model parameters,
product payoffs or a concrete model-product launcher.

Each curve keeps the same boundary:

- `parameters.hpp` defines the transferable row;
- `dataset.hpp/.cpp` load and validate rows on the host;
- `term_structure.cuh` declares the public device analytics;
- `term_structure_impl.cuh` contains their included device definitions;
- the local README documents only parameter meaning, equations and numerical
  treatment.
