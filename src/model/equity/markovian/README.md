# Markovian equity models

This directory owns equity models whose defining state is Markovian. The
`markovian` component is part of the canonical source, catalogue, and dataset
paths; public C++ namespaces do not repeat it.

Numerical lifts of rough models remain under `../rough`, even when their CUDA
execution is Markovian. The mathematical family, not the current simulator,
determines placement.

## Find an implementation

Within `src/model/equity/markovian/<model>/`:

- `parameters.hpp` defines the compact model row;
- `dataset.hpp/.cpp` load model rows on the host;
- `dynamics.cuh` declares the model policy;
- `dynamics_impl.cuh` defines device transitions;
- `sample.cuh/.cu` compose model-only sampling when available;
- `product/` contains generated or owned model-product launch units.

Method-specific files name their scheme or responsibility explicitly. A model
README documents only its equation, parameter mapping, and numerical scheme;
it does not copy the product matrix or generic payoff definitions.
