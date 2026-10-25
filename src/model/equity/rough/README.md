# Rough equity models

This directory owns equity models defined by rough or Volterra dynamics. A
model remains here when it uses a finite-factor Markovian approximation: the
mathematical family, not the current simulator, determines placement.

The `rough` component is part of the canonical source, catalogue, and dataset
paths. Public C++ namespaces do not repeat it.

## Find an implementation

Within `src/model/equity/rough/<model>/`:

- `parameters.hpp` defines the compact model row;
- `dataset.hpp/.cpp` load model rows on the host;
- `dynamics.cuh` and `dynamics_impl.cuh` own model path transformations;
- a strategy-qualified preparation or pricing file owns Volterra FFT or
  N-factor details;
- `sample.cuh/.cu` compose model-only sampling when available;
- `product/` contains generated model-product launch units.

Shared Volterra kernels and engines live under `src/volterra`; product
parameters, schedules, and payoffs live under `src/product`. Model READMEs
document only equations and model-specific numerical choices.
