# Multi-asset autocalls

Three price-only policies use the existing correlated equity/rate engine:

| Contract | Policy | Loader |
|---|---|---|
| Athena | `MultiAssetAthenaAutocallPathPolicy` | `load_multi_asset_athena_autocalls` |
| Phoenix | `MultiAssetPhoenixAutocallPathPolicy` | `load_multi_asset_phoenix_autocalls` |
| Phoenix with coupon memory | `MultiAssetPhoenixMemoryAutocallPathPolicy` | `load_multi_asset_phoenix_memory_autocalls` |

Include `pricing_policy.cuh` and `dataset.hpp`. The single-asset Athena and
Phoenix policies are independent and retain their existing arithmetic.

## Contract

Notional is one. Each asset is observed as `R_i(t) = S_i(t) / S_i(0)` using its
own initial spot, including when initial spots differ. The available indicators
are `worst_of = min(R_i)`, `best_of = max(R_i)` and `basket = mean(R_i)` with equal
weights. Four roles select indicators independently:

- `autocall_indicator` determines early redemption;
- `coupon_indicator` determines Phoenix coupon payment;
- `protection_indicator` determines terminal capital protection;
- `loss_indicator` determines redemption when protection fails.

Barriers are dimensionless levels of these indicators. A level `0.8` means
80%. They are **not** multiplied by one central spot or by the basket spot.
Dates are integer business days, with 252 days per year. Observations occur at
`observation_interval`, `2 * observation_interval`, ..., `maturity`. Maturity
must be divisible by the interval. There is no observation at inception.
Protection is tested only at maturity; it is not a continuous knock-in barrier.
All barrier comparisons use `>=`.

For interval length `d`, the coupon per observation is
`c = annual_coupon_rate * d / 252`.

- **Athena:** at fixing `k` (one-based), an early call pays `1 + k*c`. At
  maturity, crossing the call barrier pays `1 + n*c`; otherwise protection
  pays one, or failed protection pays the loss indicator. No intermediate
  coupon is paid independently of redemption.
- **Phoenix:** at every fixing, a coupon is paid only if its indicator reaches
  the coupon barrier. Early call redeems one plus that fixing's coupon. At
  maturity, capital is one if protected, otherwise the loss indicator; the
  final coupon is evaluated separately.
- **Phoenix Memory:** the same rule, but a successful coupon fixing pays all
  accumulated unpaid coupons and clears the coupon memory. Memory is not paid
  automatically on redemption or maturity if the coupon condition fails.

The loss indicator is paid directly, with no additional cap. In particular,
choosing different protection and loss indicators defines a different contract
from the usual all-worst-of note. Coupon and call conditions also remain
independent when they select different indicators.

Each cashflow is discounted at its own fixing with
`exp(-integral_0^t r(s) ds)`. There is no second terminal discount. The same
contract works with a constant rate (zero extra Brownian factors) and a
correlated stochastic short-rate policy.

## Product JSON

Example Phoenix Memory input (one row):

```json
{
  "database_id": "multi_phoenix_memory_example",
  "row_count": 1,
  "product_family": "Multi-asset Phoenix Memory Autocalls",
  "time_convention": {"unit": "business_day", "days_per_year": 252},
  "quote_convention": {
    "autocall_barrier": {"mode": "ratio_to_initial_spot", "unit": "dimensionless", "observation": "indicator(S_i(t)/S_i(0))"},
    "coupon_barrier": {"mode": "ratio_to_initial_spot", "unit": "dimensionless", "observation": "indicator(S_i(t)/S_i(0))"},
    "protection_barrier": {"mode": "ratio_to_initial_spot", "unit": "dimensionless", "observation": "indicator(S_i(t)/S_i(0))"}
  },
  "products": [
    {"id": "000001", "parameters": {
      "maturity": 756,
      "observation_interval": 63,
      "autocall_barrier": 1.0,
      "coupon_barrier": 0.8,
      "protection_barrier": 0.6,
      "annual_coupon_rate": 0.08,
      "autocall_indicator": "worst_of",
      "coupon_indicator": "worst_of",
      "protection_indicator": "worst_of",
      "loss_indicator": "worst_of"
    }}
  ]
}
```

Phoenix uses family `Multi-asset Phoenix Autocalls`. Athena uses
`Multi-asset Athena Autocalls` and omits `coupon_barrier` and `coupon_indicator`
from its parameters and conventions. There is no `option_side` for these notes.
The shared multi-asset price writer retains the mother IDs and conventions;
pass an empty `option_side` to `MultiAssetPriceDatasetRecipe`.

A recipe describes the same observation and quote conventions as the product
JSON, plus its model/rate/correlation paths, row mapping, seed, trajectory
count and simulation grid. For example, the grid may use 2 steps per business
day with `dt = 1/504`; it does not change the quarterly observation interval.
Hardware and software belong to `generation.yaml`.

## Engine integration

Use `offline::pricing::execute_cross_asset_prices<EquityPack, RoughPack,
RatePolicy, ProductPolicy, TeamSize>` from
`tools/pricing/cross_asset_price_generation.cuh`, then the common
`write_multi_asset_price_dataset` writer. `EquityPack` and `RoughPack` retain
their existing model dynamics and Philox identities. Selectors use the same
order as other equity/rate products: equity assets, rough assets, rate,
product, correlation, optional curve.

The reconstruction/payoff kernel observes the selected indicators only on
contractual dates. An early call stops that path's remaining reconstruction.
For rough assets, the existing volatility prepass still prepares the requested
path chunk before reconstruction; early redemption does not remove work
already done by that prepass. No volatility simulation kernel was added.

The engine supports the same row batching, path chunks, streams, team sizes
and paths per team as terminal products. Autocalls require
`dt * steps_per_day = 1/252`, checked in both the orchestrator and raw launcher.
This does not change the settings of existing terminal-product generators.

## Tests

- `multi_asset_autocall_cuda`: independent cashflow formulas; all indicator
  combinations; single-asset contract reduction; deterministic calendar and
  mapping checks; independent Vasicek-discount expectation; causal QRH plus
  hybrid rough Bergomi with multiple launch geometries.
- `multi_asset_autocall_dataset`: loading conventions, invalid schedules and
  selectors, output IDs, quote preservation, no spurious call/put field.
- `cross_asset_cuda` and `multi_asset_rate_price_dataset`: existing terminal
  products and output non-regression.
