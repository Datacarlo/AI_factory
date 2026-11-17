"""Contractual time inputs for single-asset fast pricers (2026-11-16).

Dates in mother datasets stay in business days. Network features use years;
regular counts/accruals are removed only after checking their defining relation.
No calendar derivative is manufactured and no target price is rescaled.
"""
from __future__ import annotations
import math

DAY_FIELDS = frozenset({
    'maturity', 'observation_interval', 'reset_time', 'fixing_time',
    'payment_time', 'option_expiry', 'bond_maturity', 'exercise_time',
    'first_exercise_time', 'payment_interval',
})


def product_features(parameters: dict, time_convention: dict | None) -> dict:
    result = dict(parameters)
    if not time_convention or time_convention.get('unit') != 'business_day':
        return result
    year = float(time_convention['days_per_year'])
    if not math.isfinite(year) or year <= 0:
        raise ValueError('Invalid business-day year')
    if 'observation_interval' in result and 'maturity' in result:
        t, delta = result['maturity'], result['observation_interval']
        if delta <= 0 or t <= 0 or t % delta != 0:
            raise ValueError('T and Delta do not describe a regular issue calendar')
    if {'fixing_time', 'payment_time', 'accrual_period'} <= result.keys():
        if result['accrual_period'] != result['payment_time'] - result['fixing_time']:
            raise ValueError('Nonredundant caplet accrual requires a dedicated encoder')
        del result['accrual_period']
    if {'payment_interval', 'payment_count'} <= result.keys():
        delta, n = result['payment_interval'], result['payment_count']
        if delta <= 0 or n < 1 or int(n) != n:
            raise ValueError('Invalid regular swap calendar')
        if 'accrual_fraction' in result:
            # Mother parameters are stored in float32; accommodate that rounding.
            if not math.isclose(result['accrual_fraction'], delta / year, rel_tol=1e-7, abs_tol=1e-8):
                raise ValueError('Nonredundant swap accrual requires a dedicated encoder')
            del result['accrual_fraction']
        result['swap_duration_years'] = n * delta / year
        del result['payment_count']
        if 'exercise_count' in result:
            e = result.pop('exercise_count')
            if int(e) != e or e < 1 or e > n:
                raise ValueError('Invalid Bermudan exercise count')
            result['exercise_horizon_years'] = (e - 1) * delta / year
    # Numerical American exercise mesh is dataset metadata, not a contract input.
    result.pop('exercise_interval', None)
    for name in DAY_FIELDS & result.keys():
        result[name + '_years'] = result.pop(name) / year
    return result


def rejects_legacy_maturity(feature_names: tuple[str, ...]) -> bool:
    names = set(feature_names)
    return bool(names & {
        'product.observation_interval_years', 'product.fixing_time_years',
        'product.swap_duration_years',
    })
