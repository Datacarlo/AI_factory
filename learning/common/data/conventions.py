"""Coordinate conversions declared by generic published data conventions."""

from __future__ import annotations


def ratio_strike_to_absolute(strike_ratio: float, central_spot: float) -> float:
    if central_spot <= 0.0:
        raise ValueError("central_spot must be positive")
    return strike_ratio * central_spot


def ratio_strike_gradient_to_absolute(
    derivative_by_ratio: float, central_spot: float
) -> float:
    """Apply dC/dK = (dC/dqK) / S0 for K=qK*S0."""
    if central_spot <= 0.0:
        raise ValueError("central_spot must be positive")
    return derivative_by_ratio / central_spot


def business_days_to_years(days: float, days_per_year: float) -> float:
    if days_per_year <= 0.0:
        raise ValueError("days_per_year must be positive")
    return days / days_per_year
