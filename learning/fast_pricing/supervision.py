"""Train-only Rj filtering and RMS balancing; no row or cell exclusion."""
from __future__ import annotations

import numpy as np


def gradient_statistics(gradients, errors, train_indices, names, scales, *, threshold=0.04, closed_formula=False):
    values = np.asarray(gradients[train_indices], dtype=np.float64)
    se = np.asarray(errors[train_indices], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Non-finite gradient targets')
    rms = np.sqrt(np.mean(np.square(values * scales), axis=0))
    positive = rms[rms > 1e-12]
    floor = max(1e-6, 0.1 * float(np.median(positive))) if len(positive) else 1e-6
    denominators = np.maximum(rms, floor)
    diagnostics = {}
    weights = []
    for j, name in enumerate(names):
        energy = float(np.square(values[:, j]).sum())
        if closed_formula and np.isnan(se[:, j]).all():
            ratio, keep, reason = None, True, 'closed formula; no Monte Carlo error'
        elif not np.isfinite(se[:, j]).all() or (se[:, j] < 0).any():
            raise ValueError('Missing or invalid gradient standard errors: ' + name)
        elif energy == 0:
            ratio, keep, reason = None, not bool(np.any(se[:, j] > 0)), 'zero train target; retain only if standard errors are zero'
        else:
            ratio = float(np.square(se[:, j]).sum() / energy)
            keep = ratio <= threshold
            reason = 'Rj <= threshold' if keep else 'Rj > threshold'
        weights.append(float(keep) / float(denominators[j] ** 2))
        diagnostics[name] = dict(Rj=ratio, supervised=keep, reason=reason,
                                 normalized_target_rms=float(rms[j]),
                                 normalized_error_denominator=float(denominators[j]))
    return dict(threshold=threshold, rms_floor=floor, coordinates=diagnostics,
                squared_error_weights=weights, denominator=len(names),
                policy='Rj only; original coordinate count retained after filtering')
