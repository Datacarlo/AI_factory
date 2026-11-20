"""Reproducible row budget, stratified by model ID and product core/stress."""
import hashlib
import numpy as np
from learning.common.data.selection import _splitmix64

ROW_LIMIT = 100_000
ROW_SEED = 20261116

def select_rows(groups, product_ids, row_ids, limit=ROW_LIMIT):
    n = len(groups)
    if n <= limit:
        ids = np.arange(n, dtype=np.int64)
    else:
        keys = np.column_stack((groups, np.asarray(product_ids) > 900))
        strata, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        if len(strata) > limit:
            raise ValueError('Row budget cannot retain every model/product-regime stratum')
        ideal = counts.astype(float) * limit / n
        quotas = np.maximum(1, np.floor(ideal).astype(np.int64))
        difference = limit - int(quotas.sum())
        if difference > 0:
            order = np.argsort(-(ideal - quotas), kind='stable')
            for j in order:
                if quotas[j] < counts[j]:
                    quotas[j] += 1
                    difference -= 1
                    if not difference: break
        elif difference < 0:
            while difference < 0:
                eligible = np.flatnonzero(quotas > 1)
                j = eligible[np.argmax((quotas - ideal)[eligible])]
                quotas[j] -= 1
                difference += 1
        assert difference == 0 and quotas.sum() == limit
        # Stable pseudorandom priorities keyed by ORIGINAL row identity.
        priority = _splitmix64(np.asarray(row_ids, dtype=np.uint64), ROW_SEED)
        order = np.lexsort((row_ids, priority, inverse))
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        ids = np.sort(np.concatenate([order[start:start+quota] for start, quota in zip(starts, quotas)]))
        assert np.array_equal(np.unique(groups[ids]), np.unique(groups))
    assert len(ids) == min(n, limit) and len(np.unique(ids)) == len(ids)
    digest = hashlib.sha256(np.asarray(row_ids[ids], dtype='<i8').tobytes()).hexdigest()
    return ids, {'source_rows': n, 'selected_rows': len(ids), 'limit': limit, 'seed': ROW_SEED,
                 'method': 'all rows' if n <= limit else 'model_id x product_core_stress, proportional quotas, SplitMix64 row priority, without replacement',
                 'original_row_ids_sha256': digest}
