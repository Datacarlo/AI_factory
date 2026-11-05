"""Shared presentation of pricing recipes and compact generation receipts.

No numerical settings are inferred from model names. Unknown fields are kept,
so formatting does not freeze the metadata vocabulary.
"""
from copy import deepcopy
import math

import yaml

RECIPE_ORDER = (
    'schema_version', 'kind', 'dataset_id', 'option_side', 'generator',
    'inputs', 'output', 'row_layout', 'outputs',
    'row_count', 'candidate_row_count', 'construction', 'product_shard',
    'curve_assignment', 'row_selection', 'quote_convention',
    'numerical_method', 'central_price_engine', 'time_grid', 'time_representation',
    'paths_per_price', 'random_number_generator', 'seeds',
    'sensitivity', 'sensitivity_strategy', 'exercise_replay',
    'fit_and_valuation_paths', 'standard_error_scope', 'execution',
)
EXECUTION_FIELDS = (
    'team_size', 'rows_per_launch', 'streams', 'threads_per_block',
    'paths_per_team', 'path_chunk_size', 'checkpoint_rows',
)
GRID_ORDER = ('steps_per_year', 'simulation_steps_per_day', 'delta_t', 'monitoring')


def ordered(data, keys):
    return {key: data[key] for key in keys if key in data} | {
        key: value for key, value in data.items() if key not in keys
    }


def normalize_recipe(document):
    result = deepcopy(document)
    if result.get('kind') not in {'prices', 'price_gradients', 'price_delta'}:
        return result
    # Historical multi-asset recipes used different names for the same settings.
    legacy = result.pop('pricing', None)
    if legacy is not None:
        steps = legacy.pop('steps_per_business_day', None)
        dt = legacy.pop('dt', None)
        if steps is None or steps <= 0:
            raise ValueError('Multi-asset recipe lacks a positive steps_per_business_day')
        year_steps = 252 * steps
        if dt is not None and not math.isclose(dt, 1 / year_steps, rel_tol=1e-12):
            raise ValueError('Contradictory multi-asset dt and steps_per_business_day')
        grid = dict(steps_per_year=year_steps, simulation_steps_per_day=steps,
                    delta_t=f'1 / {year_steps}')
        if 'barrier_monitoring' in legacy:
            grid['monitoring'] = legacy.pop('barrier_monitoring')
        if 'time_grid' in result and result['time_grid'] != grid:
            raise ValueError('Conflicting time grids')
        result['time_grid'] = grid
        execution = {key: legacy.pop(key) for key in EXECUTION_FIELDS if key in legacy}
        for name, additions in [('numerical_method', legacy), ('execution', execution)]:
            target = result.setdefault(name, {})
            for key, value in additions.items():
                if key in target and target[key] != value:
                    raise ValueError(f'Conflicting {name}.{key}')
                target[key] = value
    if isinstance(result.get('time_grid'), dict):
        result['time_grid'] = ordered(result['time_grid'], GRID_ORDER)
    if isinstance(result.get('sensitivity'), dict):
        result['sensitivity'] = ordered(result['sensitivity'], (
            'method', 'orders', 'bump_rule', 'parameters', 'mixed_second',
            'reconstructible_from_price', 'spot_contract_rule', 'maturity_coordinate',
        ))
    if isinstance(result.get('execution'), dict):
        result['execution'] = ordered(result['execution'], EXECUTION_FIELDS)
    return ordered(result, RECIPE_ORDER)


def recipe_yaml(document):
    return yaml.safe_dump(normalize_recipe(document), sort_keys=False, allow_unicode=True)


def normalize_generation(document):
    result = deepcopy(document)
    for name, keys in (
        ('output', ('path', 'sha256')),
        ('hardware', ('device', 'cpu_model', 'name', 'uuid', 'compute_capability', 'driver_version')),
        ('software', ('cmake', 'cxx_compiler', 'cuda_compiler', 'cuda_target_architectures',
                      'nlohmann_json', 'mathdx')),
    ):
        if isinstance(result.get(name), dict):
            result[name] = ordered(result[name], keys)
    for value in result.get('inputs', {}).values():
        if isinstance(value, dict):
            reordered = ordered(value, ('path', 'sha256'))
            value.clear()
            value.update(reordered)
    return ordered(result, ('schema_version', 'status', 'inputs', 'output', 'hardware', 'software'))
