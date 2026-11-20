"""Equity contract coordinates, streamed once and shared by all training variants.

No pricing or mutation of published data occurs here. Labels are converted only
when an input quote is converted to its absolute contract level.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from learning.common.data.cache import PreparedPricingDataset
from learning.common.data.calendar import product_features, rejects_legacy_maturity
from learning.common.data.contracts import GradientTarget, PricingTensorSchema
from learning.common.data.json_stream import open_object_array
from learning.common.data.selection import make_dataset_selection
from .io import dataset_path, fingerprint, sha256, write_json
from .subsample import select_rows

POLICY = 'equity_contract_coordinates_v1'


def inspect_dataset(root: Path, logical_path: str) -> dict:
    parts = Path(logical_path).parts
    if parts[:3] != ('datasets', 'single_asset', 'equity'):
        raise ValueError('This preparation policy covers single-asset equity only')
    path = dataset_path(root, logical_path)
    recipe_path = root / 'catalog' / Path(*parts[1:]).with_suffix('') / 'recipe.yaml'
    recipe = yaml.safe_load(recipe_path.read_text())
    with open_object_array(path, 'results') as (header, rows):
        first = next(rows)
    if recipe['dataset_id'] != header['database_id']:
        raise ValueError('Dataset and recipe identities disagree')
    mothers, removed = {}, {}
    for role, relative in recipe['inputs'].items():
        if role not in ('model', 'product'):
            raise ValueError('Unsupported equity input role: ' + role)
        parent = dataset_path(root, relative)
        document = json.loads(parent.read_text())
        parent_rows = {str(r['id']): r['parameters'] for r in document[role + 's']}
        parent_id = header[role + '_dataset']
        if parent_id.get('id', parent_id.get('database_id')) != document['database_id']:
            raise ValueError('Parent identity mismatch: ' + role)
        mothers[role] = dict(path=parent, logical_path=relative, rows=parent_rows, document=document)
        values = list(parent_rows.values())
        for name, value in values[0].items():
            if all(row[name] == value for row in values):
                if role == 'model' or name in ('notional', 'cash_payoff'):
                    removed[role + '.' + name] = float(value)
    model, kind, product = parts[4:7]
    if model in ('sabr', 'rough_sabr', 'cev'):
        removed.pop('model.spot', None)
    time = header['time_convention']
    if time != dict(unit='business_day', days_per_year=252):
        raise ValueError('Unsupported time convention')
    quotes = header.get('quote_convention', {})
    if quotes != mothers['product']['document'].get('quote_convention', {}):
        raise ValueError('Product quote conventions disagree')
    for row in mothers['product']['rows'].values():
        product_features(row, time)
    names = [role + '.' + name for role in ('model', 'product')
             for name in first[role + '_parameters']]
    return dict(path=path, logical_path=logical_path, recipe=recipe, recipe_path=recipe_path,
                header=header, first=first, mothers=mothers, removed=removed, raw_names=names,
                model=model, kind=kind, product=product, time=time, quotes=quotes)


def encode_parameters(raw: np.ndarray, context: dict):
    values = {name: raw[:, i].copy() for i, name in enumerate(context['raw_names'])}
    product_names = [name for name in values if name.startswith('product.')]
    unique, inverse = np.unique(np.stack([values[n] for n in product_names], axis=1),
                                axis=0, return_inverse=True)
    encoded = [product_features(dict(zip([n[8:] for n in product_names], row)), context['time'])
               for row in unique]
    keys = list(encoded[0])
    if any(list(row) != keys for row in encoded):
        raise ValueError('Inconsistent contractual calendar inputs')
    for name in product_names:
        del values[name]
    for name in keys:
        values['product.' + name] = np.array([row[name] for row in encoded])[inverse]
    for name, constant in context['removed'].items():
        if name in values:
            if not np.all(values[name] == constant):
                raise ValueError('A declared constant varies: ' + name)
            del values[name]
    factors = {}
    if 'model.spot' in values:
        spot = values['model.spot']
        if not np.all(spot > 0):
            raise ValueError('Absolute contract coordinates require positive spot')
        for field, convention in context['quotes'].items():
            if convention['mode'] == 'ratio_to_central_spot':
                name = 'product.' + field
                values[name] *= spot
                factors[name] = spot
        if 'autocall' in context['product']:
            # S_ref is fixed by the contract during a model-spot bump.
            values['product.reference_spot'] = spot.copy()
    names = sorted(values)
    gradients, indices, dropped = [], [], {}
    for name in context['first'].get('gradients', {}):
        if name == 'product.maturity_years' and rejects_legacy_maturity(tuple(names)):
            raise ValueError('Retired calendar maturity label must be migrated before training')
        if name in names:
            gradients.append(name)
            indices.append(names.index(name))
        elif name in context['removed']:
            dropped[name] = 'constant parent domain'
        else:
            raise ValueError('Unresolved gradient coordinate: ' + name)
    return np.stack([values[n] for n in names], axis=1), names, gradients, indices, factors, dropped


def input_specification(context: dict) -> dict:
    first = context['first']
    raw = np.array([[first[n.split('.')[0] + '_parameters'][n.split('.')[1]]
                     for n in context['raw_names']]], dtype=np.float64)
    _, names, gradients, _, factors, dropped = encode_parameters(raw, context)
    return dict(policy=POLICY, features=names, fixed_inputs=context['removed'],
                absolute_contract_levels=list(factors), eligible_gradients=gradients,
                omitted_gradients=dropped, time_unit='year', business_days_per_year=252,
                target='price in original dataset units')


def split_dataset(prepared: PreparedPricingDataset, specification: dict):
    if specification.get('split', 'model_groups') != 'model_groups':
        raise ValueError('Catalogue training requires the model-group split')
    return make_dataset_selection(prepared.schema.row_count, train_size=None,
        split_seed=specification.get('split_seed', 20261023),
        selection_seed=specification.get('selection_seed', 1729),
        validation_fraction=specification.get('validation_fraction', .1),
        test_fraction=specification.get('test_fraction', .1),
        group_ids=prepared.entity_ordinals[:, 0],
        core_group_count=specification.get('core_model_count', 900))


def open_cache(directory: Path, *, verify=True) -> PreparedPricingDataset:
    prepared = PreparedPricingDataset.open(directory)
    if prepared.manifest.get('preparation_policy') != POLICY:
        raise ValueError('Obsolete fast-pricing preparation cache')
    if verify:
        for name, digest in prepared.manifest['array_sha256'].items():
            if sha256(directory / name) != digest:
                raise ValueError('Corrupt training cache: ' + name)
    return prepared


def prepare_equity(root: Path, logical_path: str, cache_root: Path, *, row_limit=100_000):
    context = inspect_dataset(root, logical_path)
    path = context['path']
    identity = dict(dataset_sha256=sha256(path),
                    inputs={role: sha256(m['path']) for role, m in context['mothers'].items()},
                    policy=POLICY, row_limit=row_limit)
    key = fingerprint(identity)
    directory = cache_root / key
    if (directory / 'manifest.json').is_file():
        return open_cache(directory)
    n = context['header']['row_count']
    raw_names = context['raw_names']
    original_gradients = list(context['first'].get('gradients', {}))
    raw = np.empty((n, len(raw_names)), dtype=np.float64)
    gradients = np.empty((n, len(original_gradients)), dtype=np.float64)
    errors = np.full_like(gradients, np.nan)
    values, price_errors = np.empty((n, 1), np.float32), np.empty((n, 1), np.float32)
    model_ids, product_ids = np.empty(n, np.int64), np.empty(n, np.int64)
    closed = context['recipe'].get('paths_per_price') == 0
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    count = 0
    with open_object_array(path, 'results') as (_, rows):
        for i, row in enumerate(rows):
            if i >= n or row['id'] != i:
                raise ValueError('Noncontiguous result IDs or invalid row count')
            flat = {}
            for role, mother in context['mothers'].items():
                parameters = row[role + '_parameters']
                if parameters != mother['rows'][str(row[role + '_id'])]:
                    raise ValueError(f'Inline parameters disagree with parent: row {i}, {role}')
                flat.update({role + '.' + k: v for k, v in parameters.items()})
            raw[i] = [flat[name] for name in raw_names]
            labels = row.get('gradients', {})
            if list(labels) != original_gradients:
                raise ValueError('Inconsistent gradient schema')
            gradients[i] = list(labels.values())
            if 'gradient_standard_errors' in row:
                if list(row['gradient_standard_errors']) != original_gradients:
                    raise ValueError('Gradient error schema mismatch')
                errors[i] = list(row['gradient_standard_errors'].values())
            elif original_gradients and not closed:
                raise ValueError('Monte Carlo gradient errors missing')
            values[i] = row['price']
            if 'standard_error' not in row and not closed:
                raise ValueError('Monte Carlo price error missing')
            price_errors[i] = row.get('standard_error', 0.)
            model_ids[i], product_ids[i] = int(row['model_id']) - 1, int(row['product_id'])
            count += 1
    if count != n or before != (path.stat().st_size, path.stat().st_mtime_ns):
        raise ValueError('Dataset incomplete or changed during preparation')
    selected, row_selection = select_rows(model_ids, product_ids, np.arange(n), limit=row_limit)
    x, names, eligible, gradient_indices, factors, dropped = encode_parameters(raw[selected], context)
    columns = [original_gradients.index(name) for name in eligible]
    g, se = gradients[selected][:, columns], errors[selected][:, columns]
    for j, name in enumerate(eligible):
        if name in factors:
            g[:, j] /= factors[name]
            se[:, j] /= factors[name]
    count = len(selected)
    arrays = dict(features=x.astype(np.float32), values=values[selected], gradients=g.astype(np.float32),
                  gradient_standard_errors=se.astype(np.float32), value_standard_errors=price_errors[selected],
                  entity_ordinals=np.stack([model_ids[selected], product_ids[selected] - 1], axis=1),
                  diagonal_hessians=np.empty((count, 0), np.float32),
                  diagonal_hessian_standard_errors=np.empty((count, 0), np.float32), row_ids=selected)
    for name, array in arrays.items():
        if name == 'gradient_standard_errors' and closed and np.isnan(array).all():
            continue
        if not np.isfinite(array).all() or ('standard_errors' in name and (array < 0).any()):
            raise ValueError('Invalid values in ' + name)
    schema = PricingTensorSchema(context['header']['database_id'], count, tuple(names), ('price',),
        tuple(GradientTarget('gradients:' + name, 'price', name, index,
                             None if closed else 'gradient_standard_errors:' + name)
              for name, index in zip(eligible, gradient_indices)), ('model', 'product'))
    directory.mkdir(parents=True, exist_ok=True)
    for name, array in arrays.items():
        np.save(directory / (name + '.npy'), array)
    manifest = dict(schema_version=4, preparation_policy=POLICY, cache_key=key, identity=identity,
        tensor_schema=schema.to_dict(), arrays={k: k + '.npy' for k in arrays},
        array_sha256={k + '.npy': sha256(directory / (k + '.npy')) for k in arrays},
        dataset=logical_path, closed_formula=closed, inputs=input_specification(context),
        row_selection=row_selection, mother_paths={k: m['logical_path'] for k, m in context['mothers'].items()},
        dropped_gradients=dropped)
    write_json(directory / 'manifest.json', manifest)
    return open_cache(directory)
