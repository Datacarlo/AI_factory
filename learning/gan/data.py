"""IID or grouped training rows, with disjoint fixed-condition evaluation."""
from dataclasses import replace
from pathlib import Path
import numpy as np
from learning.common.data import prepare_terminal_dataset
from learning.common.data.selection import DatasetSelection
from learning.common.data.terminal_cache import PreparedTerminalDataset
from learning.common.io import collection_path, fingerprint, sha256, write_json
from .models import MODELS
from .references import load_reference


def parameter_keys(conditions):
    values = np.ascontiguousarray(conditions[:, :-1], dtype=np.float32)
    return values.view(np.dtype((np.void, values.shape[1]*4))).ravel()


def prepare(root, recipe, *, cache_root=None):
    spec = MODELS[recipe['model']]
    if recipe['inputs']['time_convention'] != {'days_per_year':252,'T':'maturity_days / 252'} or recipe['observable']['name'] != (list(spec.observable) if isinstance(spec.observable, tuple) else spec.observable):
        raise ValueError('Recipe time or observable convention differs from the qualified adapter')
    if recipe['inputs']['conditions'] != list(spec.conditions):
        raise ValueError('Recipe input order differs from the qualified model coordinates')
    dataset = collection_path(root, recipe['dataset']['path'], 'datasets')
    reference_path = collection_path(root, recipe['reference']['path'], 'gan')
    conditions, values, reference = load_reference(reference_path, recipe['model'])
    for name in ('groups_per_split', 'paths_per_condition', 'simulation_seed'):
        if reference[name] != recipe['reference'][name]:
            raise ValueError(f'Reference {name} differs from recipe')
    cache_root = Path(cache_root or root/'work/learning/gan/cache')
    source = prepare_terminal_dataset(dataset, cache_root=cache_root/'source', observables=list(spec.observables))
    if source.schema.database_id != recipe['dataset']['id']:
        raise ValueError('Sample dataset ID differs from recipe')
    grouped = source.schema.paths_per_parameter > 1
    expected_policy = 'exclude_reference_parameters' if grouped else 'all'
    if source.schema.paths_per_parameter < 1 or recipe['data'] != {'training_rows':expected_policy, 'normalization':'training_only'}:
        raise ValueError('Training row policy differs from the sample layout')
    covered = set(spec.conditions) | set(recipe['inputs']['fixed'])
    if set(source.schema.conditioning_names) != covered:
        raise ValueError('Source parameters are not fully covered by the declared conditions and fixed inputs')
    if source.schema.days_per_year != 252:
        raise ValueError('Qualified native reference adapters use 252 trading days per year')
    identity = dict(dataset=dict(path=recipe['dataset']['path'], sha256=source.manifest['identity']['dataset_sha256']),
                    reference=dict(path=recipe['reference']['path'], sha256=sha256(reference_path)),
                    conditions=list(spec.conditions), observable=spec.observable,
                    fixed_inputs=recipe['inputs']['fixed'],
                    selection='grouped_exclude_reference_parameters_v1' if grouped else 'all_iid_rows')
    key = fingerprint(identity)
    directory = cache_root/key
    if (directory/'manifest.json').exists():
        return PreparedTerminalDataset.open(directory)
    columns = [source.schema.conditioning_names.index(n) for n in spec.conditions]
    fixed_keys = parameter_keys(conditions)
    if len(np.unique(fixed_keys)) != len(fixed_keys):
        raise ValueError('Reference parameter groups overlap, ignoring maturity')
    source_count = source.schema.row_count
    retained = []
    for start in range(0, source_count, 65536):
        stop = min(source_count, start+65536)
        chunk = np.asarray(source.conditions[start:stop])
        selected = chunk[:, columns]
        if not np.isfinite(chunk).all():
            raise ValueError('Invalid training parameters')
        overlap = np.isin(parameter_keys(selected), fixed_keys)
        if not grouped and overlap.any():
            raise ValueError('Training parameters overlap with reference groups')
        if grouped:
            retained.append(start + np.flatnonzero(~overlap))
        for name, value in recipe['inputs']['fixed'].items():
            if not np.all(chunk[:, source.schema.conditioning_names.index(name)] == value):
                raise ValueError('Input declared fixed actually varies: '+name)
        observed = source.observations[start:stop]
        if not np.isfinite(observed).all() or (spec.nonnegative and np.any(observed < 0)):
            raise ValueError('Invalid training observations')
    train_indices = np.concatenate(retained) if grouped else np.arange(source_count)
    count = len(train_indices)
    if not count:
        raise ValueError('No training rows remain after reserving reference parameters')
    if grouped:
        train_groups, group_counts = np.unique(source.parameter_groups[train_indices], return_counts=True)
        if np.any(group_counts != source.schema.paths_per_parameter):
            raise ValueError('Reference exclusion must preserve whole parameter packs')
        parameter_count = len(train_groups)
    else:
        parameter_count = count
    groups, paths = reference['groups_per_split'], reference['paths_per_condition']
    held_conditions = np.repeat(conditions, paths, axis=0)
    held_values = values[:, :paths].reshape(-1, len(spec.observables))
    total = count + len(held_values)
    specs = {'conditions':((total,len(columns)),np.float32),
             'observations':((total,len(spec.observables)),np.float32), 'maturity_days':((total,),np.int32),
             'parameter_groups':((total,),np.int64), 'split_codes':((total,),np.uint8)}
    directory.mkdir(parents=True, exist_ok=True)
    for name,(shape,dtype) in specs.items():
        array = np.lib.format.open_memmap(directory/(name+'.npy'), mode='w+',dtype=dtype,shape=shape)
        for start in range(0,count,65536):
            stop = min(count,start+65536)
            source_ids = train_indices[start:stop]
            if name == 'conditions': chunk = np.asarray(source.conditions[source_ids])[:,columns]
            elif name == 'split_codes': chunk = np.zeros(stop-start,dtype=np.uint8)
            elif name == 'parameter_groups':
                chunk = np.searchsorted(train_groups, source.parameter_groups[source_ids]) if grouped else np.arange(start,stop)
            else: chunk = getattr(source,name)[source_ids]
            array[start:stop] = chunk
        held = {'conditions':held_conditions, 'observations':held_values,
                'maturity_days':np.repeat(np.rint(conditions[:,-1]*252).astype(np.int32),paths),
                'parameter_groups':parameter_count+np.repeat(np.arange(2*groups),paths),
                'split_codes':np.repeat(np.array([1,2],np.uint8),groups*paths)}[name]
        array[count:] = held
        array.flush()
        del array
    schema = replace(source.schema,row_count=total,parameter_count=parameter_count+2*groups,
                     paths_per_parameter=0,conditioning_names=spec.conditions)
    checks = dict(train_rows=count,validation_rows=groups*paths,test_rows=groups*paths,
                  model_parameter_overlap=0,finite_samples=True,nonnegative_required=spec.nonnegative)
    if grouped:
        checks.update(source_rows=source_count,excluded_reference_rows=source_count-count,
                      source_parameter_groups=source.schema.parameter_count,train_parameter_groups=parameter_count,
                      paths_per_parameter=source.schema.paths_per_parameter)
    write_json(directory/'manifest.json',dict(schema_version=1,identity=identity,cache_key=key,
        tensor_schema=schema.to_dict(), arrays={n:n+'.npy' for n in specs},checks=checks))
    return PreparedTerminalDataset.open(directory)


def selection(data):
    indices = [np.flatnonzero(data.split_codes==i) for i in range(3)]
    grouped = data.manifest['identity']['selection'] == 'grouped_exclude_reference_parameters_v1'
    return DatasetSelection(*indices,len(indices[0]),len(indices[0]),0,2026112091,
        split_strategy=('grouped train excluding reference parameters; disjoint fixed-condition validation and reserved test'
                        if grouped else 'iid train; disjoint fixed-condition validation and reserved test'),
        group_manifest=data.manifest['checks'] if grouped else None)
