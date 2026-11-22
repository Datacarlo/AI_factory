"""Reusable fixed-condition native samples, separate from the IID training data."""
from pathlib import Path
import json
import shutil
import subprocess
import numpy as np
import torch
from learning.common.io import sha256, collection_path
from learning.common.data import prepare_terminal_dataset
from .models import MODELS


def save_reference(path, conditions, values, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, conditions=np.asarray(conditions, dtype=np.float32),
                            values=np.asarray(values, dtype=np.float32),
                            metadata=np.asarray(json.dumps(metadata, sort_keys=True)))
    temporary.replace(path)


def load_reference(path, model):
    with np.load(path, allow_pickle=False) as archive:
        conditions = archive['conditions']
        values = archive['values']
        metadata = json.loads(str(archive['metadata']))
    spec = MODELS[model]
    groups, paths = metadata['groups_per_split'], metadata['paths_per_condition']
    if (groups <= 0 or paths < 16 or conditions.shape != (2*groups, len(spec.conditions))
            or values.shape != ((2*groups, 2*paths) if len(spec.observables)==1 else (2*groups, 2*paths,len(spec.observables)))
            or metadata['model'] != model
            or metadata['condition_names'] != list(spec.conditions)
            or metadata['observable'] != (list(spec.observable) if isinstance(spec.observable,tuple) else spec.observable)
            or conditions.dtype != np.float32 or values.dtype != np.float32
            or not np.isfinite(conditions).all() or not np.isfinite(values).all()
            or (spec.nonnegative and np.any(values < 0)) or np.any(conditions[:, -1] <= 0)):
        raise ValueError('Invalid fixed-condition reference archive')
    return conditions, values, metadata


def simulate(root, model, conditions, *, paths, seed, work, nvcc='nvcc', architecture=None):
    """Reuse source dynamics; build only the small fixed-condition reference driver."""
    spec = MODELS[model]
    conditions = np.asarray(conditions, dtype=np.float32)
    if conditions.ndim != 2 or conditions.shape[1] != len(spec.conditions):
        raise ValueError('Reference coordinates differ from native adapter')
    days = conditions[:, -1].astype(np.float64)*252
    if (not np.isfinite(conditions).all() or np.any(days <= 0)
            or not np.allclose(days, np.rint(days), atol=1e-4, rtol=0)):
        raise ValueError('Reference maturities must be positive integral trading days')
    if model in ('rough_heston','quadratic_rough_heston') and np.any(conditions[:, -1] > 2):
        raise ValueError('Qualified rough Heston reference fit horizon is 2 years')
    if paths < 2 or len(conditions) > 65535:
        raise ValueError('Invalid native reference shape')
    if spec.native_id < 0:
        from .reference_adapter import simulate_native
        return simulate_native(root, model, conditions, paths=paths, seed=seed, work=work, nvcc=nvcc, architecture=architecture)
    compiler = shutil.which(nvcc)
    if compiler is None:
        raise FileNotFoundError('CUDA compiler required to create reference samples; use a supplied reference archive to train without nvcc')
    if architecture is None:
        architecture = ''.join(map(str, torch.cuda.get_device_capability()))
    if not str(architecture).isdigit():
        raise ValueError('architecture must be a CUDA SM number, e.g. 89')
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    source = root/'learning/gan/reference_kernel.cu'
    binary = work/'reference_simulator'
    with (work/'build.log').open('w') as log:
        subprocess.run([compiler, '-std=c++20', '-O3', '-arch=sm_'+str(architecture),
            '-I'+str(root/'src'), '-DMODEL='+str(spec.native_id), str(source), '-o', str(binary)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    input_path, output_path = work/'conditions.bin', work/'values.bin'
    conditions.tofile(input_path)
    subprocess.run([str(binary), str(input_path), str(output_path), str(len(conditions)),
                    str(paths), str(seed)], check=True)
    values = np.fromfile(output_path, dtype=np.float32).reshape(len(conditions), paths)
    if not np.isfinite(values).all() or (spec.nonnegative and np.any(values < 0)):
        raise ValueError('Native reference contains invalid observations')
    return values


def create_reference(root, recipe, *, nvcc='nvcc', architecture=None):
    reference = recipe['reference']
    existing = collection_path(root, reference['path'], 'gan')
    if existing.exists():
        load_reference(existing, recipe['model'])
        return existing
    if reference.get('selection') == 'stored_in_archive':
        raise FileNotFoundError('Download the reference archive supplied with this trained GAN; its original conditions are required to reproduce training')
    dataset = collection_path(root, reference['conditions_dataset']['path'], 'datasets')
    spec = MODELS[recipe['model']]
    print('Reference: reading grouped conditions', dataset, flush=True)
    grouped = prepare_terminal_dataset(dataset, cache_root=root/'work/learning/gan/cache',
                                       observables=list(spec.observables))
    if grouped.schema.database_id != reference['conditions_dataset']['id']:
        raise ValueError('Reference source dataset ID mismatch')
    groups = reference['groups_per_split']
    rng = np.random.default_rng(reference['selection_seed'])
    columns = [grouped.schema.conditioning_names.index(n) for n in spec.conditions]
    candidates = np.asarray(grouped.conditions[::grouped.schema.paths_per_parameter])[:, columns]
    # With few parameters (notably BS), independent float32 draws can coincide.
    # Exclude these model tuples before choosing holdouts, ignoring maturity.
    from .data import parameter_keys
    keys = parameter_keys(candidates)
    _, unique = np.unique(keys, return_index=True)
    eligible = np.zeros(len(keys), dtype=bool)
    eligible[unique] = True
    train_path = collection_path(root, recipe['dataset']['path'], 'datasets')
    print('Reference: checking disjoint IID parameters', train_path, flush=True)
    train = prepare_terminal_dataset(train_path, cache_root=root/'work/learning/gan/cache/source',
                                     observables=list(spec.observables))
    train_columns = [train.schema.conditioning_names.index(n) for n in spec.conditions]
    # Grouped training reserves these model tuples later, including all their paths.
    if recipe['data']['training_rows'] != 'exclude_reference_parameters':
        for start in range(0, train.schema.row_count, 65536):
            batch = np.asarray(train.conditions[start:start+65536])[:,train_columns]
            eligible &= ~np.isin(keys, parameter_keys(batch))
    print('Reference: eligible parameter groups', int(eligible.sum()), '/', len(eligible), flush=True)
    ids = rng.choice(np.flatnonzero(eligible), 2*groups, replace=False)
    conditions = candidates[ids]
    if 'spot' in spec.fixed:
        spot = grouped.schema.conditioning_names.index('spot')
        if not np.all(grouped.conditions[ids*grouped.schema.paths_per_parameter, spot] == 1):
            raise ValueError('This reference adapter requires fixed unit spot')
    print('Reference: compiling and simulating', recipe['model'], flush=True)
    values = simulate(root, recipe['model'], conditions,
        paths=2*reference['paths_per_condition'], seed=reference['simulation_seed'],
        work=root/'work/learning/gan/references'/recipe['id'], nvcc=nvcc, architecture=architecture)
    metadata = dict(schema_version=1, model=recipe['model'], condition_names=list(spec.conditions),
        observable=spec.observable, groups_per_split=groups,
        paths_per_condition=reference['paths_per_condition'], simulation_seed=reference['simulation_seed'],
        selection_seed=reference['selection_seed'], parameter_groups=ids.tolist(),
        method=spec.method, conditions_dataset=reference['conditions_dataset'],
        conditions_dataset_sha256=sha256(dataset))
    path = collection_path(root, reference['path'], 'gan', output=True)
    save_reference(path, conditions, values, metadata)
    return path
