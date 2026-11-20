"""A compact recipe per network; one shared preparation and training engine."""
from __future__ import annotations
import platform
from pathlib import Path

import numpy as np
import torch
import yaml

from learning.common.data.transforms import fit_standardization
from .data import prepare_equity, split_dataset
from .engine import train_experiment
from .io import sha256, write_json, write_yaml
from .supervision import gradient_statistics

ARCHITECTURES = {'small': [64] * 3, 'medium': [128] * 4, 'large': [512] * 4}
LAMBDAS = (0.1, 1.0, 10.0)
TRAINING = dict(seed=4242, epochs=500, min_epochs=50, batch_size=1024,
    learning_rate=0.001, optimizer='adam', weight_decay=0., scheduler='plateau',
    lr_factor=0.5, lr_patience=10, min_learning_rate=1e-6,
    improvement_threshold=1e-4, early_stopping_patience=40,
    validation_interval=1, checkpoint_interval=10, deterministic=True,
    resident=True, device='cuda', evaluate_test=True)


def make_recipe(record, inputs, size, method, lam=0.):
    suffix = 'price' if method == 'price' else f'sobolev_lambda_{lam:g}'
    identity = f"{record['id']}__{size}__{suffix}"
    logical = record['logical_path']
    dataset_rel = Path(*Path(logical).parts[1:]).with_suffix('')
    leaf = Path('learning/fast_pricing/catalog') / dataset_rel / size / suffix
    output = Path('fast_pricers') / dataset_rel / size / suffix
    return dict(schema_version=1, kind='fast_pricer', id=identity,
        training=str(leaf / 'training.py'),
        dataset={'id': record['id'], 'path': logical},
        inputs=inputs,
        data=dict(max_rows=100000, split='model_groups', split_seed=20261023,
                  selection_seed=1729, core_model_count=900,
                  validation_fraction=0.1, test_fraction=0.1),
        network=dict(name='mlp', hidden_sizes=ARCHITECTURES[size], activation='silu'),
        objective=dict(method=method, **({'lambda': lam, 'gradient_weighting': 'train_rms',
                       'Rj_threshold': 0.04, 'cell_mask': False} if method == 'sobolev' else {})),
        training_options=dict(TRAINING),
        output={'model': str(output/'model.pt'), 'metrics': str(output/'metrics.json')})


ENTRYPOINT = '''"""Train this network using the shared fast-pricing routine."""
from pathlib import Path
import sys

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "learning/fast_pricing").is_dir())
sys.path.insert(0, str(ROOT))
from learning.fast_pricing.train import recipe_main

if __name__ == "__main__":
    recipe_main(Path(__file__).with_name("recipe.yaml"))
'''


def configuration(recipe, prepared, selection):
    objective = recipe['objective']
    config = dict(network=recipe['network'], representation={'name': 'identity'},
                  training=dict(recipe['training_options']), method=objective['method'])
    terms = [{'name': 'value_mse', 'weight': 1.}]
    if objective['method'] == 'sobolev':
        if objective.get('cell_mask', False):
            raise ValueError('This campaign uses Rj only, including VG')
        transform = fit_standardization(prepared.features, prepared.values, selection.train_indices)
        names = [g.wrt_name for g in prepared.schema.gradients]
        indices = np.array([g.wrt_index for g in prepared.schema.gradients], dtype=np.int64)
        statistics = gradient_statistics(prepared.gradients, prepared.gradient_standard_errors,
            selection.train_indices, names, transform.normalized_gradient_scales(indices),
            threshold=objective['Rj_threshold'], closed_formula=prepared.manifest['closed_formula'])
        terms.append(dict(name='rms_gradient_mse', weight=objective['lambda'],
                          weights=statistics['squared_error_weights']))
        config['supervision'] = statistics
    elif objective['method'] != 'price':
        raise ValueError('Unknown training objective')
    config['loss'] = {'terms': terms}
    return config


def environment():
    cpu = platform.processor() or platform.machine()
    if Path('/proc/cpuinfo').exists():
        cpu = next((s.split(':', 1)[1].strip() for s in Path('/proc/cpuinfo').read_text().splitlines()
                    if s.startswith('model name')), cpu)
    hardware = dict(device='gpu' if torch.cuda.is_available() else 'cpu', cpu_model=cpu)
    if torch.cuda.is_available():
        hardware.update(name=torch.cuda.get_device_name(),
                        compute_capability='.'.join(map(str, torch.cuda.get_device_capability())))
    return dict(hardware=hardware, software=dict(python=platform.python_version(),
        pytorch=str(torch.__version__), cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), numpy=np.__version__, pyyaml=yaml.__version__))


def run_recipe(recipe_path: Path, root: Path, *, cache_root=None, prepared=None,
               output_root=None, resident=None, resume=False):
    recipe = yaml.safe_load(recipe_path.read_text())
    cache_root = cache_root or root/'work/learning/fast_pricing/cache'
    output_root = output_root or root/'work/learning'
    if prepared is None:
        prepared = prepare_equity(root, recipe['dataset']['path'], cache_root,
                                  row_limit=recipe['data']['max_rows'])
    if (prepared.schema.database_id != recipe['dataset']['id']
            or prepared.manifest['dataset'] != recipe['dataset']['path']):
        raise ValueError('Prepared dataset identity mismatch')
    if prepared.manifest['inputs'] != recipe['inputs']:
        raise ValueError('Input specification changed; update the recipe explicitly')
    selection = split_dataset(prepared, recipe['data'])
    config = configuration(recipe, prepared, selection)
    model_path = output_root/recipe['output']['model']
    run_directory = output_root/'runs'/recipe['id']
    result = train_experiment(prepared, selection, config, run_directory,
                              resident=resident, resume=resume)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    # The best checkpoint contains weights, schema, fixed inputs, normalization and Rj decisions.
    import shutil
    temporary = model_path.with_suffix('.tmp')
    shutil.copyfile(run_directory/'best_model.pt', temporary)
    temporary.replace(model_path)
    metrics_path = output_root/recipe['output']['metrics']
    write_json(metrics_path, dict(validation=result.manifest['metrics']['validation'],
        test=result.manifest['metrics']['test'], strata=result.manifest['metrics']['strata'],
        selection=selection.manifest(),
        initial_model_hash=result.manifest['initial_model_hash'],
        parameter_count=result.manifest['runtime']['parameter_count']))
    inputs = {'dataset': dict(path=recipe['dataset']['path'],
                             sha256=prepared.manifest['identity']['dataset_sha256'])}
    for role, path in prepared.manifest['mother_paths'].items():
        inputs[role] = dict(path=path, sha256=prepared.manifest['identity']['inputs'][role])
    generation = dict(schema_version=1, status='complete', inputs=inputs,
        output={name: dict(path=recipe['output'][name], sha256=sha256(path))
                for name,path in [('model',model_path),('metrics',metrics_path)]}, **environment())
    write_yaml(recipe_path.with_name('generation.yaml'), generation)
    return result
