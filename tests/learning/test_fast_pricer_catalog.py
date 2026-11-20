"""Catalogue relocation, shared entry points and local model publication paths."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from learning.fast_pricing.io import pricer_output_path, sha256
from learning.fast_pricing.recipes import ENTRYPOINT, make_recipe, run_recipe
from tools.datasets.metadata_schemas import catalogue_documents

ROOT = Path(__file__).resolve().parents[2]


def test_catalogue_reproduces_all_network_recipes():
    manifest = json.loads((ROOT/'catalog/fast_pricers/manifest.json').read_text())
    paths = set()
    for record in manifest['datasets']:
        expected_count = 12 if record['kind'] == 'price_gradients' else 3
        assert len(record['recipes']) == expected_count
        for relative in record['recipes']:
            assert relative.startswith('catalog/fast_pricers/')
            assert relative not in paths
            paths.add(relative)
            path = ROOT/relative
            recipe = yaml.safe_load(path.read_text())
            size = path.parent.parent.name
            objective = recipe['objective']
            assert recipe == make_recipe(record, recipe['inputs'], size,
                                         objective['method'], objective.get('lambda', 0.))
            assert ROOT/recipe['training'] == path.with_name('training.py')
            assert path.with_name('training.py').read_text() == ENTRYPOINT
    assert paths == {str(p.relative_to(ROOT)) for p in
                     (ROOT/'catalog/fast_pricers').rglob('recipe.yaml')}
    assert not (ROOT/'learning/fast_pricing/catalog').exists()


def test_dataset_metadata_scan_keeps_its_own_scope():
    documents = list(catalogue_documents(ROOT))
    assert documents
    assert not any('fast_pricers' in p.relative_to(ROOT/'catalog').parts for _, p in documents)


def test_catalogue_entrypoint_works_outside_repository(tmp_path):
    leaf = next((ROOT/'catalog/fast_pricers').rglob('training.py'))
    result = subprocess.run([sys.executable, str(leaf), '--help'], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert '--output-root' in result.stdout


@pytest.mark.parametrize('isolated', [False, True])
def test_recipe_exports_model_and_receipt_to_selected_destination(tmp_path, isolated):
    record = dict(id='test', logical_path='datasets/single_asset/equity/test.json')
    recipe = make_recipe(record, {}, 'small', 'price')
    path = tmp_path/Path(recipe['training']).with_name('recipe.yaml')
    path.parent.mkdir(parents=True)
    path.write_text(yaml.safe_dump(recipe))
    prepared = SimpleNamespace(schema=SimpleNamespace(database_id='test'),
        manifest=dict(dataset=record['logical_path'], inputs={},
                      identity={'dataset_sha256': 'a'*64}, mother_paths={}))
    selection = SimpleNamespace(manifest=lambda: {'train': 8, 'validation': 1, 'test': 1})
    output_root = tmp_path/'isolated' if isolated else None
    run_root = output_root if isolated else tmp_path/'work/learning/fast_pricing'

    def train(_prepared, _selection, _config, run_directory, **kwargs):
        assert run_directory == run_root/'runs'/recipe['id']
        run_directory.mkdir(parents=True)
        (run_directory/'best_model.pt').write_bytes(b'unchanged-checkpoint')
        return SimpleNamespace(manifest=dict(metrics=dict(validation={}, test={}, strata={}),
                               initial_model_hash='initial', runtime={'parameter_count': 1}))

    with patch('learning.fast_pricing.recipes.split_dataset', return_value=selection), \
         patch('learning.fast_pricing.recipes.configuration', return_value={}), \
         patch('learning.fast_pricing.recipes.train_experiment', side_effect=train), \
         patch('learning.fast_pricing.recipes.environment', return_value={'hardware': {}, 'software': {}}):
        run_recipe(path, tmp_path, prepared=prepared, output_root=output_root)

    receipt = yaml.safe_load(path.with_name('generation.yaml').read_text())
    for name, logical in recipe['output'].items():
        physical = pricer_output_path(tmp_path, logical, output_root=output_root)
        assert physical.is_file()
        assert receipt['output'][name] == {'path': logical, 'sha256': sha256(physical)}
        if not isolated:
            assert physical.is_relative_to(tmp_path/'fast_pricers/unpublished')
    model = pricer_output_path(tmp_path, recipe['output']['model'], output_root=output_root)
    assert model.read_bytes() == b'unchanged-checkpoint'
    assert receipt['inputs']['dataset']['sha256'] == 'a'*64
    assert not (tmp_path/'fast_pricers/published').exists()


def test_model_output_requires_logical_collection_path(tmp_path):
    for bad in ('../outside.pt', '/tmp/outside.pt', 'datasets/test.json', 'fast_pricers'):
        with pytest.raises(ValueError):
            pricer_output_path(tmp_path, bad)
