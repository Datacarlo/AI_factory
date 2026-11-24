"""Create a GAN leaf from one sample dataset, with no model-specific training loop."""
import argparse
import json
from pathlib import Path
import yaml
from learning.common.io import collection_path, repository_root, write_json, write_yaml
from learning.common.terminal_samples import iter_terminal_samples
from .models import model_for_path
from .recipes import ENTRYPOINT, make_recipe


def create_leaf(root, dataset):
    path = Path(dataset)
    if path.is_absolute(): path = path.relative_to(root)
    parts = list(path.parts)
    if len(parts)>1 and parts[:2] in (['datasets','published'],['datasets','unpublished']):
        del parts[1]
    logical=Path(*parts)
    actual=collection_path(root,str(logical),'datasets')
    model,spec=model_for_path(logical)
    reader=iter_terminal_samples(actual,limit=1)
    try: schema,_=next(reader)
    finally: reader.close()
    if schema.paths_per_parameter != 1:
        raise ValueError('The public GAN catalogue uses IID samples: one path per parameter')
    recipe=make_recipe(model,schema.database_id,dataset_name=logical.stem)
    path=root/Path(recipe['training']).with_name('recipe.yaml')
    if path.exists():
        existing=yaml.safe_load(path.read_text())
        if existing['dataset']!=recipe['dataset']:
            raise ValueError('Existing GAN leaf refers to a different sample dataset')
        return path
    write_yaml(path,recipe)
    path.with_name('training.py').write_text(ENTRYPOINT)
    manifest=root/'catalog/gan/manifest.json'
    inventory=json.loads(manifest.read_text()) if manifest.exists() else {'schema_version':1,'networks':[]}
    inventory['networks'].append(dict(id=recipe['id'],model=model,dataset=recipe['dataset'],recipe=str(path.relative_to(root))))
    write_json(manifest,inventory)
    return path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset')
    parser.add_argument('--root',type=Path)
    args=parser.parse_args()
    root=(args.root or repository_root(Path(__file__))).resolve()
    print(create_leaf(root,args.dataset))


if __name__=='__main__': main()
