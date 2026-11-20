"""Create compact training leaves for published equity pricing datasets."""
from pathlib import Path
import argparse
from .data import inspect_dataset, input_specification
from .io import repository_root, write_json, write_yaml
from .recipes import ARCHITECTURES, LAMBDAS, ENTRYPOINT, make_recipe


def add_dataset(root: Path, logical_path: str):
    context = inspect_dataset(root, logical_path)
    record = dict(id=context['header']['database_id'], logical_path=logical_path,
                  model=context['model'], kind=context['kind'], product=context['product'],
                  rows=context['header']['row_count'], recipes=[])
    inputs = input_specification(context)
    variants = [('price', 0.)]
    if record['kind'] == 'price_gradients':
        variants += [('sobolev', value) for value in LAMBDAS]
    for size in ARCHITECTURES:
        for method, value in variants:
            recipe = make_recipe(record, inputs, size, method, value)
            leaf = root/Path(recipe['training']).parent
            leaf.mkdir(parents=True, exist_ok=True)
            # Updating a completed training recipe must be an explicit operation.
            if (leaf/'generation.yaml').exists():
                import yaml
                if yaml.safe_load((leaf/'recipe.yaml').read_text()) != recipe:
                    raise ValueError('Completed training leaf differs: ' + str(leaf))
            (leaf/'training.py').write_text(ENTRYPOINT)
            write_yaml(leaf/'recipe.yaml', recipe)
            record['recipes'].append(str((leaf/'recipe.yaml').relative_to(root)))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('datasets', nargs='+', help='Logical paths starting with datasets/single_asset/equity/')
    parser.add_argument('--root', type=Path, default=repository_root(Path(__file__)))
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    records = [add_dataset(args.root, name) for name in args.datasets]
    write_json(args.manifest, dict(schema_version=1, datasets=records))


if __name__ == '__main__':
    main()
