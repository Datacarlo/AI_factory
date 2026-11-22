"""Train conditional terminal GANs with common catalogue recipes or experimental configs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learning.common.data import (
    make_preassigned_selection,
    prepare_terminal_dataset,
)

from .config import load_config
from .engine import train_experiment


_UNSET = object()


def _train_size(value: str) -> int | None:
    if value.lower() == "all":
        return None
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError(
            "train size must be positive or 'all'"
        )
    return number


def _overrides(arguments: argparse.Namespace) -> dict[str, Any]:
    value: dict[str, Any] = {}
    if arguments.dataset is not None:
        value["dataset"] = arguments.dataset
    configured_train_size = arguments.train_size
    if configured_train_size is not _UNSET:
        value.setdefault("data", {})["train_size"] = (
            configured_train_size
        )
    for source, target in (
        ("split_seed", "split_seed"),
        ("selection_seed", "selection_seed"),
    ):
        configured = getattr(arguments, source)
        if configured is not None:
            value.setdefault("data", {})[target] = configured
    if arguments.observables:
        value.setdefault("data", {})["observables"] = (
            arguments.observables
        )
    for source, target in (
        ("epochs", "epochs"),
        ("batch_size", "batch_size"),
        ("critic_steps", "critic_steps"),
        ("generator_learning_rate", "generator_learning_rate"),
        ("critic_learning_rate", "critic_learning_rate"),
        ("device", "device"),
    ):
        configured = getattr(arguments, source)
        if configured is not None:
            value.setdefault("training", {})[target] = configured
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dataset", type=Path)
    parser.add_argument(
        "--observable",
        dest="observables",
        action="append",
        help=(
            "observable to generate; repeat for multiple outputs; "
            "the dataset default is all observables"
        ),
    )
    parser.add_argument(
        "--train-size", type=_train_size, default=_UNSET
    )
    parser.add_argument("--split-seed", type=int)
    parser.add_argument("--selection-seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--critic-steps", type=int)
    parser.add_argument("--generator-learning-rate", type=float)
    parser.add_argument("--critic-learning-rate", type=float)
    parser.add_argument("--device")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    arguments = parser.parse_args()

    config = load_config(
        arguments.config, _overrides(arguments)
    )
    data = config["data"]
    prepared = prepare_terminal_dataset(
        config["dataset"],
        cache_root=data.get("cache_root"),
        split_seed=int(data["split_seed"]),
        observables=data.get("observables"),
    )
    selection = make_preassigned_selection(
        prepared.split_codes,
        train_size=data.get("train_size"),
        split_seed=int(data["split_seed"]),
        selection_seed=int(data["selection_seed"]),
        group_ids=prepared.parameter_groups,
    )
    if arguments.run_dir is None:
        stamp = datetime.now(timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ"
        )
        observable_id = "-".join(
            prepared.schema.observable_names
        )
        run_directory = (
            Path("work/learning/gan-runs")
            / (
                f"{prepared.schema.database_id}-{observable_id}-"
                f"{stamp}"
            )
        )
    else:
        run_directory = arguments.run_dir
    result = train_experiment(
        prepared,
        selection,
        config,
        run_directory,
        resume=arguments.resume,
    )
    print(f"run: {result.run_directory}")
    return 0


def recipe_main(default_recipe=None):
    """One command accepts a leaf recipe or a sample dataset."""
    from learning.common.io import repository_root
    from .catalogue import create_leaf
    from .references import create_reference
    from .recipes import run_recipe
    from .data import prepare
    import yaml
    parser=argparse.ArgumentParser(description='Train a common conditional GAN from a recipe or dataset')
    parser.add_argument('--recipe',type=Path,default=default_recipe)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--root',type=Path)
    parser.add_argument('--output-root',type=Path)
    parser.add_argument('--cache-root',type=Path)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--prepare-only',action='store_true')
    parser.add_argument('--create-reference',action='store_true',
        help='Create missing native reference samples (requires CUDA compiler); changes holdouts if original archive is unavailable')
    parser.add_argument('--nvcc',default='nvcc')
    parser.add_argument('--architecture',help='CUDA SM for native references; default is current GPU')
    args=parser.parse_args()
    root=(args.root or repository_root(Path(__file__))).resolve()
    if args.dataset:
        if args.recipe is not None: parser.error('Choose --dataset or --recipe')
        args.recipe=create_leaf(root,args.dataset)
        args.create_reference=True
    if args.recipe is None: parser.error('--dataset or --recipe is required')
    recipe_path=args.recipe.resolve()
    recipe=yaml.safe_load(recipe_path.read_text())
    if args.create_reference:
        create_reference(root,recipe,nvcc=args.nvcc,architecture=args.architecture)
    if args.prepare_only:
        data=prepare(root,recipe,cache_root=args.cache_root)
        print('Prepared:',data.directory)
        return 0
    run_recipe(recipe_path,root,cache_root=args.cache_root,output_root=args.output_root,resume=args.resume)
    print('Completed:',recipe['output']['model'])
    return 0


if __name__ == "__main__":
    import sys
    # Keep explicit experimental configurations available; catalogue is the default.
    raise SystemExit(main() if '--config' in sys.argv else recipe_main())
