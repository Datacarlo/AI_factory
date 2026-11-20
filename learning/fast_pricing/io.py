"""Small, shared I/O helpers for recipes, checkpoints and preparation caches."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import yaml


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True))
    os.replace(temporary, path)


def repository_root(path: Path) -> Path:
    for parent in (path.resolve(), *path.resolve().parents):
        if (parent / 'learning/fast_pricing').is_dir():
            return parent
    raise ValueError('Cannot locate learning/fast_pricing above ' + str(path))


def dataset_path(root: Path, relative: str) -> Path:
    """Prefer a newly validated delivery, then the published collection."""
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError('Expected a repository-relative dataset path')
    if path.parts[:1] == ('datasets',) and path.parts[1] not in ('published', 'unpublished'):
        for state in ('unpublished', 'published'):
            candidate = root / 'datasets' / state / Path(*path.parts[1:])
            if candidate.is_file():
                return candidate
    return root / path


def pricer_output_path(root: Path, relative: str, *, output_root: Path | None = None) -> Path:
    """Route new models to the local collection, or an explicit campaign directory."""
    path = Path(relative)
    if (path.is_absolute() or '..' in path.parts
            or path.parts[:1] != ('fast_pricers',) or len(path.parts) < 2):
        raise ValueError('Expected a logical fast_pricers/... output path')
    if output_root is not None:
        return output_root / path
    return root / 'fast_pricers/unpublished' / Path(*path.parts[1:])
