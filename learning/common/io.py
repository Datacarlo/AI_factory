"""I/O and collection paths shared by neural-network catalogues."""
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


def collection_path(root: Path, relative: str, collection: str, *, output=False, output_root=None) -> Path:
    path = Path(relative)
    if path.is_absolute() or '..' in path.parts or path.parts[:1] != (collection,) or len(path.parts) < 2:
        raise ValueError(f'Expected a logical {collection}/... path')
    suffix = Path(*path.parts[1:])
    if output_root is not None:
        return Path(output_root) / path
    if output:
        return root / collection / 'unpublished' / suffix
    if path.parts[1] not in ('published', 'unpublished'):
        for state in ('unpublished', 'published'):
            candidate = root / collection / state / suffix
            if candidate.is_file():
                return candidate
    return root / path


def repository_root(path: Path) -> Path:
    for parent in (path.resolve(), *path.resolve().parents):
        if (parent / 'learning').is_dir() and (parent / 'catalog').is_dir():
            return parent
    raise ValueError('Cannot locate repository above ' + str(path))
