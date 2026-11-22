"""Fast-pricer collection paths; shared I/O lives in learning.common."""
from pathlib import Path
from learning.common.io import sha256, fingerprint, write_json, write_yaml


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
