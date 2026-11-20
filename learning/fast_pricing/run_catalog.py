"""Run a catalogue sequentially, keeping each dataset resident for its variants."""
from __future__ import annotations
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .io import fingerprint, repository_root, sha256, write_json


def completed(recipe_path, output_root, dataset_fingerprint):
    import yaml
    recipe = yaml.safe_load(recipe_path.read_text())
    marker = output_root/'runs'/recipe['id']/'published.json'
    if not marker.exists():
        return False
    saved = json.loads(marker.read_text())
    return (saved['recipe_sha256'] == sha256(recipe_path)
            and saved.get('dataset_fingerprint') == dataset_fingerprint
            and recipe_path.with_name('generation.yaml').exists()
            and all((output_root/p).is_file() and sha256(output_root/p) == h
                    for p,h in saved['outputs'].items()))


def train_dataset(root, record, ready_path, output_root):
    import torch
    import yaml
    from .batches import ResidentDataset
    from .data import open_cache
    from .recipes import run_recipe
    torch.set_num_threads(2)
    ready = json.loads(ready_path.read_text())
    if ready['id'] != record['id'] or ready['logical_path'] != record['logical_path']:
        raise ValueError('Cache delivery identity mismatch')
    prepared = open_cache(root/ready['cache'])
    if prepared.fingerprint != ready['fingerprint'] or prepared.manifest['identity']['dataset_sha256'] != ready['source_sha256']:
        raise ValueError('Cache delivery fingerprint mismatch')
    resident = ResidentDataset(prepared, torch.device('cuda'))
    paired = {}
    for relative in record['recipes']:
        path = root/relative
        recipe = yaml.safe_load(path.read_text())
        run_directory = output_root/'runs'/recipe['id']
        write_json(output_root/'current_training.json', dict(dataset=record['id'], recipe=relative, run=str(run_directory)))
        if not completed(path, output_root, prepared.fingerprint):
            # Only pre-checkpoint state may be restarted; completed results are never overwritten silently.
            if (run_directory/'initial_model.pt').exists() and not (run_directory/'checkpoint.pt').exists():
                archived = run_directory.with_name(run_directory.name + '.interrupted_' + str(time.time_ns()))
                run_directory.rename(archived)
            result = run_recipe(path, root, prepared=prepared, resident=resident,
                                output_root=output_root, resume=(run_directory/'checkpoint.pt').exists())
            write_json(run_directory/'published.json', dict(recipe_sha256=sha256(path), dataset_fingerprint=prepared.fingerprint,
                outputs={p:sha256(output_root/p) for p in recipe['output'].values()}))
        metrics = json.loads((output_root/recipe['output']['metrics']).read_text())
        signature = (metrics['initial_model_hash'], metrics['selection'])
        architecture = tuple(recipe['network']['hidden_sizes'])
        if architecture in paired and signature != paired[architecture]:
            raise ValueError('Price/Sobolev initialization or partition differs')
        paired[architecture] = signature
        print('completed ' + recipe['id'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=repository_root(Path(__file__)))
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--ready', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    records = manifest['datasets'] if isinstance(manifest, dict) else manifest
    if args.worker is not None:
        record = records[args.worker]
        train_dataset(args.root, record, args.ready/(record['id']+'.json'), args.output_root)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    with (args.output_root/'campaign.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = fingerprint(records)
        count = 0
        for i,record in enumerate(records):
            state = dict(status='waiting_for_data', pid=os.getpid(), dataset=record['id'],
                         dataset_index=i, dataset_count=len(records), completed_runs=count,
                         total_runs=sum(len(r['recipes']) for r in records), manifest_fingerprint=identity)
            write_json(args.output_root/'campaign.json', state)
            while not (args.ready/(record['id']+'.json')).exists():
                if (args.ready/'preparation_failed.json').exists():
                    raise RuntimeError('Dataset preparation failed; inspect preparation_failed.json')
                time.sleep(15)
            state['status'] = 'training'
            write_json(args.output_root/'campaign.json', state)
            log = args.output_root/'logs'/(record['id']+'.log')
            log.parent.mkdir(exist_ok=True)
            command = [sys.executable,'-m','learning.fast_pricing.run_catalog','--root',str(args.root),
                       '--manifest',str(args.manifest),'--ready',str(args.ready),
                       '--output-root',str(args.output_root),'--worker',str(i)]
            with log.open('a') as stream:
                result = subprocess.run(command, cwd=args.root, stdout=stream, stderr=subprocess.STDOUT)
            if result.returncode:
                state.update(status='failed', returncode=result.returncode, log=str(log))
                write_json(args.output_root/'campaign.json', state)
                raise RuntimeError('Training worker failed: '+str(log))
            count += len(record['recipes'])
        state.update(status='complete', completed_runs=count, dataset_index=len(records))
        write_json(args.output_root/'campaign.json', state)


if __name__ == '__main__':
    main()
