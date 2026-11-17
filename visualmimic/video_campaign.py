"""Prepare deterministic video shards and merge verified reconstruction outputs."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from .batch_edit import atomic_json
from .batch_video import validate_rows
from .contracts import ArtifactRef
from .storage import ArtifactStore, digest


def prepare(manifest, output, shards, expected_count):
    rows = json.loads(Path(manifest).read_text())
    validate_rows(rows, 'render')
    if len(rows) != expected_count or not 1 <= shards <= len(rows):
        raise ValueError('Unexpected dataset size or shard count')
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / 'source').mkdir()
    specification = {'rows': rows, 'shards': shards, 'partition': 'ordered-strided-v1'}
    atomic_json(root / 'plan.json', {'identity': digest(specification), **specification})
    atomic_json(root / 'tasks.json', [{'key': r['key'], 'instruction': r['instruction']} for r in rows])
    for index in range(shards):
        atomic_json(root / 'source' / f'{index:03d}.json', rows[index::shards])
    return {'expected': len(rows), 'shards': shards, 'identity': digest(specification)}


def merge(campaign):
    root = Path(campaign).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    specification = {k: plan[k] for k in ('rows', 'shards', 'partition')}
    if plan['identity'] != digest(specification):
        raise ValueError('Campaign plan changed')
    expected = plan['rows']
    validate_rows(expected, 'render')
    out = root / 'merged'
    out.mkdir(exist_ok=False)
    roots = [root / 'shards' / f'{i:03d}' / 'reconstruction_artifacts' for i in range(plan['shards'])]
    store = ArtifactStore(out / 'artifacts', roots)
    succeeded, missing, backends = {}, [], set()
    for index in range(plan['shards']):
        run = root / 'shards' / f'{index:03d}' / 'reconstruction'
        if (run / 'run.json').exists():
            record = json.loads((run / 'run.json').read_text())
            if record['identity'] != digest(record['specification']):
                raise ValueError('Shard run identity mismatch')
            spec = record['specification']
            if spec['stage'] != 'reconstruct':
                raise ValueError('Unexpected stage')
            validate_rows(spec['rows'], 'reconstruct')
            backends.add(digest(spec['backend']['backend']))
            run_identity = record['identity']
        else:
            run_identity = None
        for source in expected[index::plan['shards']]:
            key = source['key']
            item = run / 'items' / (digest(key) + '.json')
            if not item.exists():
                missing.append({'key': key, 'status': 'not_recorded'})
                continue
            record = json.loads(item.read_text())
            if run_identity is None or record['run_identity'] != run_identity or record['key'] != key:
                raise ValueError('Shard item provenance mismatch')
            if record['status'] != 'succeeded':
                missing.append({'key': key, 'status': record['status'], 'error': record.get('error')})
                continue
            result = record['result']
            validate_rows([result], 'render')
            if result['key'] != key or result['instruction'] != source['instruction']:
                raise ValueError('Reconstruction changed sample or instruction')
            ref = ArtifactRef(**result['source'])
            if not Path(ref.uri).resolve().is_relative_to(roots[index].resolve()):
                raise ValueError('Reconstruction output outside its shard artifact store')
            store.read_motion(ref)
            copied = store.put_bytes(store.read_bytes(ref), ref.media_type)
            succeeded[digest(key)] = {**result, 'source': asdict(copied)}
    if len(backends) > 1:
        raise ValueError('Reconstruction model/config changed across shards')
    rows = [succeeded[digest(row['key'])] for row in expected if digest(row['key']) in succeeded]
    atomic_json(out / 'source_manifest.json', rows)
    report = {'expected': len(expected), 'available': len(rows), 'missing': missing,
              'coverage': len(rows) / len(expected), 'plan_identity': plan['identity'],
              'reconstruction_backend_ids': sorted(backends),
              'status': 'complete' if not missing else 'incomplete',
              'evaluation': 'Always use the original complete fixed candidate set'}
    atomic_json(out / 'coverage.json', report)
    return {k: v for k, v in report.items() if k != 'missing'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--manifest', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--shards', type=int, default=64)
    prep.add_argument('--expected-count', type=int, default=1013)
    collect = sub.add_parser('merge')
    collect.add_argument('--campaign', required=True)
    args = parser.parse_args()
    result = prepare(args.manifest, args.output, args.shards, args.expected_count) if args.command == 'prepare' else merge(args.campaign)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
