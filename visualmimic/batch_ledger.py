"""Shared sequential stage ledger; no silent replay of uncertain model calls."""
from dataclasses import asdict
import fcntl
import json
from pathlib import Path
from .batch_edit import atomic_json
from .contracts import SampleKey
from .storage import digest


def run_ledger(rows, stage, identity, output, process, verify, resume=False,
              retry_interrupted=False):
    keys = [SampleKey(**row['key']) for row in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('Expected nonempty unique sample keys')
    out = Path(output)
    out.mkdir(parents=True, exist_ok=resume)
    items = out / 'items'; items.mkdir(exist_ok=True)
    specification = {'version': 1, 'stage': stage, 'rows': rows, 'backend': identity}
    run_identity = digest(specification)
    with (out / 'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_file = out / 'run.json'
        if run_file.exists():
            if not resume or json.loads(run_file.read_text())['identity'] != run_identity:
                raise ValueError('Resume input/backend/runtime mismatch')
        else:
            atomic_json(run_file, {'identity': run_identity, 'specification': specification})
        results, failures = [], []
        for index, (row, key) in enumerate(zip(rows, keys)):
            item = items / (digest(key) + '.json')
            previous = json.loads(item.read_text()) if item.exists() else None
            if previous and (previous['run_identity'] != run_identity or previous['key'] != asdict(key)):
                raise ValueError('Item provenance mismatch')
            if previous and previous['status'] not in {'running', 'failed', 'succeeded'}:
                raise ValueError('Unknown item status; refusing replay')
            if previous and previous['status'] == 'succeeded':
                verify(previous['result'])
                results.append(previous['result'])
            elif previous and previous['status'] == 'failed':
                failures.append(previous)
            elif previous and previous['status'] == 'running' and not retry_interrupted:
                atomic_json(out / 'status.json', {'status': 'interrupted', 'key': key,
                    'succeeded': len(results), 'expected': len(rows)})
                raise RuntimeError('Uncertain previous call; explicitly retry interrupted work')
            else:
                record = {'key': asdict(key), 'run_identity': run_identity, 'status': 'running',
                          'attempt': previous['attempt'] + 1 if previous else 1}
                atomic_json(item, record)
                try:
                    result = process(row)
                    if result['key'] != row['key'] or result['instruction'] != row['instruction']:
                        raise ValueError('Stage changed sample identity or instruction')
                    verify(result)
                    record.update(status='succeeded', result=result)
                    results.append(result)
                except Exception as exc:
                    record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
                    failures.append(record)
                atomic_json(item, record)
            atomic_json(out / 'manifest.json', results)
            atomic_json(out / 'failures.json', failures)
            atomic_json(out / 'status.json', {'status': 'running', 'processed': index + 1,
                'expected': len(rows), 'succeeded': len(results), 'failed': len(failures)})
            print(f'{stage.upper()}_PROGRESS {index + 1}/{len(rows)} succeeded={len(results)} failed={len(failures)}', flush=True)
        atomic_json(out / 'status.json', {'status': 'completed' if not failures else 'completed_with_failures',
            'expected': len(rows), 'succeeded': len(results), 'failed': len(failures)})
    return results

