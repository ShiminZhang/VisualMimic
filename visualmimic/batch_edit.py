"""Run real editing over a source-only manifest with auditable resumability.

Execution is sequential so each sample retains the single-example seed protocol.
A live lock prevents concurrent writers. An interrupted model call is never silently
replayed: --retry-interrupted is required to repeat it explicitly.
"""
import argparse
from dataclasses import asdict
import fcntl
import json
import os
from pathlib import Path
import tempfile
from .contracts import ArtifactRef, SampleKey
from .storage import ArtifactStore, encode, digest
from .backends.motionfix_editor import MotionFixEditor


def atomic_json(path, value):
    data = encode(value)
    fd, temporary = tempfile.mkstemp(dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def run_batch(rows, store, editor, options, output, resume=False, retry_interrupted=False):
    out = Path(output)
    out.mkdir(parents=True,exist_ok=resume)
    items = out/'items';items.mkdir(exist_ok=True)
    keys = [SampleKey(**r['key']) for r in rows]
    if not rows or len(set(keys)) != len(keys):
        raise ValueError('Expected nonempty unique source keys')
    if any(set(r) != {'key','instruction','source'} for r in rows):
        raise ValueError('Only source-only manifest fields accepted')
    specification = {'rows':rows,'editor':editor.identity,'options':options,'execution':'sequential-per-example-seed'}
    identity = digest(specification)
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest = out/'run.json'
        if manifest.exists():
            if not resume or json.loads(manifest.read_text())['identity'] != identity:
                raise ValueError('Resume input/model/options mismatch')
        else:
            atomic_json(manifest,{'identity':identity,'specification':specification})
        predictions, failures = [], []
        for index,(row,key) in enumerate(zip(rows,keys)):
            item = items/(digest(key)+'.json')
            previous = json.loads(item.read_text()) if item.exists() else None
            if previous and (previous.get('run_identity') != identity or previous.get('key') != asdict(key)):
                raise ValueError('Item provenance mismatch')
            if previous and previous['status'] == 'succeeded':
                store.read_motion(ArtifactRef(**previous['motion']))
                predictions.append({'key':row['key'],'motion':previous['motion']})
            elif previous and previous['status'] == 'running' and not retry_interrupted:
                atomic_json(out/'status.json',{'status':'interrupted','sample':key,'completed':len(predictions),'expected':len(rows)})
                raise RuntimeError(f'Uncertain previous call for {key.sample_id}; explicitly retry interrupted work')
            elif previous and previous['status'] == 'failed':
                failures.append(previous)
            else:
                atomic_json(item,{'status':'running','key':key,'run_identity':identity})
                try:
                    source = store.read_motion(ArtifactRef(**row['source']))
                    motion = editor.edit(source,row['instruction'],options)
                    ref = store.put_motion(motion)
                    record = {'status':'succeeded','key':key,'motion':ref,'run_identity':identity}
                    atomic_json(item,record)
                    predictions.append({'key':row['key'],'motion':asdict(ref)})
                except Exception as exc:
                    record = {'status':'failed','key':asdict(key),'error_type':type(exc).__name__,
                              'error':str(exc),'run_identity':identity}
                    atomic_json(item,record); failures.append(record)
            atomic_json(out/'predictions.json',predictions)
            atomic_json(out/'failures.json',failures)
            atomic_json(out/'status.json',{'status':'running','processed':index+1,'expected':len(rows),
                        'succeeded':len(predictions),'failed':len(failures)})
            print(f'EDIT_PROGRESS {index+1}/{len(rows)} succeeded={len(predictions)} failed={len(failures)}',flush=True)
        atomic_json(out/'status.json',{'status':'completed' if not failures else 'completed_with_failures',
                    'expected':len(rows),'succeeded':len(predictions),'failed':len(failures)})
    return predictions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--source-artifacts',required=True)
    p.add_argument('--repository',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--body-models',required=True);p.add_argument('--clip-config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--device',default='cpu');p.add_argument('--resume',action='store_true')
    p.add_argument('--retry-interrupted',action='store_true')
    args=p.parse_args()
    out=Path(args.output).resolve()
    if out.exists() and not args.resume: raise FileExistsError(out)
    rows=json.loads(Path(args.manifest).read_text())
    # Keep artifacts next to the run because ArtifactStore creates its own directory.
    store=ArtifactStore(out.with_name(out.name+'_artifacts'),[args.source_artifacts])
    editor=MotionFixEditor(args.repository,args.checkpoint,args.body_models,args.clip_config,device=args.device)
    run_batch(rows,store,editor,editor.options(args.seed),out,args.resume,args.retry_interrupted)


if __name__ == '__main__': main()
