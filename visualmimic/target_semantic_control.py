"""Evaluate official MotionFix targets with the frozen visual judge.

No robot retargeting or policy execution is applied. The unchanged source is an
anonymous control candidate, not assumed to fail every preservation instruction.
"""
import argparse
import importlib.util
import json
from pathlib import Path


def frozen_judge(path):
    spec = importlib.util.spec_from_file_location('frozen_visual_judge', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare(root, output):
    import shutil
    original = root/'campaigns/ab_semantic_full_v1'
    j = frozen_judge(original/'judge.py')
    tasks = json.loads((original/'tasks.json').read_text())
    selected = []
    for row in tasks:
        result = json.loads((original/'items'/(row['key']['sample_id']+'.json')).read_text())
        assert result['key'] == row['key']
        if result['status'] == 'assessed':
            selected.append(row)
    assert len(selected) == 300
    labels = {json.dumps(r['key'], sort_keys=True): r for r in
              json.loads((root/'datasets/motionfix_test_v1/labels/manifest.json').read_text())}
    output.mkdir(exist_ok=False)
    shutil.copy2(original/'judge.py', output/'judge.py')
    shutil.copy2(original/'model.json', output/'model.json')
    j.dump(output/'tasks.json', selected)
    for index in range(8):
        rows = []
        for row in selected[index::8]:
            label = labels[json.dumps(row['key'], sort_keys=True)]
            rows.append({'key': row['key'], 'instruction': row['instruction'], 'source': label['target']})
        j.dump(output/'shards'/f'{index:03d}'/'target_manifest.json', rows)
    j.dump(output/'protocol.json', {
        'expected': len(selected), 'selection': 'same 300 assessed A/B pairs; original 7 technical failures excluded per user',
        'candidate_A': 'original MotionFix target human motion, rendered only',
        'candidate_B': 'unchanged original source human motion control',
        'blind': 'anonymous X/Y, both orders; target identity not passed to judge',
        'scope': 'judge acceptance of dataset targets, not true annotated success rate',
        'prompt_sha256': j.digest(j.PROMPT.encode()),
        'judge_sha256': j.digest((output/'judge.py').read_bytes()),
        'tasks_sha256': j.digest((output/'tasks.json').read_bytes()),
        'original_protocol': str(original/'protocol.json'),
        'render': 'same human renderer and configuration as source; full target duration',
        'sampling': 'same 32-frame maximum with actual timestamps',
        'unknowns': 'retained in denominator; no forced verdicts'})


def run(output, index):
    j = frozen_judge(output/'judge.py')
    protocol = json.loads((output/'protocol.json').read_text())
    assert j.digest((output/'judge.py').read_bytes()) == protocol['judge_sha256']
    assert j.digest((output/'tasks.json').read_bytes()) == protocol['tasks_sha256']
    tasks = json.loads((output/'tasks.json').read_text())[index::8]
    manifest = output/'shards'/f'{index:03d}'/'render/manifest.json'
    rendered = {json.dumps(r['key'], sort_keys=True): r for r in json.loads(manifest.read_text())} if manifest.exists() else {}
    model, processor = j.load_model(json.loads((output/'model.json').read_text())['path'])
    for row in tasks:
        path = output/'items'/(row['key']['sample_id']+'.json')
        if path.exists():
            raise FileExistsError('Refuse silent replay: '+str(path))
        j.dump(path, {'key': row['key'], 'status': 'running'})
        try:
            target = rendered[json.dumps(row['key'], sort_keys=True)]
            assert target['instruction'] == row['instruction']
            src = row['source_video']
            source_video = (j.clip(src['video'], src['timestamps_s']),
                            {'duration_s': src['timestamps_s'][-1], 'body': 'human source'})
            target_video = (j.clip(target['video'], target['timestamps_s']),
                            {'duration_s': target['timestamps_s'][-1], 'body': 'human'})
            control_video = (source_video[0], {'duration_s': src['timestamps_s'][-1], 'body': 'human'})
            raws = []
            for candidates in [(target_video, control_video), (control_video, target_video)]:
                raws.append(j.infer(model, processor, row['instruction'], (source_video, *candidates)))
                j.dump(output/'raw'/path.name, {'key': row['key'], 'outputs': raws})
            parsed = [j.parse(r) for r in raws]
            verdicts = j.consensus(*parsed)
            j.dump(path, {'key': row['key'], 'instruction': row['instruction'], 'status': 'assessed',
                          'verdicts': verdicts, 'passes': parsed})
            print(row['key']['sample_id'], verdicts, flush=True)
        except Exception as exc:
            j.dump(path, {'key': row['key'], 'instruction': row['instruction'], 'status': 'error', 'error': str(exc)})
            print(row['key']['sample_id'], str(exc), flush=True)
            if 'out of memory' in str(exc).lower():
                raise


def summary(output):
    j = frozen_judge(output/'judge.py')
    j.summarize(output)
    report = json.loads((output/'report.json').read_text())
    report.pop('calibration', None)
    report['scope'] = 'original human MotionFix target and unchanged source, same 300 samples, frozen automatic judge'
    report['group_mapping'] = {'A': 'original_target', 'B': 'unchanged_source'}
    report['target_pass_rate_all_expected'] = report['verdict_counts']['A'].get('pass', 0)/report['expected']
    j.dump(output/'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['prepare','run','summary'])
    p.add_argument('--root', type=Path); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--index', type=int, default=0); a=p.parse_args()
    if a.stage == 'prepare': prepare(a.root, a.output)
    elif a.stage == 'run': run(a.output, a.index)
    else: summary(a.output)
