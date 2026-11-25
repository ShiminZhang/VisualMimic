"""Prepare an isolated full-cohort extension while preserving the pilot evaluator."""
import ast
import hashlib
import json
from pathlib import Path
import shutil

OUT = Path(__file__).resolve().parent
PILOT = OUT.parent / 'visualmimic_r2_20261124'
source = (PILOT / 'experiment.py').read_text()
pilot_protocol = json.loads((PILOT / 'protocol.json').read_text())
assert hashlib.sha256(source.encode()).hexdigest() == pilot_protocol['script_sha256']


def replace_once(text, old, new):
    assert text.count(old) == 1, (old[:100], text.count(old))
    return text.replace(old, new, 1)


text = replace_once(source, "SEED = 20261124", "SEED = 20261124\nPILOT = ROOT.parent / 'visualmimic_r2_20261124'\nCOHORT_SIZE = 100")
text = replace_once(text, 'positions = sorted(random.Random(SEED).sample(range(100), 20))', 'positions = list(range(COHORT_SIZE))')
text = replace_once(text,
    "'selection': 'Uniform sample of 20 distinct cohort positions using Python random.Random(seed).sample; no outcome-based selection',",
    "'selection': 'All 100 cases in the original judging cohort; full-cohort endpoint fixed before additional scoring; extension follows the inspected 20-case pilot',")
text = replace_once(text, "'conditions': CONDITIONS, 'requests': 120, 'judge': 'gpt-6-luna',", "'conditions': CONDITIONS, 'requests': 600, 'new_requests': 480, 'judge': 'gpt-6-luna',")
text = replace_once(text, "'analysis': 'Fixed denominator 20;", "'analysis': 'Fixed denominator 100;")
text = replace_once(text, "'limits': 'Small selected-cohort VLM study;", "'limits': 'Selected completed-rollout cohort VLM study;")
text = replace_once(text, "    save(OUT / 'inputs.json', selected)", "    retain_pilot(selected, protocol)\n    save(OUT / 'inputs.json', selected)")
text = replace_once(text, "{'cases': 20, 'sample_ids': protocol['sample_ids'], 'requests': 120}", "{'cases': COHORT_SIZE, 'sample_ids': protocol['sample_ids'], 'retained_requests': 120, 'new_requests': 480, 'requests': 600}")

helper = '''
def retain_pilot(selected, protocol):
    prior_protocol = json.loads((PILOT / 'protocol.json').read_text())
    prior_inputs = json.loads((PILOT / 'inputs.json').read_text())
    prior_results = json.loads((PILOT / 'results.json').read_text())
    ids = set(prior_protocol['sample_ids'])
    assert len(ids) == 20 and len(prior_results) == 120
    assert [r for r in selected if r['key']['sample_id'] in ids] == prior_inputs
    fields = ('conditions', 'judge', 'prompt', 'reasoning', 'max_output_tokens',
              'sampling', 'human_render', 'robot_render', 'source', 'judge_blinding', 'outcomes')
    for field in fields:
        assert protocol[field] == prior_protocol[field] or tuple(protocol[field]) == tuple(prior_protocol[field]), field
    assert prior_protocol['script_sha256'] == hashlib.sha256((PILOT / 'experiment.py').read_bytes()).hexdigest()
    assert {(r['key']['sample_id'], r['condition']) for r in prior_results} == {(sid, c) for sid in ids for c in CONDITIONS}
    assert all(r['status'] == 'assessed' for r in prior_results)
    snapshot = []
    paths = [PILOT / n for n in ['experiment.py', 'protocol.json', 'inputs.json', 'results.json', 'report.json', 'verification.json']]
    for sid in sorted(ids):
        paths.extend(p for p in (PILOT / 'videos' / sid).rglob('*') if p.is_file())
        for condition in CONDITIONS:
            paths.extend(p for p in (PILOT / 'judgments' / (sid + '_' + condition)).rglob('*') if p.is_file())
    for path in sorted(set(paths)):
        snapshot.append({'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    save(OUT / 'retained_pilot_manifest.json', snapshot)
    for directory in ('videos', 'judgments'):
        (OUT / directory).mkdir(exist_ok=False)
    for sid in sorted(ids):
        (OUT / 'videos' / sid).symlink_to(PILOT / 'videos' / sid, target_is_directory=True)
        for condition in CONDITIONS:
            name = sid + '_' + condition
            (OUT / 'judgments' / name).symlink_to(PILOT / 'judgments' / name, target_is_directory=True)
    protocol['extension'] = {
        'pilot': str(PILOT), 'retained_sample_ids': sorted(ids), 'new_cases': 80,
        'pilot_protocol_sha256': hashlib.sha256((PILOT / 'protocol.json').read_bytes()).hexdigest(),
        'pilot_results_sha256': hashlib.sha256((PILOT / 'results.json').read_bytes()).hexdigest(),
        'pilot_file_manifest_sha256': hashlib.sha256((OUT / 'retained_pilot_manifest.json').read_bytes()).hexdigest(),
        'reuse': 'All 120 pilot verdicts and their exact videos, requests and responses are reused without modification or reselection',
        'scope': 'No new human ratings, no new policy execution, no repository or document edits',
    }


def retained_render(row, conditions):
    protocol = json.loads((OUT / 'protocol.json').read_text())
    sid = row['key']['sample_id']
    if sid not in protocol['extension']['retained_sample_ids']:
        return False
    for condition in conditions:
        metadata = json.loads((OUT / 'videos' / sid / (condition + '.json')).read_text())
        checked(metadata['video'])
    print('REUSED_RENDER', sid, ','.join(conditions), flush=True)
    return True


'''
text = replace_once(text, '\ndef human():\n', '\n' + helper + 'def human():\n')
text = replace_once(text, "    for row in rows():\n        sid = row['key']['sample_id']\n        vertices, timestamps, shifts = {}, {}, {}", "    for row in rows():\n        if retained_render(row, ('A_human', 'B_human')):\n            continue\n        sid = row['key']['sample_id']\n        vertices, timestamps, shifts = {}, {}, {}")
text = replace_once(text, "    for row in rows():\n        sid = row['key']['sample_id']\n        spec = mujoco.MjSpec.from_file(str(xml))", "    for row in rows():\n        if retained_render(row, ('A_reference', 'B_reference', 'A_execution', 'B_execution')):\n            continue\n        sid = row['key']['sample_id']\n        spec = mujoco.MjSpec.from_file(str(xml))")
text = replace_once(text, "'expected': 120,", "'expected': len(tasks),")
text = replace_once(text, "== 120\n    index =", "== COHORT_SIZE * len(CONDITIONS)\n    index =")
ast.parse(text)

# Verify that scoring and render computations are unchanged after removing reuse guards.
before = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
after = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
assert ast.dump(before['frames']) == ast.dump(after['frames'])
for name in ('human', 'robot'):
    node = after[name]
    for child in ast.walk(node):
        if isinstance(child, ast.For) and child.body and isinstance(child.body[0], ast.If):
            condition = child.body[0].test
            if isinstance(condition, ast.Call) and isinstance(condition.func, ast.Name) and condition.func.id == 'retained_render':
                child.body.pop(0)
    assert ast.dump(before[name]) == ast.dump(node), name
old_evaluate = next(n for n in before['judge'].body if isinstance(n, ast.FunctionDef))
new_evaluate = next(n for n in after['judge'].body if isinstance(n, ast.FunctionDef))
assert ast.dump(old_evaluate) == ast.dump(new_evaluate)
(OUT / 'experiment.py').write_text(text)
shutil.copy2(PILOT / 'matched_settings.json', OUT / 'matched_settings.json')
shutil.copy2(PILOT / 'check_renders.py', OUT / 'check_renders.py')
for name in ['render.sbatch', 'judge.sbatch', 'qa.sbatch']:
    script = (PILOT / name).read_text().replace('visualmimic_r2_20261124', 'visualmimic_r2_full100_20261124').replace('vm-stage-', 'vm-full100-')
    (OUT / name).write_text(script)
verification = (PILOT / 'verify_results.py').read_text()
verification = replace_once(verification, "    sid, condition = r['key']['sample_id'], r['condition']", "    sid, condition = r['key']['sample_id'], r['condition']\n    assert r['key'] == source_rows[sid]['key']\n    assert r['instruction'] == source_rows[sid]['instruction']")
verification = verification.replace('== 120', '== 600').replace('== 20', '== 100')
verification = verification.replace('== 20 -', '== 100 -').replace(' - 5 * (', ' - (')
verification = verification.replace("'requests_verified': 120", "'requests_verified': 600")
verification = replace_once(verification, "(root / 'verification.json').write_text", "snapshot = json.loads((root / 'retained_pilot_manifest.json').read_text())\nfor item in snapshot:\n    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']\nverification['retained_pilot_files_unchanged'] = len(snapshot)\nverification['new_judgments'] = 480\nverification['retained_judgments'] = 120\n(root / 'verification.json').write_text")
ast.parse(verification)
(OUT / 'verify_results.py').write_text(verification)
save = {'render_computations_unchanged': True, 'frame_sampling_unchanged': True,
        'judge_request_and_response_code_unchanged': True, 'pilot_script_sha256': pilot_protocol['script_sha256'],
        'extension_script_sha256': hashlib.sha256(text.encode()).hexdigest()}
(OUT / 'implementation_conformance.json').write_text(json.dumps(save, indent=2))
print(json.dumps(save, indent=2))
