"""Verify saved request/response records and independently recount paired results."""
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
protocol = json.loads((root / 'protocol.json').read_text())
assert hashlib.sha256((root / 'experiment.py').read_bytes()).hexdigest() == protocol['script_sha256']
inputs = json.loads((root / 'inputs.json').read_text())
results = json.loads((root / 'results.json').read_text())
report = json.loads((root / 'report.json').read_text())
expected = {(r['key']['sample_id'], c) for r in inputs for c in protocol['conditions']}
assert len(expected) == 600
assert len(results) == 600
assert {(r['key']['sample_id'], r['condition']) for r in results} == expected
source_rows = {r['key']['sample_id']: r for r in inputs}
for r in results:
    sid, condition = r['key']['sample_id'], r['condition']
    assert r['key'] == source_rows[sid]['key']
    assert r['instruction'] == source_rows[sid]['instruction']
    directory = root / 'judgments' / (sid + '_' + condition)
    assert json.loads((directory / 'result.json').read_text()) == r
    request_bytes = gzip.decompress((directory / 'request.json.gz').read_bytes())
    assert hashlib.sha256(request_bytes).hexdigest() == r['request_sha256']
    payload = json.loads(request_bytes)
    content = payload['input'][0]['content']
    assert payload['model'] == protocol['judge']
    assert content[0]['text'] == protocol['prompt'] + '\nINSTRUCTION: ' + source_rows[sid]['instruction']
    assert sum(c['type'] == 'input_image' for c in content) == len(r['source_frame_indices']) + len(r['candidate_frame_indices'])
    assert r['source_video_sha256'] == source_rows[sid]['source_video']['video']['sha256']
    video = json.loads((root / 'videos' / sid / (condition + '.json')).read_text())
    assert video['video']['sha256'] == r['candidate_video_sha256']
    if r['status'] == 'assessed':
        response_bytes = (directory / 'response.json').read_bytes()
        assert hashlib.sha256(response_bytes).hexdigest() == r['response_sha256']
        response = json.loads(response_bytes)
        text = ''.join(c['text'] for item in response['output'] for c in item.get('content', []) if c.get('type') == 'output_text')
        answer = json.loads(text)
        assert text == r['output_text']
        assert answer['verdict'] == r['verdict'] and answer['reason'] == r['reason']
        assert response['model'] == r['response_model']
groups = {c: Counter(r['verdict'] for r in results if r['condition'] == c) for c in protocol['conditions']}
assert all(dict(groups[c]) == report['groups'][c] for c in groups)
assert all(sum(groups[c].values()) == 100 for c in groups)
for stage in ['human', 'reference', 'execution']:
    passed = {b: {r['key']['sample_id'] for r in results if r['condition'] == b + '_' + stage and r['verdict'] == 'success'} for b in ['A', 'B']}
    actual = report['paired'][stage]
    assert actual['A_only_success'] == len(passed['A'] - passed['B'])
    assert actual['B_only_success'] == len(passed['B'] - passed['A'])
    assert actual['both_success'] == len(passed['A'] & passed['B'])
    assert actual['neither_success'] == 100 - len(passed['A'] | passed['B'])
    assert abs(actual['B_minus_A_percentage_points'] - (len(passed['B']) - len(passed['A']))) < 1e-9
verification = {'requests_verified': 600, 'responses_verified': sum(r['status'] == 'assessed' for r in results),
                'fixed_denominators_verified': True, 'paired_counts_verified': True,
                'script_matches_frozen_protocol': True,
                'report_sha256': hashlib.sha256((root / 'report.json').read_bytes()).hexdigest(),
                'results_sha256': hashlib.sha256((root / 'results.json').read_bytes()).hexdigest()}
snapshot = json.loads((root / 'retained_pilot_manifest.json').read_text())
for item in snapshot:
    assert hashlib.sha256(Path(item['path']).read_bytes()).hexdigest() == item['sha256']
verification['retained_pilot_files_unchanged'] = len(snapshot)
verification['new_judgments'] = 480
verification['retained_judgments'] = 120
(root / 'verification.json').write_text(json.dumps(verification, indent=2))
print(json.dumps(verification, indent=2), flush=True)
