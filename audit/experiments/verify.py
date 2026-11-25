"""Verify the released experiment records using standard-library Python."""
from collections import Counter
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parent


def read(path):
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return json.loads(raw)


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def check_stage(directory, expected):
    root = ROOT / directory
    data = read(root / 'results.json.gz')
    protocol, report = read(root / 'protocol.json'), read(root / 'report.json')
    validation = read(root / 'verification.json')
    assert digest(gzip.decompress((root / 'results.json.gz').read_bytes())) == validation['results_sha256']
    assert digest((root / 'report.json').read_bytes()) == validation['report_sha256']
    assert digest((root / 'experiment.py').read_bytes()) == protocol['script_sha256']
    index = {(r['key']['sample_id'], r['condition']): r for r in data}
    assert len(index) == len(data) == expected * 6
    inputs = read(root / 'inputs.json.gz')
    ids = [r['key']['sample_id'] for r in inputs]
    assert len(set(ids)) == expected
    assert set(index) == {(sid, c) for sid in ids for c in protocol['conditions']}
    groups = {c: dict(Counter(index[(sid, c)]['verdict'] for sid in ids)) for c in protocol['conditions']}
    assert groups == report['groups']
    for stage in ('human', 'reference', 'execution'):
        pairs = [(index[(sid, 'A_' + stage)]['verdict'], index[(sid, 'B_' + stage)]['verdict']) for sid in ids]
        only_a = sum(a == 'success' and b != 'success' for a, b in pairs)
        only_b = sum(b == 'success' and a != 'success' for a, b in pairs)
        p = report['paired'][stage]
        assert (p['A_only_success'], p['B_only_success']) == (only_a, only_b)
        differences = [int(b == 'success') - int(a == 'success') for a, b in pairs]
        rng = random.Random(protocol['seed'])
        bootstrap = sorted(sum(rng.choices(differences, k=expected)) * 100 / expected for _ in range(10000))
        assert all(abs(a - b) < 1e-10 for a, b in zip([bootstrap[249], bootstrap[9749]], p['paired_bootstrap_95_percent_interval']))
        n = only_a + only_b
        exact = min(1., 2 * sum(math.comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2**n) if n else 1.
        assert abs(exact - p['exact_paired_binary_p']) < 1e-12
    responses = read(root / 'response_records.json.gz')
    requests = read(root / 'request_records.json.gz')
    videos = read(root / 'video_records.json.gz')
    for records, key in [(responses, lambda r: (r['key']['sample_id'], r['condition'])),
                         (requests, lambda r: (r['metadata']['key']['sample_id'], r['metadata']['condition'])),
                         (videos, lambda r: (r['sample_id'], r['condition']))]:
        assert len(records) == len(index) and {key(r) for r in records} == set(index)
    for r in responses:
        original = index[(r['key']['sample_id'], r['condition'])]
        assert r['source_response_sha256'] == original['response_sha256']
        assert r['output_text'] == original['output_text']
        verdict = json.loads(r['output_text'])
        assert verdict['verdict'] == original['verdict'] and verdict['reason'] == original['reason']
    for r in requests:
        meta = r['metadata']; original = index[(meta['key']['sample_id'], meta['condition'])]
        for field in ('key', 'instruction', 'condition', 'source_video_sha256', 'candidate_video_sha256', 'source_frame_indices', 'candidate_frame_indices'):
            assert meta[field] == original[field]
        assert r['original_payload_sha256'] == meta['request_sha256'] == original['request_sha256']
        payload = r['payload_without_image_bytes']
        assert payload['model'] == protocol['judge']
        content = payload['input'][0]['content']
        assert content[0]['text'] == protocol['prompt'] + '\nINSTRUCTION: ' + original['instruction']
        assert sum(c['type'] == 'input_image' for c in content) == len(meta['source_frame_indices']) + len(meta['candidate_frame_indices'])
        assert all('image_url' not in c for c in content)
    for r in videos:
        assert digest(r['source_text'].encode()) == r['metadata_sha256']
        assert json.loads(r['source_text'])['video']['sha256'] == index[(r['sample_id'], r['condition'])]['candidate_video_sha256']
    for branch in ('A', 'B'):
        for before, after in [('human', 'reference'), ('reference', 'execution')]:
            counts = dict(Counter(index[(sid, branch + '_' + before)]['verdict'] + ' -> ' + index[(sid, branch + '_' + after)]['verdict'] for sid in ids))
            assert counts == report['transitions'][branch + '_' + before + '_to_' + after]
    with (root / 'case_results.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == expected
    assert {row['sample_id'] for row in rows} == set(ids)
    for row in rows:
        for condition in protocol['conditions']:
            assert row[condition] == index[(row['sample_id'], condition)]['verdict']
    return index


def main():
    manifest = read(ROOT / 'manifest.json')
    for r in manifest['files']:
        raw = (ROOT / r['path']).read_bytes()
        assert len(raw) == r['bytes'] and digest(raw) == r['sha256'], r['path']
    pilot = check_stage('r2_pilot', 20)
    full = check_stage('r2_full100', 100)
    assert all(full[k] == v for k, v in pilot.items())
    original = read(ROOT.parent / 'evidence/c2_c4.json')
    records = read(ROOT / 'c2_c4/source_records.json.gz')
    assert len({r['source_path'] for r in records}) == len(records)
    source_hashes = set()
    for r in records:
        assert digest(r['source_text'].encode()) == r['sha256']
        json.loads(r['source_text'])
        source_hashes.add(r['sha256'])
    assert set(original['source_reports'].values()) <= source_hashes
    for item in original['c2']['items']:
        for branch in ('A', 'B'):
            for field in ['execution_result_sha256', 'execution_metadata_sha256', 'execution_run_sha256', 'retarget_run_sha256', 'retarget_report_sha256', 'editor_run_sha256', 'editor_item_sha256']:
                assert item[branch][field] in source_hashes
    responses = read(ROOT / 'c2_c4/response_records.json.gz')
    response_index = {(r['key']['sample_id'], r['branch']): r for r in responses}
    assert len(response_index) == len(responses) == 400
    for r in original['c4']:
        response = response_index[(r['key']['sample_id'], r['branch'])]
        assert response['source_response_sha256'] == r['response_sha256']
        assert response['output_text'] == r['output_text']
        assert r['request_record_sha256'] in source_hashes and r['result_sha256'] in source_hashes
    print(json.dumps({'status': 'passed', 'package_files_verified': len(manifest['files']), 'technical_source_records': len(records), 'technical_matched_pairs': len(original['c2']['items']), 'technical_judgments': len(responses), 'r2_pilot_judgments': len(pilot), 'r2_full_cohort_judgments': len(full), 'pilot_judgments_reused_unchanged': len(pilot), 'paired_statistics_reproduced': True, 'limits': 'Public records omit image bytes and API envelopes. Their original hashes were verified on ICE before export; this command verifies exported records and counts, not model accuracy.'}, indent=2))


if __name__ == '__main__':
    main()
