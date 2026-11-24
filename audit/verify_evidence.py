"""Recheck the portable C2-C4 evidence against the released audit tables."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit.evaluate import load_tables, STAGES, VERDICTS
from visualmimic.render_provenance import digest_json, key_of, unique_index


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(bundle, verdicts, retrieval):
    require(bundle['schema_version'] == 1, 'Unsupported evidence schema')
    require(bundle['source_reports'] == {
        'judgments': '10ad60bd0f99f653260b2e4c86cfcd94c033c99b48e22f914de4120ff4a6835a',
        'retrieval': 'eb287b227dc7dd8af9a4a5c8855fd6417ff81cd945856ef774f6d6a75ae68b64'}, 'Historical report hashes differ')
    for checksum, protocol in bundle['protocols'].items():
        require(hashlib.sha256(protocol['source_text'].encode()).hexdigest() == checksum, 'Protocol checksum mismatch')
        require(json.loads(protocol['source_text']) == protocol['content'], 'Protocol content mismatch')
    configs = bundle['c2']['configurations']
    for identity, config in configs.items():
        require(digest_json(config) == identity, 'Settings checksum mismatch')
    c2 = unique_index(bundle['c2']['items'])
    require(len(c2) == 307, 'Expected 307 C2 cases')
    require({row['key']['sample_id'] for row in c2.values()} == {row['sample_id'] for row in retrieval}, 'C2/retrieval cohort mismatch')
    for row in c2.values():
        a, b = row['A']['settings_sha256'], row['B']['settings_sha256']
        require(a in configs and b in configs and a == b, 'A/B settings mismatch')
    c3 = unique_index(bundle['c3'])
    require(len(c3) == 100 and c3.keys() == {key for key, row in c2.items() if row['judged']}, 'C3/judged cohort mismatch')
    for key, row in c3.items():
        require(row['source_sha256'] == row['A_edited_motion_sha256'] == c2[key]['A']['edited_motion_sha256'], 'Pre-GMR source mismatch')
        require(row['canonical_source_sha256'] == row['rendered_source_sha256'], 'Rendered source mismatch')
        require(row['video_sha256'] == row['prepared_video_sha256'] == row['candidate_videos']['pre_GMR'], 'Pre-GMR video mismatch')
        require(row['A_reference_sha256'] == row['C_reference_sha256'], 'C reference mismatch')
    exported = {row['sample_id']: row for row in verdicts}
    require({row['key']['sample_id'] for row in c3.values()} == exported.keys(), 'C3/CSV cohort mismatch')
    judgments, counts, instructions = set(), {g: Counter() for g in STAGES}, {}
    for row in bundle['c4']:
        key, branch = key_of(row), row['branch']
        require(key in c3 and branch in STAGES, 'Unexpected judgment sample/branch')
        require((key, branch) not in judgments, 'Duplicate judgment')
        judgments.add((key, branch))
        require(row['verdict'] in VERDICTS, 'Invalid evidence verdict')
        require(row['verdict'] == json.loads(row['output_text'])['verdict'], 'Response/verdict mismatch')
        require(row['verdict'] == exported[row['key']['sample_id']][branch], 'Response/CSV mismatch')
        require(row['protocol_sha256'] in bundle['protocols'], 'Missing judge protocol')
        require(row['source_video_sha256'] == c3[key]['source_video_sha256'], 'Judge source mismatch')
        require(row['candidate_video_sha256'] == c3[key]['candidate_videos'][branch], 'Judge candidate mismatch')
        for field in ('source_frame_indices', 'candidate_frame_indices'):
            ids = row[field]
            require(ids and all(type(v) is int and v >= 0 for v in ids) and ids == sorted(set(ids)) and ids[0] == 0, 'Invalid frame indices')
        if key in instructions:
            require(instructions[key] == row['instruction'], 'Judgment instructions differ')
        instructions[key] = row['instruction']
        counts[branch][row['verdict']] += 1
    require(judgments == {(key, branch) for key in c3 for branch in STAGES}, 'Incomplete judgment cohort')
    return {'C2_matched_pairs': len(c2), 'C2_unique_settings': len(configs),
            'C3_verified_links': len(c3), 'C4_judgments': len(judgments),
            'verdicts': {g: {v: counts[g][v] for v in VERDICTS} for g in STAGES}}


def main():
    root = Path(__file__).parent
    manifest = json.loads((root / 'evidence/manifest.json').read_text())
    data = (root / 'evidence/c2_c4.json').read_bytes()
    require(hashlib.sha256(data).hexdigest() == manifest['evidence_sha256'], 'Evidence file checksum mismatch')
    bundle = json.loads(data)
    print(json.dumps(verify(bundle, *load_tables(root)), indent=2))


if __name__ == '__main__':
    main()
