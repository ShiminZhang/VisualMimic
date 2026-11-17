from dataclasses import asdict
import json
import numpy as np
import pytest
from visualmimic.contracts import MotionSequence
from visualmimic.human_motion import HUMAN_SPEC
from visualmimic.storage import ArtifactStore, digest
from visualmimic.batch_edit import atomic_json
from visualmimic.video_campaign import prepare, merge


def fixture(tmp_path):
    rows = [{'key': {'dataset_version': 'fixture', 'sample_id': str(i), 'split': 'test'},
             'instruction': 'walk', 'source': {}} for i in range(3)]
    manifest = tmp_path / 'input.json'; atomic_json(manifest, rows)
    root = tmp_path / 'campaign'
    prepare(manifest, root, 2, 3)
    return rows, root


def test_sharding_preserves_split_and_covers_each_sample_once(tmp_path):
    rows, root = fixture(tmp_path)
    a = json.loads((root / 'source/000.json').read_text())
    b = json.loads((root / 'source/001.json').read_text())
    assert a == rows[::2] and b == rows[1::2]
    assert json.loads((root / 'tasks.json').read_text()) == [
        {'key': row['key'], 'instruction': row['instruction']} for row in rows]


def test_merge_restores_order_and_retains_missing_denominator(tmp_path):
    rows, root = fixture(tmp_path)
    run = root / 'shards/000/reconstruction'; (run / 'items').mkdir(parents=True)
    store = ArtifactStore(run.with_name('reconstruction_artifacts'))
    motion = MotionSequence(HUMAN_SPEC, np.arange(2)/30,
        {'rots': np.zeros((2,22,3),np.float32), 'trans': np.zeros((2,3),np.float32),
         'joint_positions': np.zeros((2,22,3),np.float32)}, np.ones(2,bool))
    ref = asdict(store.put_motion(motion))
    spec = {'stage': 'reconstruct', 'backend': {'backend': 'fixture'}, 'rows': [
        {'key': r['key'], 'instruction': r['instruction'], 'video': {},
         'timestamps_s': [0,1/30], 'width': 640, 'height': 480} for r in rows[::2]]}
    identity = digest(spec)
    atomic_json(run / 'run.json', {'identity': identity, 'specification': spec})
    for row in reversed(rows[::2]):
        atomic_json(run / 'items' / (digest(row['key'])+'.json'), {
            'run_identity': identity, 'key': row['key'], 'status': 'succeeded',
            'result': {**row, 'source': ref}})
    result = merge(root)
    assert result['expected'] == 3 and result['available'] == 2 and result['status'] == 'incomplete'
    merged = json.loads((root / 'merged/source_manifest.json').read_text())
    assert [r['key'] for r in merged] == [r['key'] for r in rows[::2]]
    report = json.loads((root / 'merged/coverage.json').read_text())
    assert report['missing'][0]['key'] == rows[1]['key']


def test_plan_changes_are_rejected(tmp_path):
    rows, root = fixture(tmp_path)
    plan = json.loads((root / 'plan.json').read_text()); plan['shards'] = 1
    atomic_json(root / 'plan.json', plan)
    with pytest.raises(ValueError, match='plan changed'): merge(root)
