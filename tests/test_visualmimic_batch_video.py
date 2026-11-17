import json
import pytest
from visualmimic.batch_video import run_stage, validate_rows


def rows():
    return [{'key': {'dataset_version': 'fixture', 'sample_id': 'one', 'split': 'test'},
             'instruction': 'walk', 'source': {'uri': 'fixture', 'sha256': 'fixture'}}]


def test_video_boundary_rejects_ground_truth_and_duplicate_keys():
    source = rows()
    with pytest.raises(ValueError, match='boundary'):
        validate_rows(source, 'reconstruct')
    with pytest.raises(ValueError, match='boundary'):
        validate_rows([{**source[0], 'target': {}}], 'render')
    with pytest.raises(ValueError, match='Duplicate'):
        validate_rows(source * 2, 'render')


def test_stage_resume_checks_outputs_without_replay(tmp_path):
    calls, verified = [], []
    def process(row):
        calls.append(row)
        return row
    def verify(row):
        verified.append(row)
    output = tmp_path / 'run'
    run_stage(rows(), 'render', {'runtime': 'one'}, output, process, verify)
    run_stage(rows(), 'render', {'runtime': 'one'}, output, process, verify, resume=True)
    assert len(calls) == 1 and len(verified) == 2
    with pytest.raises(ValueError, match='mismatch'):
        run_stage(rows(), 'render', {'runtime': 'two'}, output, process, verify, resume=True)
    assert json.loads((output / 'manifest.json').read_text())[0]['key']['split'] == 'test'


def test_interruption_requires_explicit_retry_and_preserves_attempt(tmp_path):
    def interrupt(row):
        raise KeyboardInterrupt()
    output = tmp_path / 'run'
    with pytest.raises(KeyboardInterrupt):
        run_stage(rows(), 'render', {}, output, interrupt, lambda row: None)
    with pytest.raises(RuntimeError, match='Uncertain'):
        run_stage(rows(), 'render', {}, output, lambda row: row, lambda row: None, resume=True)
    run_stage(rows(), 'render', {}, output, lambda row: row, lambda row: None,
              resume=True, retry_interrupted=True)
    item = json.loads(next((output / 'items').glob('*.json')).read_text())
    assert item['attempt'] == 2 and item['status'] == 'succeeded'


def test_failed_item_keeps_full_denominator(tmp_path):
    def fail(row):
        raise ValueError('model failed')
    output = tmp_path / 'run'
    assert run_stage(rows(), 'render', {}, output, fail, lambda row: None) == []
    status = json.loads((output / 'status.json').read_text())
    assert status == {'status': 'completed_with_failures', 'expected': 1, 'succeeded': 0, 'failed': 1}
