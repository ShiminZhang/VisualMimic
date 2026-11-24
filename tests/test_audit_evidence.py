import json
from copy import deepcopy
from pathlib import Path

import pytest

from audit.evaluate import load_tables, summarize
from audit.verify_evidence import verify
from visualmimic.render_provenance import digest_json
from visualmimic.judgment_accounting import combine_judgments


AUDIT = Path(__file__).resolve().parents[1] / 'audit'


def small_tables(tmp_path, verdict='uncertain', rank='1'):
    (tmp_path / 'judgments.csv').write_text('sample_id,pre_GMR,C,A,B\none,success,failure,' + verdict + ',technical_error\n')
    (tmp_path / 'composite_ranks.csv').write_text('sample_id,A,B,C\none,' + rank + ',1,1\n')


def test_uncertain_and_error_remain_in_full_denominator(tmp_path):
    small_tables(tmp_path)
    results = summarize(*load_tables(tmp_path, 1, 1))
    assert results['A']['judged'] == results['B']['judged'] == 1
    assert results['A']['uncertain'] == results['B']['technical_error'] == 1
    assert results['A']['success'] == results['B']['success'] == 0


@pytest.mark.parametrize('verdict', ['unknown', '', 'Success'])
def test_unknown_verdict_is_rejected(tmp_path, verdict):
    small_tables(tmp_path, verdict=verdict)
    with pytest.raises(ValueError, match='verdict'):
        load_tables(tmp_path, 1, 1)


@pytest.mark.parametrize('rank', ['nan', 'inf', '0', '2', '0.5'])
def test_invalid_rank_is_rejected(tmp_path, rank):
    small_tables(tmp_path, rank=rank)
    with pytest.raises(ValueError, match='Rank'):
        load_tables(tmp_path, 1, 1)


def test_duplicate_ids_and_missing_columns_are_rejected(tmp_path):
    small_tables(tmp_path)
    with (tmp_path / 'judgments.csv').open('a') as stream:
        stream.write('one,success,failure,uncertain,technical_error\n')
    with pytest.raises(ValueError, match='duplicate'):
        load_tables(tmp_path, 2, 1)
    (tmp_path / 'judgments.csv').write_text('sample_id,pre_GMR,C,A\none,success,failure,success\n')
    with pytest.raises(ValueError, match='columns'):
        load_tables(tmp_path, 1, 1)


@pytest.fixture(scope='module')
def released():
    return json.loads((AUDIT / 'evidence/c2_c4.json').read_text())


def test_released_evidence_reproduces_audit_tables(released):
    result = verify(released, *load_tables(AUDIT))
    assert (result['C2_matched_pairs'], result['C3_verified_links'], result['C4_judgments']) == (307, 100, 400)
    assert [result['verdicts'][g]['success'] for g in ('pre_GMR', 'C', 'A', 'B')] == [35, 27, 24, 21]


def test_changed_seed_does_not_pass_as_matched_settings(released):
    data = deepcopy(released)
    first = data['c2']['items'][0]
    changed = deepcopy(data['c2']['configurations'][first['B']['settings_sha256']])
    changed['editor']['options']['seed'] += 1
    identity = digest_json(changed)
    data['c2']['configurations'][identity] = changed
    first['B']['settings_sha256'] = identity
    with pytest.raises(ValueError, match='A/B settings mismatch'):
        verify(data, *load_tables(AUDIT))


@pytest.mark.parametrize('change', ['duplicate', 'missing', 'changed_verdict', 'wrong_video'])
def test_bad_judgment_evidence_is_rejected(released, change):
    data = deepcopy(released)
    if change == 'duplicate': data['c4'].append(deepcopy(data['c4'][0]))
    if change == 'missing': data['c4'].pop()
    if change == 'changed_verdict': data['c4'][0]['verdict'] = 'uncertain'
    if change == 'wrong_video': data['c4'][0]['candidate_video_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        verify(data, *load_tables(AUDIT))


def test_failed_assessment_is_counted_as_technical_error():
    task = {'key': {'dataset_version': 'fixture', 'sample_id': 'one', 'split': 'test'}, 'instruction': 'walk'}
    items = [{**task, 'branch': g, 'assessment': {'verdict': 'uncertain', 'reason': 'occluded'}} for g in ('pre_GMR', 'A', 'B', 'C')]
    items[0].pop('assessment')
    items[0]['error'] = 'request failed'
    rows, counts = combine_judgments([task], items)
    assert rows[0]['pre_GMR']['verdict'] == 'technical_error'
    assert counts['pre_GMR'] == {'success': 0, 'failure': 0, 'uncertain': 0, 'technical_error': 1}
    assert counts['A']['uncertain'] == 1
    with pytest.raises(ValueError, match='Missing judgment'):
        combine_judgments([task], items[:-1])
    with pytest.raises(ValueError, match='Duplicate judgment'):
        combine_judgments([task], items + [items[0]])
