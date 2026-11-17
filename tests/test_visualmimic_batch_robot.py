import pytest
from visualmimic.batch_robot import join_inputs, validate_expected, human_predictions


def tasks():
    return [{'key': {'dataset_version': 'fixture', 'sample_id': name, 'split': 'test'},
             'instruction': 'walk'} for name in ('one', 'two')]


def test_missing_predictions_retain_official_order_and_denominator():
    expected = tasks()
    predictions = [{'key': expected[1]['key'], 'motion': {'uri': 'fixture', 'sha256': 'fixture'}}]
    joined = join_inputs(expected, predictions, 'motion')
    assert len(joined) == 2 and joined[0]['motion'] is None
    assert joined[1]['motion'] == predictions[0]['motion']
    assert [r['key'] for r in joined] == [r['key'] for r in expected]


def test_join_rejects_wrong_split_duplicates_and_label_fields():
    expected = tasks()
    prediction = {'key': expected[0]['key'], 'motion': {}}
    with pytest.raises(ValueError, match='Duplicate'):
        join_inputs(expected, [prediction, prediction], 'motion')
    with pytest.raises(ValueError, match='unexpected'):
        join_inputs(expected, [{**prediction, 'key': {**expected[0]['key'], 'split': 'train'}}], 'motion')
    with pytest.raises(ValueError, match='fields'):
        join_inputs(expected, [{**prediction, 'target': {}}], 'motion')
    with pytest.raises(ValueError, match='task-only'):
        validate_expected([{**expected[0], 'target': {}}])


def test_robot_join_checks_instruction_and_retains_provenance():
    expected = tasks()
    row = {**expected[0], 'robot_motion': {}, 'retarget_report': {'sha256': 'fixture'}}
    joined = join_inputs(expected, [row], 'robot_motion')
    assert joined[0]['retarget_report'] == row['retarget_report']
    assert joined[1]['robot_motion'] is None and joined[1]['retarget_report'] is None
    with pytest.raises(ValueError, match='instruction'):
        join_inputs(expected, [{**row, 'instruction': 'jump'}], 'robot_motion')


def test_reconstructed_sources_join_without_ground_truth_access():
    expected = tasks()
    reconstructed = [{**expected[1], 'source': {'uri': 'video-only-output', 'sha256': 'fixture'}}]
    joined = join_inputs(expected, human_predictions(expected, reconstructed), 'motion')
    assert joined[0]['motion'] is None
    assert joined[1]['motion']['uri'] == 'video-only-output'
    with pytest.raises(ValueError, match='instruction'):
        human_predictions(expected, [{**reconstructed[0], 'instruction': 'jump'}])
