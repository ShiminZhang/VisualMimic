import hashlib
import json
from copy import deepcopy

import pytest

from visualmimic.render_provenance import digest_json, verify_pre_gmr_videos


def artifact(directory, value, media_type='application/json'):
    data = value if isinstance(value, bytes) else json.dumps(value, indent=2).encode()
    checksum = hashlib.sha256(data).hexdigest()
    path = directory / checksum
    path.write_bytes(data)
    return {'uri': str(path), 'sha256': checksum, 'media_type': media_type, 'schema_version': '1'}


def fixture(tmp_path):
    prep = tmp_path / 'preparation'
    prep.mkdir()
    # Pretty JSON deliberately differs from the canonical MotionSequence digest.
    source = {'motion': [1.0, 2.0]}
    motion = artifact(tmp_path, source)
    video_ref = artifact(tmp_path, b'original rendered clip', 'video/mp4')
    config = {'width': 640, 'height': 480, 'fps': 30}
    metadata = artifact(prep, {'source_hash': digest_json(source), 'frames': 2, 'config': config})
    key = {'dataset_version': 'fixture', 'split': 'test', 'sample_id': 'one'}
    task = {'key': key, 'instruction': 'raise the left arm', 'pregmr_human': motion}
    video = {'key': key, 'instruction': task['instruction'], 'video': video_ref,
             'timestamps_s': [0., 1 / 30], 'width': 640, 'height': 480}
    binding = {'key': key, 'source': motion, 'render': {'video': video_ref,
               'metadata': metadata, 'timestamps_s': video['timestamps_s'], 'config_sha256': digest_json(config)}}
    artifact(prep, binding)
    return task, video, prep, binding


def test_exact_binding_accepts_different_canonical_and_file_hashes(tmp_path):
    task, video, prep, _ = fixture(tmp_path)
    before = deepcopy(video)
    evidence = verify_pre_gmr_videos([task], [video], prep)[0]
    assert evidence['canonical_source_sha256'] != evidence['source_sha256']
    assert evidence['video_sha256'] == video['video']['sha256']
    assert video == before  # No preparation data leaks into the video-only manifest.


def test_swapped_video_with_same_key_and_instruction_is_rejected(tmp_path):
    task, video, prep, _ = fixture(tmp_path)
    video['video'] = artifact(tmp_path, b'another clip', 'video/mp4')
    with pytest.raises(ValueError, match='video mismatch'):
        verify_pre_gmr_videos([task], [video], prep)


def test_wrong_expected_motion_is_rejected(tmp_path):
    task, video, prep, _ = fixture(tmp_path)
    task['pregmr_human'] = artifact(tmp_path, {'motion': [3., 4.]})
    with pytest.raises(ValueError, match='Source motion mismatch'):
        verify_pre_gmr_videos([task], [video], prep)


def test_changed_video_bytes_are_rejected(tmp_path):
    task, video, prep, _ = fixture(tmp_path)
    from pathlib import Path
    Path(video['video']['uri']).write_bytes(b'changed after rendering')
    with pytest.raises(ValueError, match='checksum mismatch'):
        verify_pre_gmr_videos([task], [video], prep)


def test_duplicate_video_keys_are_not_silently_overwritten(tmp_path):
    task, video, prep, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match='Duplicate'):
        verify_pre_gmr_videos([task], [video, video], prep)


def test_wrong_timestamps_are_rejected(tmp_path):
    task, video, prep, _ = fixture(tmp_path)
    video['timestamps_s'] = [0., .25]
    with pytest.raises(ValueError, match='timestamps mismatch'):
        verify_pre_gmr_videos([task], [video], prep)


def test_missing_preparation_record_is_rejected(tmp_path):
    task, video, prep, binding = fixture(tmp_path)
    record = artifact(prep, binding)
    from pathlib import Path
    Path(record['uri']).unlink()
    with pytest.raises(ValueError, match='cohort mismatch'):
        verify_pre_gmr_videos([task], [video], prep)
