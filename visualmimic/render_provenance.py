"""Verify preparation-only motion/video links without changing inference inputs.

This module uses only the standard library. It can audit saved renders without
loading a body model, renderer, motion editor, or policy.
"""
import hashlib
import json
from pathlib import Path


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def checked_bytes(ref):
    if ref.get('schema_version') != '1':
        raise ValueError('Unsupported artifact schema')
    data = Path(ref['uri']).read_bytes()
    if hashlib.sha256(data).hexdigest() != ref['sha256']:
        raise ValueError('Artifact checksum mismatch')
    return data


def checked_json(ref):
    return json.loads(checked_bytes(ref))


def key_of(row):
    key = row['key']
    if set(key) != {'dataset_version', 'sample_id', 'split'}:
        raise ValueError('Invalid sample key')
    return tuple(key[name] for name in ('dataset_version', 'split', 'sample_id'))


def unique_index(rows):
    result = {}
    for row in rows:
        key = key_of(row)
        if key in result:
            raise ValueError('Duplicate sample key')
        result[key] = row
    return result


def preparation_index(directory):
    """Read immutable binding records from an ArtifactStore, skipping media."""
    records = []
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():
            continue
        with path.open('rb') as stream:
            if stream.read(1) != b'{':
                continue
            data = b'{' + stream.read()
        value = json.loads(data)
        if not isinstance(value, dict) or not {'key', 'source', 'render'} <= value.keys():
            continue
        if hashlib.sha256(data).hexdigest() != path.name:
            raise ValueError('Preparation record checksum mismatch')
        records.append({**value, 'record_sha256': path.name})
    return unique_index(records)


def verify_binding(task, video, preparation):
    """Bind the expected GMR input to the exact rendered bytes and timestamps."""
    if key_of(task) != key_of(video) or key_of(task) != key_of(preparation):
        raise ValueError('Preparation sample mismatch')
    if task['instruction'] != video['instruction']:
        raise ValueError('Video instruction mismatch')
    expected = task['pregmr_human']
    if expected['sha256'] != preparation['source']['sha256']:
        raise ValueError('Source motion mismatch')
    source = checked_json(expected)
    rendered = preparation['render']
    metadata = checked_json(rendered['metadata'])
    # source_hash is the canonical MotionSequence hash, not an arbitrary file hash.
    if metadata['source_hash'] != digest_json(source):
        raise ValueError('Rendered source hash mismatch')
    if rendered['video']['sha256'] != video['video']['sha256']:
        raise ValueError('Rendered video mismatch')
    checked_bytes(video['video'])
    if rendered['timestamps_s'] != video['timestamps_s']:
        raise ValueError('Rendered timestamps mismatch')
    if metadata['frames'] != len(video['timestamps_s']):
        raise ValueError('Rendered frame count mismatch')
    if rendered['config_sha256'] != digest_json(metadata['config']):
        raise ValueError('Render configuration checksum mismatch')
    for field in ('width', 'height'):
        if metadata['config'][field] != video[field]:
            raise ValueError('Rendered dimensions mismatch')
    return {'key': task['key'], 'source_sha256': expected['sha256'],
            'canonical_source_sha256': digest_json(source),
            'rendered_source_sha256': metadata['source_hash'],
            'video_sha256': video['video']['sha256'],
            'prepared_video_sha256': rendered['video']['sha256'],
            'metadata_sha256': rendered['metadata']['sha256'],
            'preparation_sha256': preparation['record_sha256'],
            'config_sha256': rendered['config_sha256'],
            'timestamps_sha256': digest_json(video['timestamps_s'])}


def verify_pre_gmr_videos(tasks, videos, directory):
    expected, actual = unique_index(tasks), unique_index(videos)
    preparation = preparation_index(directory)
    if not expected or expected.keys() != actual.keys() or expected.keys() != preparation.keys():
        raise ValueError('Preparation/video cohort mismatch')
    return [verify_binding(task, actual[key_of(task)], preparation[key_of(task)])
            for task in tasks]
