"""Normalized dataset boundary. Official MotionFix format adapter remains external."""
from dataclasses import dataclass, asdict
from .contracts import ArtifactRef, SampleKey, Source3DRequest, VideoRequest, VideoInput, EvalReference


@dataclass(frozen=True)
class DatasetRecord:
    key: SampleKey
    instruction: str
    source_motion_id: str
    target_motion_id: str
    source: ArtifactRef
    target: ArtifactRef


@dataclass(frozen=True)
class PreparedDataset:
    preparation_manifest: ArtifactRef
    source_manifest: ArtifactRef
    label_manifest: ArtifactRef


def prepare(records, input_store, label_store):
    records = list(records)
    seen, splits = set(), {}
    for r in records:
        if r.key in seen:
            raise ValueError('Duplicate sample key')
        seen.add(r.key)
        for identity in (r.source_motion_id, r.target_motion_id, r.source.sha256, r.target.sha256):
            token = (r.key.dataset_version, identity)
            if token in splits and splits[token] != r.key.split:
                raise ValueError('Motion appears across official splits; refusing preparation')
            splits[token] = r.key.split
        input_store.read_bytes(r.source)
        label_store.read_bytes(r.target)
    return PreparedDataset(
        label_store.put_json(records),
        input_store.put_json([Source3DRequest(r.key, r.instruction, r.source) for r in records]),
        label_store.put_json([EvalReference(r.key, r.source, r.target) for r in records]))


def source_requests(store, manifest, split):
    for row in store.read_json(manifest):
        if set(row) != {'key', 'instruction', 'source'}:
            raise ValueError('Unexpected source request fields')
        r = Source3DRequest(SampleKey(**row['key']), row['instruction'], ArtifactRef(**row['source']))
        if r.key.split == split:
            yield r


def video_requests(store, manifest, split):
    for row in store.read_json(manifest):
        if set(row) != {'key', 'instruction', 'video'}:
            raise ValueError('Unexpected video request fields (possible label leakage)')
        v = dict(row['video'])
        v['key'], v['video'] = SampleKey(**v['key']), ArtifactRef(**v['video'])
        v['timestamps_s'] = tuple(v['timestamps_s'])
        r = VideoRequest(SampleKey(**row['key']), row['instruction'], VideoInput(**v))
        if r.key.split == split:
            yield r


def references(store, manifest, split):
    for row in store.read_json(manifest):
        r = EvalReference(SampleKey(**row['key']), ArtifactRef(**row['source']), ArtifactRef(**row['target']))
        if r.key.split == split:
            yield r


def render_dataset(requests, source_store, video_store, renderer, config):
    """Publish video-only requests; metadata stays with renderer/preparation storage."""
    from .storage import digest
    rows, seen = [], set()
    for r in requests:
        if r.key in seen:
            raise ValueError('Duplicate sample key')
        seen.add(r.key)
        result = renderer.render(source_store.read_motion(r.source), config)
        if result.config_sha256 != digest(config):
            raise ValueError('Renderer configuration mismatch')
        video_store.read_bytes(result.video)
        video = VideoInput(r.key, result.video, result.timestamps_s, config.width, config.height)
        rows.append(VideoRequest(r.key, r.instruction, video))
    return video_store.put_json(rows)
