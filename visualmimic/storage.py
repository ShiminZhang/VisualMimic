"""Content-addressed local artifacts, verified at every read. No pickle."""
from dataclasses import asdict, is_dataclass
from hashlib import sha256
from pathlib import Path
import json
import os
import tempfile
import numpy as np
from .contracts import ArtifactRef, FieldSpec, MotionSpec, MotionSequence


def _default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def encode(value):
    return json.dumps(value, default=_default, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return sha256(encode(value)).hexdigest()


class ArtifactStore:
    def __init__(self, root, allowed_read_roots=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.allowed = tuple(Path(p).resolve() for p in (allowed_read_roots or [])) + (self.root,)

    def put_bytes(self, data, media_type='application/json'):
        checksum = sha256(data).hexdigest()
        path = self.root / checksum
        fd, temporary = tempfile.mkstemp(dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(data)
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return ArtifactRef(str(path), checksum, media_type)

    def read_bytes(self, ref):
        path = Path(ref.uri).resolve()
        if not any(path.is_relative_to(root) for root in self.allowed):
            raise PermissionError('Artifact is outside allowed input roots')
        if ref.schema_version != '1':
            raise ValueError('Unsupported artifact schema')
        data = path.read_bytes()
        if sha256(data).hexdigest() != ref.sha256:
            raise ValueError('Artifact checksum mismatch')
        return data

    def put_json(self, value):
        return self.put_bytes(encode(value))

    def read_json(self, ref):
        return json.loads(self.read_bytes(ref))

    def put_motion(self, motion):
        motion.validate()
        return self.put_json(motion)

    def read_motion(self, ref):
        value = self.read_json(ref)
        spec = dict(value['spec'])
        spec['field_schema'] = {k: FieldSpec(tuple(v['shape']), v['dtype']) for k, v in spec['field_schema'].items()}
        spec = MotionSpec(**spec)
        if any(type(v) is not bool for v in value['valid_mask']):
            raise ValueError('Frame mask must contain JSON booleans')
        return MotionSequence(spec, np.asarray(value['timestamps_s'], dtype=np.float64),
                              {k: np.asarray(v, dtype=spec.field_schema[k].dtype) for k, v in value['fields'].items()},
                              np.asarray(value['valid_mask'], dtype=bool)).validate()
