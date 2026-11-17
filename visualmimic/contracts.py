"""Validated, label-free inference contracts."""
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass(frozen=True)
class ArtifactRef:
    uri: str
    sha256: str
    media_type: str = 'application/json'
    schema_version: str = '1'


@dataclass(frozen=True)
class SampleKey:
    dataset_version: str
    sample_id: str
    split: str

    def __post_init__(self):
        if not self.dataset_version or not self.sample_id or self.split not in {'train', 'val', 'test'}:
            raise ValueError('Invalid sample identity or split')


@dataclass(frozen=True)
class FieldSpec:
    shape: tuple[int, ...]  # trailing dimensions; every field is time-indexed
    dtype: str = 'float32'


@dataclass(frozen=True)
class MotionSpec:
    spec_id: str
    representation: str
    skeleton_id: str
    coordinate_system: str
    units: str
    field_schema: Mapping[str, FieldSpec]


@dataclass
class MotionSequence:
    spec: MotionSpec
    timestamps_s: np.ndarray
    fields: Mapping[str, np.ndarray]
    valid_mask: np.ndarray

    def validate(self):
        t = np.asarray(self.timestamps_s)
        if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('Timestamps must be nonempty, finite and strictly increasing')
        if self.valid_mask.dtype != np.bool_ or self.valid_mask.shape != t.shape:
            raise ValueError('Invalid frame mask')
        if not self.valid_mask.any():
            raise ValueError('Motion contains no valid frames')
        if not self.spec.field_schema or set(self.fields) != set(self.spec.field_schema):
            raise ValueError('Motion fields disagree with specification')
        for name, schema in self.spec.field_schema.items():
            a = self.fields[name]
            if a.shape != (len(t), *schema.shape) or a.dtype != np.dtype(schema.dtype):
                raise ValueError(f'Invalid shape/dtype: {name}')
            if not np.isfinite(a).all():
                raise ValueError(f'Nonfinite motion field: {name}')
        return self


@dataclass(frozen=True)
class VideoInput:
    key: SampleKey
    video: ArtifactRef
    timestamps_s: tuple[float, ...]
    width: int
    height: int

    def __post_init__(self):
        t = np.asarray(self.timestamps_s)
        if self.width <= 0 or self.height <= 0 or t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('Invalid video dimensions/timestamps')


@dataclass(frozen=True)
class Source3DRequest:
    key: SampleKey
    instruction: str
    source: ArtifactRef


@dataclass(frozen=True)
class VideoRequest:
    key: SampleKey
    instruction: str
    video: VideoInput

    def __post_init__(self):
        if self.key != self.video.key:
            raise ValueError('Video/request sample mismatch')


@dataclass(frozen=True)
class EvalReference:
    key: SampleKey
    source: ArtifactRef
    target: ArtifactRef


@dataclass(frozen=True)
class GenerationOptions:
    seed: int
    duration_policy_id: str
    sampling_config_id: str


@dataclass(frozen=True)
class BackendIdentity:
    name: str
    version: str
    config_sha256: str
    checkpoint_sha256: str | None = None
    fixture: bool = False


@dataclass(frozen=True)
class Prediction:
    key: SampleKey
    condition_id: str
    seed: int
    motion: ArtifactRef
    trace: ArtifactRef


@dataclass(frozen=True)
class StageError:
    stage: str
    code: str
    message: str
    retryable: bool = False


@dataclass(frozen=True)
class RunResult:
    key: SampleKey
    condition_id: str
    seed: int
    status: str
    prediction: Prediction | None
    error: StageError | None
    trace: ArtifactRef
