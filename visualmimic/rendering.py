from dataclasses import dataclass
from typing import Protocol
from .contracts import ArtifactRef, BackendIdentity, MotionSequence


@dataclass(frozen=True)
class RenderConfig:
    camera: str
    appearance: str
    lighting: str
    background: str
    fps: float
    width: int
    height: int

    def __post_init__(self):
        if not 0 < self.fps < float('inf') or self.width <= 0 or self.height <= 0:
            raise ValueError('Invalid render configuration')


@dataclass(frozen=True)
class RenderResult:
    video: ArtifactRef
    timestamps_s: tuple[float, ...]
    config_sha256: str
    metadata: ArtifactRef  # preparation side only


class Renderer(Protocol):
    identity: BackendIdentity

    def render(self, source: MotionSequence, config: RenderConfig) -> RenderResult: ...
