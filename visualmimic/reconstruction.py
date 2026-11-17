from dataclasses import dataclass
from typing import Protocol
from .contracts import ArtifactRef, BackendIdentity, VideoInput


@dataclass(frozen=True)
class ReconstructionResult:
    motion: ArtifactRef
    confidence: ArtifactRef | None
    diagnostics: ArtifactRef


class Reconstructor(Protocol):
    identity: BackendIdentity

    def reconstruct(self, video: VideoInput) -> ReconstructionResult: ...
