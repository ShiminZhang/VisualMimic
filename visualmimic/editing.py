from typing import Protocol
from .contracts import BackendIdentity, MotionSpec, MotionSequence, GenerationOptions


class Editor(Protocol):
    identity: BackendIdentity
    input_spec: MotionSpec
    output_spec: MotionSpec

    def edit(self, source: MotionSequence, instruction: str,
             options: GenerationOptions) -> MotionSequence: ...
