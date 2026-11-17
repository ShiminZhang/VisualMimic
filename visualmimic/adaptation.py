from dataclasses import dataclass
from typing import Protocol
from .contracts import ArtifactRef, BackendIdentity, MotionSequence, MotionSpec
from .storage import digest


@dataclass(frozen=True)
class AdaptationPolicy:
    policy_id: str = 'strict-identity-v1'


@dataclass(frozen=True)
class AdaptedMotion:
    motion: MotionSequence
    transform_record: ArtifactRef
    quality_flags: tuple[str, ...] = ()


class MotionAdapter(Protocol):
    identity: BackendIdentity

    def adapt(self, motion: MotionSequence, target_spec: MotionSpec,
              policy: AdaptationPolicy) -> AdaptedMotion: ...


class IdentityAdapter:
    """Safe initial adapter. Unknown conversions fail rather than fill fields."""
    identity = BackendIdentity('identity-adapter', '1', digest({'conversion': 'none'}))

    def __init__(self, store):
        self.store = store

    def adapt(self, motion, target_spec, policy):
        motion.validate()
        if policy.policy_id != 'strict-identity-v1' or motion.spec != target_spec:
            raise ValueError('No registered conversion for this specification/policy')
        return AdaptedMotion(motion, self.store.put_json({'operation': 'identity', 'spec': target_spec, 'policy': policy}))
