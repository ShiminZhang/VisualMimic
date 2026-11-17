"""Robot-side contracts. No learned backend is imported by this module."""
from dataclasses import dataclass
import numpy as np
from .contracts import ArtifactRef
from .storage import digest


@dataclass(frozen=True)
class RobotSpec:
    robot_id: str
    model_asset: ArtifactRef
    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    root_body: str = 'pelvis'
    coordinate_spec: str = 'right-handed-z-up-metres-wxyz'

    @property
    def identity(self):
        return digest({"robot_id": self.robot_id, "model_sha256": self.model_asset.sha256,
                       "joint_names": self.joint_names, "body_names": self.body_names,
                       "root_body": self.root_body, "coordinate_spec": self.coordinate_spec})


@dataclass
class RobotMotion:
    robot_spec_id: str
    timestamps_s: np.ndarray
    root_pos_w: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray
    valid_mask: np.ndarray

    def validate(self, robot):
        t = np.asarray(self.timestamps_s)
        if self.robot_spec_id != robot.identity:
            raise ValueError('Robot identity mismatch')
        if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
            raise ValueError('At least two increasing finite timestamps required')
        for name, shape in [('root_pos_w', (len(t), 3)), ('root_quat_wxyz', (len(t), 4)),
                            ('joint_pos', (len(t), len(robot.joint_names)))]:
            a = getattr(self, name)
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'Invalid {name}')
        if not np.allclose(np.linalg.norm(self.root_quat_wxyz, axis=-1), 1, atol=1e-5):
            raise ValueError('Root quaternions must be normalized')
        if self.valid_mask.dtype != np.bool_ or self.valid_mask.shape != t.shape or not self.valid_mask.all():
            raise ValueError('Execution does not accept missing robot frames')
        return self


@dataclass(frozen=True)
class TrackingReference:
    reference_id: str
    robot_spec_id: str
    motion_npz: ArtifactRef
    num_frames: int
    fps: float
    duration_s: float
    compiler_identity: str
    provenance: ArtifactRef


@dataclass(frozen=True)
class PolicyBundle:
    checkpoint: ArtifactRef
    implementation_commit: str
    robot_spec_id: str
    observation_spec_id: str
    action_spec_id: str
    environment_config: ArtifactRef
    training_manifest: ArtifactRef | None = None


@dataclass(frozen=True)
class ExecutionProtocol:
    seed: int = 0
    initialization: str = 'reference_start'
    phase_start: int = 0
    max_steps: int | None = None
    obs_noise: bool = False
    reset_noise: bool = False
    action_noise_std: float = 0.0
    domain_randomization: bool = False
    root_height_threshold: float = 0.25
    root_tilt_threshold: float = 0.8

    def validate(self):
        if self.initialization != 'reference_start' or self.phase_start != 0:
            raise ValueError('Only reference_start at frame zero is implemented')
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError('max_steps must be positive')
        if self.obs_noise or self.reset_noise or self.action_noise_std != 0 or self.domain_randomization:
            raise ValueError('Stage-one executor currently supports deterministic evaluation only')
        return self

    @property
    def identity(self):
        return digest(self)


@dataclass(frozen=True)
class RobotRollout:
    rollout_id: str
    reference_id: str
    policy_id: str
    protocol_id: str
    trajectory: ArtifactRef
    events: ArtifactRef
    planned_steps: int
    executed_steps: int
    outcome: str
    metadata: ArtifactRef


def validate_rollout_arrays(a):
    k = len(a['actions'])
    if k < 1 or a['states'].shape != (k + 1, 71) or a['actions'].shape != (k, 29):
        raise ValueError('Expected K actions and K+1 complete G1 states')
    for name in ('executed_actions', 'joint_targets'):
        if a[name].shape != (k, 29):
            raise ValueError(f'Invalid {name}')
    if a['reference_qpos'].shape != (k + 1, 36) or a['reference_frames'].shape != (k + 1,):
        raise ValueError('Invalid reference/state correspondence')
    if a['timestamps_s'].shape != (k + 1,) or np.any(np.diff(a['timestamps_s']) <= 0):
        raise ValueError('Invalid rollout timestamps')
    for name, value in a.items():
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise ValueError(f'Nonfinite rollout field: {name}')
    return a
