"""Real GMR inverse-kinematics adapter; instantiate once per clip to avoid state leakage."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from .contracts import ArtifactRef
from .human_motion import world_joint_frames
from .robot_contracts import RobotMotion
from .storage import digest


@dataclass(frozen=True)
class RetargetResult:
    motion: RobotMotion | None
    mapping_id: str
    diagnostics: ArtifactRef
    status: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RetargetConfig:
    actual_human_height: float = 1.8  # fixed declared assumption, not per-test truth fitting
    solver: str = 'daqp'
    offset_to_ground: bool = False
    use_velocity_limit: bool = False
    limit_tolerance_rad: float = 1e-4
    ground_placement: str = 'clip-min-foot-spheres'
    ground_clearance_m: float = 0.002


class GMRRetargeter:
    def __init__(self, store):
        self.store = store

    def retarget(self, motion, robot, config=RetargetConfig()):
        from general_motion_retargeting import GeneralMotionRetargeting, IK_CONFIG_DICT
        import mujoco
        frames = world_joint_frames(motion)
        engine = GeneralMotionRetargeting(src_human='smplx', tgt_robot='unitree_g1',
                                          actual_human_height=config.actual_human_height,
                                          solver=config.solver, verbose=False,
                                          use_velocity_limit=config.use_velocity_limit)
        model = engine.model
        names = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(1, model.njnt))
        if set(names) != set(robot.joint_names) or model.nq != 36:
            raise ValueError('GMR model is incompatible with target G1 joints')
        indices = [names.index(n) for n in robot.joint_names]
        mapping_ref = self.store.put_bytes(Path(IK_CONFIG_DICT['smplx']['unitree_g1']).read_bytes())
        model_ref = self.store.put_bytes(Path(engine.xml_file).read_bytes(), 'application/xml')
        qpos, errors = [], []
        for frame in frames:
            q = engine.retarget(frame, offset_to_ground=config.offset_to_ground)
            if not np.isfinite(q).all():
                raise ValueError('Retargeting produced nonfinite state')
            qpos.append(q.copy())
            errors.append([float(engine.error1()), float(engine.error2())])
        qpos = np.stack(qpos)
        if config.ground_placement != 'clip-min-foot-spheres' or config.offset_to_ground:
            raise ValueError('Use constant clip ground placement; per-frame grounding destroys vertical motion')
        ground_shift = ground_foot_spheres(model, qpos, config.ground_clearance_m)
        bounds = model.jnt_range[1:]
        violation = np.maximum(bounds[:,0] - qpos[:,7:], qpos[:,7:] - bounds[:,1])
        worst = float(np.maximum(violation, 0).max())
        accepted = worst <= config.limit_tolerance_rad
        diag = self.store.put_json({'mapping': mapping_ref, 'gmr_model': model_ref, 'config': config,
                                    'weighted_ik_errors': errors, 'max_joint_limit_violation_rad': worst,
                                    'frames': len(qpos), 'input': digest(motion),
                                    'ground_shift_z_m': ground_shift,
                                    'ground_assumption': 'lowest foot collision sphere across clip touches floor; constant translation',
                                    'limits_checked_against': 'GMR model; target compiler/executor still validates robot assets',
                                    'dynamic_feasibility': 'not_verified'})
        robot_motion = RobotMotion(robot.identity, motion.timestamps_s.copy(), qpos[:,:3], qpos[:,3:7],
                                   qpos[:,7:][:,indices], motion.valid_mask.copy()).validate(robot)
        return RetargetResult(robot_motion if accepted else None, mapping_ref.sha256, diag,
                              'accepted' if accepted else 'rejected', () if accepted else ('joint_limit_violation',))


def ground_foot_spheres(model, qpos, clearance=0.002):
    """Place an unknown-height clip using foot collision geometry, preserving all deltas.

    This assumes the clip contains a supporting foot contact. Fully airborne clips need
    a separately calibrated floor and must not use this protocol.
    """
    import mujoco
    if not np.isfinite(clearance) or clearance < 0:
        raise ValueError('Ground clearance must be finite and nonnegative')
    feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n)
            for n in ('left_ankle_roll_link', 'right_ankle_roll_link')}
    geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] in feet
             and model.geom_contype[g] != 0
             and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE]
    if len(geoms) != 8:
        raise ValueError('Expected eight G1 foot collision spheres for floor calibration')
    data = mujoco.MjData(model)
    lowest = np.inf
    for q in qpos:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        lowest = min(lowest, float(np.min(data.geom_xpos[geoms, 2] - model.geom_size[geoms, 0])))
    shift = float(clearance - lowest)
    qpos[:, 2] += shift
    return shift
