"""G1 reference import/compilation using the pinned SuperMimic CPU FK backend."""
import io
from pathlib import Path
import numpy as np
from .robot_contracts import RobotSpec, TrackingReference
from .storage import digest


def put_npz(store, arrays):
    stream = io.BytesIO()
    np.savez_compressed(stream, **arrays)
    return store.put_bytes(stream.getvalue(), 'application/x-npz')


def read_npz(store, ref):
    with np.load(io.BytesIO(store.read_bytes(ref)), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def g1_robot_spec(store):
    from supermimic.data.motion_convert import load_g1_model, g1_xml_path, G1_JOINT_NAMES
    import mujoco
    model = load_g1_model()
    bodies = tuple(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(1, model.nbody))
    asset = store.put_bytes(Path(g1_xml_path()).read_bytes(), 'application/xml')
    return RobotSpec('unitree-g1-29dof', asset, G1_JOINT_NAMES, bodies)


def validate_reference(arrays, robot):
    j = arrays['joint_pos']
    t, nj, nb = len(j), len(robot.joint_names), len(robot.body_names)
    if t < 2 or tuple(arrays['joint_names'].tolist()) != robot.joint_names or tuple(arrays['body_names'].tolist()) != robot.body_names:
        raise ValueError('Reference joint/body order or frame count mismatch')
    fps = np.asarray(arrays['fps']).reshape(-1)
    if len(fps) != 1 or not np.isfinite(fps[0]) or fps[0] <= 0:
        raise ValueError('Invalid reference rate')
    for name, shape in [('joint_pos', (t, nj)), ('joint_vel', (t, nj)), ('body_pos_w', (t, nb, 3)),
                        ('body_quat_w', (t, nb, 4)), ('body_lin_vel_w', (t, nb, 3)), ('body_ang_vel_w', (t, nb, 3))]:
        a = arrays[name]
        if a.shape != shape or not np.isfinite(a).all():
            raise ValueError(f'Invalid reference field {name}')
    if not np.allclose(np.linalg.norm(arrays['body_quat_w'], axis=-1), 1, atol=1e-4):
        raise ValueError('Invalid reference quaternions')
    return float(fps[0])


class ReferenceCompiler:
    identity = 'supermimic-cpu-fk-v1'

    def __init__(self, store, robot):
        self.store, self.robot = store, robot

    def _publish(self, arrays, provenance):
        fps = validate_reference(arrays, self.robot)
        ref = put_npz(self.store, arrays)
        prov = self.store.put_json(provenance)
        return TrackingReference(ref.sha256, self.robot.identity, ref, len(arrays['joint_pos']), fps,
                                 (len(arrays['joint_pos']) - 1) / fps, self.identity, prov)

    def import_npz(self, path):
        # Explicit trusted local input; bytes are ingested before validation.
        ref = self.store.put_bytes(Path(path).read_bytes(), 'application/x-npz')
        arrays = read_npz(self.store, ref)
        return self._publish(arrays, {'operation': 'import', 'input': ref, 'robot': self.robot})

    def compile(self, motion, output_fps=50.0):
        from supermimic.data.motion_convert import resample_motion, forward_kinematics, load_g1_model
        motion.validate(self.robot)
        dt = np.diff(motion.timestamps_s)
        if not np.allclose(dt, dt[0], atol=1e-8, rtol=1e-5):
            raise ValueError('Compiler requires explicit uniform robot sampling')
        raw = np.concatenate([motion.root_pos_w, motion.root_quat_wxyz[:, [1, 2, 3, 0]], motion.joint_pos], axis=-1)
        sampled = resample_motion(raw, 1 / dt[0], output_fps)
        model = load_g1_model()
        from supermimic.data.motion_convert import g1_xml_path
        from hashlib import sha256
        if sha256(Path(g1_xml_path()).read_bytes()).hexdigest() != self.robot.model_asset.sha256:
            raise ValueError('Compiler model asset differs from RobotSpec')
        if np.any(motion.joint_pos < model.jnt_range[1:, 0] - 1e-4) or np.any(motion.joint_pos > model.jnt_range[1:, 1] + 1e-4):
            raise ValueError('Input robot motion exceeds target-model joint limits')
        if tuple(self.robot.joint_names) != tuple(__import__('supermimic.data.motion_convert', fromlist=['G1_JOINT_NAMES']).G1_JOINT_NAMES):
            raise ValueError('This compiler supports only the configured G1 joint order')
        bodies = forward_kinematics(model, sampled)
        arrays = {'fps': np.array([output_fps], dtype=np.float32), 'joint_pos': sampled.joint_pos.astype(np.float32),
                  'joint_vel': sampled.joint_vel.astype(np.float32), 'body_pos_w': bodies.pos_w.astype(np.float32),
                  'body_quat_w': bodies.quat_w.astype(np.float32), 'body_lin_vel_w': bodies.lin_vel_w.astype(np.float32),
                  'body_ang_vel_w': bodies.ang_vel_w.astype(np.float32), 'joint_names': np.array(self.robot.joint_names),
                  'body_names': np.array(bodies.body_names)}
        return self._publish(arrays, {'operation': 'compile', 'input_hash': digest(motion), 'robot': self.robot,
                                     'input_fps': float(1 / dt[0]), 'output_fps': output_fps})
