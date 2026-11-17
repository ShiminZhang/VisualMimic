import dataclasses
from pathlib import Path
import tempfile
import unittest
import numpy as np
from visualmimic.storage import ArtifactStore
from visualmimic.robot_contracts import RobotSpec, RobotMotion, ExecutionProtocol, validate_rollout_arrays, RobotRollout
from visualmimic.reference_compiler import ReferenceCompiler, read_npz
from visualmimic.tracking_evaluation import evaluate_tracking


class RobotContractsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = ArtifactStore(self.tmp.name)
        self.robot = RobotSpec('test', self.store.put_bytes(b'xml'), tuple(f'j{i}' for i in range(29)), ('pelvis', 'foot'))
        quat = np.zeros((3, 2, 4), dtype=np.float32)
        quat[:, :, 0] = 1
        self.arrays = {'fps': np.array([50.]), 'joint_names': np.array(self.robot.joint_names),
                       'body_names': np.array(self.robot.body_names), 'joint_pos': np.zeros((3,29)),
                       'joint_vel': np.zeros((3,29)), 'body_pos_w': np.zeros((3,2,3)), 'body_quat_w': quat,
                       'body_lin_vel_w': np.zeros((3,2,3)), 'body_ang_vel_w': np.zeros((3,2,3))}

    def test_reference_import_checks_order_and_quaternions(self):
        compiler = ReferenceCompiler(self.store, self.robot)
        p = Path(self.tmp.name) / 'input.npz'
        np.savez(p, **self.arrays)
        ref = compiler.import_npz(p)
        self.assertEqual(ref.num_frames, 3)
        self.assertAlmostEqual(ref.duration_s, .04)
        self.arrays['joint_names'] = self.arrays['joint_names'][::-1]
        np.savez(p, **self.arrays)
        with self.assertRaises(ValueError): compiler.import_npz(p)

    def test_missing_robot_frames_and_nonunit_quaternion_rejected(self):
        motion = RobotMotion(self.robot.identity, np.array([0., .02]), np.zeros((2,3)),
                             np.array([[1.,0,0,0],[1.,0,0,0]]), np.zeros((2,29)), np.array([True,True]))
        motion.validate(self.robot)
        with self.assertRaises(ValueError):
            dataclasses.replace(motion, valid_mask=np.array([True,False])).validate(self.robot)
        with self.assertRaises(ValueError):
            dataclasses.replace(motion, root_quat_wxyz=motion.root_quat_wxyz * 2).validate(self.robot)

    def test_deterministic_protocol_rejects_silent_unsupported_modes(self):
        ExecutionProtocol().validate()
        for cfg in (ExecutionProtocol(initialization='unknown'), ExecutionProtocol(max_steps=0),
                    ExecutionProtocol(domain_randomization=True)):
            with self.assertRaises(ValueError): cfg.validate()

    def test_rollout_requires_terminal_state_and_finite_actions(self):
        a = {'states': np.zeros((3,71)), 'actions': np.zeros((2,29)), 'executed_actions': np.zeros((2,29)),
             'joint_targets': np.zeros((2,29)), 'reference_qpos': np.zeros((3,36)),
             'reference_frames': np.arange(3), 'timestamps_s': np.array([0.,.02,.04])}
        validate_rollout_arrays(a)
        with self.assertRaises(ValueError): validate_rollout_arrays(dict(a, states=np.zeros((2,71))))
        with self.assertRaises(ValueError): validate_rollout_arrays(dict(a, actions=np.full((2,29),np.nan)))


if __name__ == '__main__': unittest.main()
