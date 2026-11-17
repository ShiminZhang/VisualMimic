"""Real SuperMimic/MuJoCo execution. Imported lazily in its GPU environment."""
import json
from pathlib import Path
import numpy as np
from .contracts import ArtifactRef
from .robot_contracts import PolicyBundle, RobotRollout, validate_rollout_arrays
from .reference_compiler import read_npz, put_npz, validate_reference
from .storage import digest


def load_policy_bundle(checkpoint, robot, store, implementation_commit):
    import torch
    ref = store.put_bytes(Path(checkpoint).read_bytes(), 'application/x-pytorch-checkpoint')
    ck = torch.load(ref.uri, map_location='cpu', weights_only=False)
    cfg = ck['config']
    env = cfg['env']
    return PolicyBundle(ref, implementation_commit, robot.identity,
                        digest({'policy': cfg['policy'], 'future_frames': env['future_frames'],
                                'anchor_obs': env.get('anchor_obs', False),
                                'history_length': env.get('history_length', 10)}),
                        'g1-normalized-joint-position-v1', store.put_json(cfg))


class SuperMimicExecutor:
    def __init__(self, store, robot, policy_bundle, device='cuda:0'):
        from supermimic.eval.benchmark import load_policy_from_checkpoint
        self.store, self.robot, self.bundle, self.device = store, robot, policy_bundle, device
        if policy_bundle.robot_spec_id != robot.identity:
            raise ValueError('Policy robot mismatch')
        store.read_bytes(policy_bundle.checkpoint)
        self.policy, self.config = load_policy_from_checkpoint(policy_bundle.checkpoint.uri, device)
        if digest(self.config) != digest(store.read_json(policy_bundle.environment_config)):
            raise ValueError('Policy configuration does not match bundle')

    def execute(self, reference, protocol):
        import torch
        from supermimic.envs import SuperMimicEnvCfgArgs, make_env
        from supermimic.envs.state_io import StateIO
        protocol.validate()
        if reference.robot_spec_id != self.robot.identity:
            raise ValueError('Reference robot mismatch')
        arrays = read_npz(self.store, reference.motion_npz)
        fps = validate_reference(arrays, self.robot)
        if reference.reference_id != reference.motion_npz.sha256 or reference.num_frames != len(arrays['joint_pos']) or reference.fps != fps:
            raise ValueError('Reference metadata mismatch')
        # MotionLibrary discovers .npz suffixes. Materialize verified bytes, not a mutable source path.
        materialized = self.store.root / (reference.motion_npz.sha256 + '.npz')
        materialized.write_bytes(self.store.read_bytes(reference.motion_npz))
        cfg = self.config['env']
        torch.manual_seed(protocol.seed)
        args = SuperMimicEnvCfgArgs(npz_files=[str(materialized)], num_envs=1,
                                   future_frames=cfg['future_frames'], history_length=cfg.get('history_length', 10),
                                   anchor_obs=cfg.get('anchor_obs', False), episode_length_s=1e5,
                                   action_noise_std=0., obs_noise=False, reset_noise=False,
                                   domain_randomization=False, sampling_mode='start', record_transitions=False,
                                   root_height_threshold=protocol.root_height_threshold,
                                   root_tilt_threshold=protocol.root_tilt_threshold, seed=protocol.seed)
        env = make_env(args, device=self.device)
        try:
            if not hasattr(env.cfg, 'auto_reset'):
                raise RuntimeError('Executor requires mjlab manual-reset support to preserve terminal states')
            env.cfg.auto_reset = False
            if not np.isclose(fps, 1 / env.step_dt):
                raise ValueError('Reference fps differs from environment control rate')
            cmd = env.command_manager.get_term('motion')
            cmd.forced_frame[:] = cmd.library.clip_start[0]
            obs, _ = env.reset(seed=protocol.seed)
            io = StateIO(env)
            def state():
                return io.read()[0].detach().cpu().numpy().copy()
            def ref_qpos():
                r = cmd.library.ref(cmd.frame)
                return torch.cat([r.root_pos_w, r.root_quat_w, r.joint_pos], -1)[0].cpu().numpy().copy()
            states, refs, frames = [state()], [ref_qpos()], [int(cmd.frame[0])]
            acts, exec_acts, targets, rewards, reward_terms, action_refs, action_frames = [], [], [], [], [], [], []
            term_names = list(env.reward_manager.active_terms)
            events, outcome = [], 'truncated'
            # Native SuperMimic clip completion may include one held-last-frame step.
            planned = reference.num_frames
            max_steps = min(protocol.max_steps or (planned + 2), planned + 2)
            with torch.inference_mode():
                for step in range(max_steps):
                    action_ref, action_frame = ref_qpos(), int(cmd.frame[0])
                    action = self.policy(obs['policy'], obs['goal'])
                    if not torch.isfinite(action).all():
                        raise ValueError('Policy produced nonfinite action')
                    obs, rew, term, tout, _ = env.step(action)
                    action_term = env.action_manager.get_term('joint_pos')
                    states.append(state())
                    refs.append(ref_qpos())
                    frames.append(int(cmd.frame[0]))
                    acts.append(action[0].cpu().numpy().copy())
                    exec_acts.append(action_term.executed_raw[0].cpu().numpy().copy())
                    targets.append(action_term._processed_actions[0].cpu().numpy().copy())
                    rewards.append(float(rew[0]))
                    reward_terms.append(env.reward_manager.term_rewards[0].cpu().numpy().copy())
                    action_refs.append(action_ref)
                    action_frames.append(action_frame)
                    if bool(term[0]) or bool(tout[0]):
                        outcome = 'terminated' if bool(term[0]) else ('completed' if bool(cmd.clip_end[0]) else 'truncated')
                        events.append({'step': step + 1, 'time_s': (step + 1) * env.step_dt,
                                       'terminated': bool(term[0]), 'timeout': bool(tout[0]),
                                       'clip_end': bool(cmd.clip_end[0]), 'outcome': outcome,
                                       'native_termination_terms': {
                                           name: bool(env.termination_manager.get_term(name)[0])
                                           for name in env.termination_manager.active_terms}})
                        break
            if not events:
                events.append({'step': len(acts), 'time_s': len(acts) * env.step_dt,
                               'outcome': 'truncated', 'reason': 'explicit_step_limit'})
            data = {'states': np.stack(states), 'actions': np.stack(acts), 'executed_actions': np.stack(exec_acts),
                    'joint_targets': np.stack(targets), 'reference_qpos': np.stack(refs),
                    'reference_frames': np.array(frames), 'action_reference_qpos': np.stack(action_refs),
                    'action_reference_frames': np.array(action_frames), 'timestamps_s': np.arange(len(states)) * env.step_dt,
                    'rewards': np.array(rewards), 'reward_terms': np.stack(reward_terms)}
            validate_rollout_arrays(data)
            trajectory = put_npz(self.store, data)
            from .deployment.environment_manifest import capture
            metadata = self.store.put_json({'runtime': capture(), 'robot': self.robot, 'policy': self.bundle, 'protocol': protocol,
                                            'reference': reference, 'step_dt': env.step_dt, 'physics_dt': env.physics_dt,
                                            'reward_terms': term_names, 'auto_reset': False,
                                            'state_layout': 'qpos[36],qvel[35]; free-joint angular velocity body-frame',
                                            'action_units': 'normalized policy output; joint_targets in radians',
                                            'terminal_state': 'captured after physics with auto_reset disabled',
                                            'initial_reference_frame': frames[0], 'initialization': 'native reference_start',
                                            'reference_alignment': 'state refs are next commands; action refs are control-step targets',
                                            'unavailable': ['contact_forces', 'actuator_torques']})
            return RobotRollout(trajectory.sha256, reference.reference_id, self.bundle.checkpoint.sha256,
                                protocol.identity, trajectory, self.store.put_json(events), planned, len(acts), outcome, metadata)
        finally:
            env.close()
