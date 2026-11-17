import numpy as np
from .reference_compiler import read_npz
from .robot_contracts import validate_rollout_arrays


def evaluate_tracking(reference, rollout, store):
    if reference.reference_id != rollout.reference_id:
        raise ValueError('Rollout reference mismatch')
    a = validate_rollout_arrays(read_npz(store, rollout.trajectory))
    if len(a['actions']) != rollout.executed_steps:
        raise ValueError('Rollout step count mismatch')
    actual, target = a['states'][1:, :36], a['action_reference_qpos']
    q = actual[:, 3:7] / np.linalg.norm(actual[:, 3:7], axis=1, keepdims=True)
    r = target[:, 3:7] / np.linalg.norm(target[:, 3:7], axis=1, keepdims=True)
    orientation_error = 2 * np.arccos(np.clip(np.abs(np.sum(q * r, axis=1)), 0, 1))
    meta = store.read_json(rollout.metadata)
    reward_upper_per_step = 5.0 * meta['step_dt']
    return {'outcome': rollout.outcome, 'completed': rollout.outcome == 'completed',
            'executed_steps': rollout.executed_steps, 'planned_native_steps': rollout.planned_steps,
            'time_coverage': min(1., rollout.executed_steps / rollout.planned_steps),
            'duration_s': float(a['timestamps_s'][-1]), 'reference_duration_s': reference.duration_s,
            'root_position_rmse_m': float(np.sqrt(np.mean(np.sum((actual[:, :3] - target[:, :3]) ** 2, axis=1)))),
            'root_orientation_mean_rad': float(orientation_error.mean()),
            'joint_position_rmse_rad': float(np.sqrt(np.mean((actual[:, 7:] - target[:, 7:]) ** 2))),
            'native_normalized_return_executed': float(a['rewards'].sum() / (reward_upper_per_step * rollout.executed_steps)),
            'normalized_return_planned': float(a['rewards'].sum() / (reward_upper_per_step * rollout.planned_steps)),
            'metric_alignment': 'post-step state versus action-time reference, no spatial alignment',
            'reference_frame_coverage': len(set(a['action_reference_frames'].tolist())) / reference.num_frames}
