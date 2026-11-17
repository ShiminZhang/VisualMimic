"""Explicit physical task criteria on actual motion. Unsupported text stays unassessed."""
from dataclasses import dataclass
import numpy as np
from .reference_compiler import read_npz
from .robot_contracts import validate_rollout_arrays


@dataclass(frozen=True)
class InstructionTask:
    task_id: str
    instruction: str
    metric: str
    direction: str
    minimum_delta: float
    maximum_speed_change_mps: float | None = None


def _speed(arrays):
    dt = np.diff(arrays['timestamps_s'])
    return float(np.mean(np.linalg.norm(np.diff(arrays['states'][:, :2], axis=0), axis=1) / dt))


def evaluate_instruction(instruction, baseline, edited, task, store):
    if task is None:
        return {'status': 'unassessed', 'reason': 'No validated task definition for this instruction'}
    if instruction != task.instruction:
        raise ValueError('Task definition is for a different instruction')
    if baseline.policy_id != edited.policy_id or baseline.protocol_id != edited.protocol_id:
        raise ValueError('Instruction comparison requires the same policy and execution protocol')
    if baseline.outcome != 'completed' or edited.outcome != 'completed':
        return {'status': 'execution_incomplete', 'baseline_outcome': baseline.outcome,
                'edited_outcome': edited.outcome, 'success': False}
    a = validate_rollout_arrays(read_npz(store, baseline.trajectory))
    b = validate_rollout_arrays(read_npz(store, edited.trajectory))
    if task.metric == 'mean_planar_speed_mps':
        before, after = _speed(a), _speed(b)
    else:
        return {'status': 'unassessed', 'reason': f'Metric not implemented: {task.metric}'}
    if task.direction not in {'increase','decrease'} or task.minimum_delta < 0:
        raise ValueError('Invalid task criterion')
    delta = after - before
    directional = delta if task.direction == 'increase' else -delta
    preserve = (task.maximum_speed_change_mps is None or abs(_speed(b) - _speed(a)) <= task.maximum_speed_change_mps)
    return {'status': 'assessed', 'task_id': task.task_id, 'metric': task.metric,
            'baseline': before, 'edited': after, 'delta': delta,
            'success': bool(directional >= task.minimum_delta and preserve),
            'scope': 'explicit physical criterion only, not general language understanding'}
