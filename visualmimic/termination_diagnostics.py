"""Recompute native height/tilt predicates from a saved, non-reset rollout.

This is retrospective evidence, not a recorded termination-manager reason.
References are action-time targets because mjlab checks termination before
advancing the motion command. No policy or physics is rerun.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from .contracts import ArtifactRef
from .storage import ArtifactStore, encode
from .reference_compiler import read_npz
from .robot_contracts import validate_rollout_arrays


def predicates(states, references, height_threshold, tilt_threshold):
    height = np.abs(states[:, 2] - references[:, 2])
    # z component of inverse-rotated unit downward gravity, wxyz quaternions.
    def gravity_z(q):
        if not np.allclose(np.linalg.norm(q, axis=1), 1, atol=1e-4):
            raise ValueError('Expected unit root quaternions')
        return 2 * (q[:, 1] ** 2 + q[:, 2] ** 2) - 1
    tilt = np.abs(gravity_z(states[:, 3:7]) - gravity_z(references[:, 3:7]))
    return height, tilt, height > height_threshold, tilt > tilt_threshold


def diagnose(run):
    run = Path(run).resolve()
    result = json.loads((run / 'result.json').read_text())
    store = ArtifactStore(run / 'artifacts')
    rollout = result['rollout']
    arrays = read_npz(store, ArtifactRef(**rollout['trajectory']))
    validate_rollout_arrays(arrays)
    metadata = store.read_json(ArtifactRef(**rollout['metadata']))
    if metadata['auto_reset'] is not False:
        raise ValueError('Terminal states require auto-reset disabled')
    protocol = metadata['protocol']
    height, tilt, bad_height, bad_tilt = predicates(
        arrays['states'][1:], arrays['action_reference_qpos'],
        protocol['root_height_threshold'], protocol['root_tilt_threshold'])
    crossed = np.flatnonzero(bad_height | bad_tilt)
    return {
        'scope': 'retrospective native predicates; not recorded manager reasons',
        'limitation': 'mjlab checks derived poses one physics substep before the saved refreshed state; predicates are retrospective evidence, not exact manager replay',
        'trajectory_sha256': rollout['trajectory']['sha256'],
        'outcome': rollout['outcome'], 'executed_steps': rollout['executed_steps'],
        'alignment': 'post-physics state versus action-time target, before command advance',
        'height_threshold_m': protocol['root_height_threshold'],
        'tilt_threshold': protocol['root_tilt_threshold'],
        'first_crossing_step': int(crossed[0] + 1) if len(crossed) else None,
        'terminal_height_error_m': float(height[-1]),
        'terminal_tilt_error': float(tilt[-1]),
        'terminal_predicates': {'root_height': bool(bad_height[-1]), 'root_tilt': bool(bad_tilt[-1])},
        'max_height_error_m': float(height.max()), 'max_tilt_error': float(tilt.max()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = diagnose(args.run)
    with Path(args.output).open('xb') as stream:
        stream.write(encode(report))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
