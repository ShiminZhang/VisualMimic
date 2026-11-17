"""GPU entrypoint: python -m visualmimic.execute_reference --help."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import traceback
from .storage import ArtifactStore
from .reference_compiler import g1_robot_spec, ReferenceCompiler
from .supermimic_executor import load_policy_bundle, SuperMimicExecutor
from .robot_contracts import ExecutionProtocol
from .tracking_evaluation import evaluate_tracking


def main():
    p = argparse.ArgumentParser()
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--reference')
    inputs.add_argument('--robot-motion')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--implementation-commit', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--max-steps', type=int)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)  # refuse uncertain replay into an existing run
    store = ArtifactStore(out / 'artifacts')
    (out / 'status.json').write_text(json.dumps({'status': 'running', 'arguments': vars(args)}, indent=2))
    try:
        robot = g1_robot_spec(store)
        compiler = ReferenceCompiler(store, robot)
        if args.reference:
            reference = compiler.import_npz(args.reference)
        else:
            import numpy as np
            from .robot_contracts import RobotMotion
            with np.load(args.robot_motion, allow_pickle=False) as data:
                if tuple(data['joint_names'].tolist()) != robot.joint_names:
                    raise ValueError('Robot motion joint order mismatch')
                motion = RobotMotion(robot.identity, data['timestamps_s'], data['root_pos_w'],
                                     data['root_quat_wxyz'], data['joint_pos'], data['valid_mask'])
            reference = compiler.compile(motion)
        policy = load_policy_bundle(args.checkpoint, robot, store, args.implementation_commit)
        executor = SuperMimicExecutor(store, robot, policy, args.device)
        rollout = executor.execute(reference, ExecutionProtocol(seed=args.seed, max_steps=args.max_steps))
        report = evaluate_tracking(reference, rollout, store)
        (out / 'result.json').write_text(json.dumps({'robot': asdict(robot), 'reference': asdict(reference),
                                                    'policy': asdict(policy), 'rollout': asdict(rollout),
                                                    'tracking': report}, indent=2))
        (out / 'status.json').write_text(json.dumps({'status': 'succeeded', 'outcome': rollout.outcome}))
        print(json.dumps(report, indent=2), flush=True)
    except Exception as exc:
        (out / 'status.json').write_text(json.dumps({'status': 'failed', 'error': str(exc), 'traceback': traceback.format_exc()}, indent=2))
        raise


if __name__ == '__main__':
    main()
