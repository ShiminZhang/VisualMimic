"""Retarget saved human predictions, then execute them with frozen SuperMimic.

The expected source-only manifest supplies the denominator and instructions.
Missing upstream outputs remain failed entries. Models and checkpoints are
content-addressed once per batch, rather than copied for every sample.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import numpy as np
from .batch_ledger import run_ledger
from .contracts import ArtifactRef, SampleKey
from .storage import ArtifactStore
from .reference_compiler import read_npz, put_npz
from .deployment.environment_manifest import capture


def validate_expected(expected):
    if not expected or any(set(row) not in ({'key', 'instruction'}, {'key', 'instruction', 'source'})
                           for row in expected):
        raise ValueError('Expected task-only or source-only manifest')
    keys = [SampleKey(**row['key']) for row in expected]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate expected key')
    return keys


def join_inputs(expected, actual, field):
    """Join by complete key, rejecting extras and retaining missing predictions."""
    validate_expected(expected)
    by_key = {}
    expected_keys = {SampleKey(**row['key']) for row in expected}
    for row in actual:
        key = SampleKey(**row['key'])
        if key in by_key or key not in expected_keys:
            raise ValueError('Duplicate or unexpected upstream sample key')
        allowed = {'key', field} | ({'instruction', 'retarget_report'} if field == 'robot_motion' else set())
        if set(row) != allowed:
            raise ValueError('Unsupported upstream manifest fields')
        by_key[key] = row
    joined = []
    for row in expected:
        upstream = by_key.get(SampleKey(**row['key']))
        if upstream and 'instruction' in upstream and upstream['instruction'] != row['instruction']:
            raise ValueError('Upstream instruction mismatch')
        joined.append({'key': row['key'], 'instruction': row['instruction'],
                       field: upstream[field] if upstream else None})
        if field == 'robot_motion':
            joined[-1]['retarget_report'] = upstream['retarget_report'] if upstream else None
    return joined


def human_predictions(expected, manifest):
    """Accept editor predictions or canonical reconstruction source manifests."""
    if manifest and all(set(row) == {'key', 'instruction', 'source'} for row in manifest):
        instructions = {SampleKey(**row['key']): row['instruction'] for row in expected}
        for row in manifest:
            if instructions.get(SampleKey(**row['key'])) != row['instruction']:
                raise ValueError('Reconstruction instruction/key mismatch')
        return [{'key': row['key'], 'motion': row['source']} for row in manifest]
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['retarget', 'execute'])
    parser.add_argument('--expected-manifest', required=True, help='Full task-only manifest; source-only also accepted for A')
    parser.add_argument('--manifest', help='Saved edited predictions or robot-motion manifest')
    parser.add_argument('--input-artifacts', required=True)
    parser.add_argument('--gmr-repository')
    parser.add_argument('--checkpoint')
    parser.add_argument('--implementation-commit')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--retry-interrupted', action='store_true')
    args = parser.parse_args()
    expected = json.loads(Path(args.expected_manifest).read_text())
    validate_expected(expected)
    output = Path(args.output).resolve()
    if output.exists() and not args.resume:
        raise FileExistsError(output)
    store = ArtifactStore(output.with_name(output.name + '_artifacts'), [args.input_artifacts])
    if args.stage == 'retarget':
        if not args.gmr_repository:
            parser.error('retarget requires --gmr-repository')
        if args.manifest:
            predictions = human_predictions(expected, json.loads(Path(args.manifest).read_text()))
        else:
            if any('source' not in row for row in expected):
                parser.error('Retarget without --manifest requires source references')
            predictions = [{'key': row['key'], 'motion': row['source']} for row in expected]
        rows = join_inputs(expected, predictions, 'motion')
        sys.path.insert(0, str(Path(args.gmr_repository).resolve()))
        from .prepare_robot_pair import gmr_robot_spec
        from .retargeting import GMRRetargeter, RetargetConfig
        from general_motion_retargeting import IK_CONFIG_DICT
        robot = gmr_robot_spec(store)
        config = RetargetConfig()
        mapping = store.put_bytes(Path(IK_CONFIG_DICT['smplx']['unitree_g1']).read_bytes())
        identity = {'adapter': 'gmr-batch-v1', 'robot': robot, 'config': config,
                    'mapping': mapping.sha256, 'runtime': capture()}

        def process(row):
            if row['motion'] is None:
                raise ValueError('Missing upstream human prediction')
            human = store.read_motion(ArtifactRef(**row['motion']))
            # GMR holds state within a clip; construct a fresh engine for each.
            retargeted = GMRRetargeter(store).retarget(human, robot, config)
            if retargeted.status != 'accepted':
                raise ValueError(f'Retarget rejected: {retargeted.reasons}; diagnostics={retargeted.diagnostics.uri}')
            motion = retargeted.motion
            robot_ref = put_npz(store, {'timestamps_s': motion.timestamps_s,
                'root_pos_w': motion.root_pos_w, 'root_quat_wxyz': motion.root_quat_wxyz,
                'joint_pos': motion.joint_pos, 'valid_mask': motion.valid_mask,
                'joint_names': np.asarray(robot.joint_names)})
            provenance = store.put_json({'key': row['key'], 'human': row['motion'], 'robot_motion': robot_ref,
                           'diagnostics': retargeted.diagnostics, 'mapping_id': retargeted.mapping_id})
            return {'key': row['key'], 'instruction': row['instruction'], 'robot_motion': asdict(robot_ref),
                    'retarget_report': asdict(provenance)}

        def verify(row):
            provenance = store.read_json(ArtifactRef(**row['retarget_report']))
            if provenance['key'] != row['key'] or provenance['robot_motion'] != row['robot_motion']:
                raise ValueError('Retarget provenance mismatch')
            store.read_json(ArtifactRef(**provenance['diagnostics']))
            data = read_npz(store, ArtifactRef(**row['robot_motion']))
            from .robot_contracts import RobotMotion
            if tuple(data['joint_names'].tolist()) != robot.joint_names:
                raise ValueError('Robot joint order mismatch')
            RobotMotion(robot.identity, data['timestamps_s'], data['root_pos_w'],
                        data['root_quat_wxyz'], data['joint_pos'], data['valid_mask']).validate(robot)
    else:
        if not args.manifest or not args.checkpoint or not args.implementation_commit:
            parser.error('execute requires --manifest, --checkpoint and --implementation-commit')
        rows = join_inputs(expected, json.loads(Path(args.manifest).read_text()), 'robot_motion')
        from .reference_compiler import g1_robot_spec, ReferenceCompiler
        from .robot_contracts import RobotMotion, ExecutionProtocol, validate_rollout_arrays
        from .supermimic_executor import load_policy_bundle, SuperMimicExecutor
        from .tracking_evaluation import evaluate_tracking
        from .rollout_rendering import render_rollout, RobotRenderConfig
        robot = g1_robot_spec(store)
        compiler = ReferenceCompiler(store, robot)
        policy = load_policy_bundle(args.checkpoint, robot, store, args.implementation_commit)
        executor = SuperMimicExecutor(store, robot, policy, args.device)
        protocol = ExecutionProtocol(seed=args.seed)
        identity = {'adapter': 'supermimic-batch-v1', 'robot': robot, 'policy': policy,
                    'protocol': protocol, 'render': RobotRenderConfig(), 'device': args.device,
                    'compiler': compiler.identity, 'runtime': capture()}

        def process(row):
            if row['robot_motion'] is None:
                raise ValueError('Missing upstream robot motion')
            provenance = store.read_json(ArtifactRef(**row['retarget_report']))
            if provenance['key'] != row['key'] or provenance['robot_motion'] != row['robot_motion']:
                raise ValueError('Upstream retarget provenance mismatch')
            data = read_npz(store, ArtifactRef(**row['robot_motion']))
            if tuple(data['joint_names'].tolist()) != robot.joint_names:
                raise ValueError('Robot joint order mismatch')
            motion = RobotMotion(robot.identity, data['timestamps_s'], data['root_pos_w'],
                                 data['root_quat_wxyz'], data['joint_pos'], data['valid_mask'])
            reference = compiler.compile(motion)
            rollout = executor.execute(reference, protocol)
            report = {'key': row['key'], 'instruction': row['instruction'], 'robot': robot,
                      'input_robot_motion': row['robot_motion'], 'retarget_report': row['retarget_report'],
                      'reference': reference, 'policy': policy, 'rollout': rollout,
                      'tracking': evaluate_tracking(reference, rollout, store),
                      'instruction_evaluation': {'status': 'unassessed',
                          'reason': 'Tracking metrics do not establish instruction success'}}
            # Persist physics evidence before rendering, even if the renderer fails.
            physics = store.put_json(report)
            try:
                with tempfile.TemporaryDirectory(prefix='rollout-render-', dir=store.root) as tmp:
                    report['render'] = render_rollout(rollout, store, Path(tmp) / 'rollout.mp4')
            except Exception as exc:
                raise RuntimeError(f'Render failed; saved physics report={physics.uri}: {exc}') from exc
            result_ref = store.put_json(report)
            return {'key': row['key'], 'instruction': row['instruction'], 'result': asdict(result_ref),
                    'outcome': rollout.outcome}

        def verify(row):
            report = store.read_json(ArtifactRef(**row['result']))
            if report['key'] != row['key'] or report['instruction'] != row['instruction']:
                raise ValueError('Execution report identity mismatch')
            if report['policy']['checkpoint']['sha256'] != policy.checkpoint.sha256:
                raise ValueError('Execution policy mismatch')
            if report['rollout']['protocol_id'] != protocol.identity:
                raise ValueError('Execution protocol mismatch')
            validate_rollout_arrays(read_npz(store, ArtifactRef(**report['rollout']['trajectory'])))
            for field in ('events', 'metadata'):
                store.read_json(ArtifactRef(**report['rollout'][field]))
            store.read_bytes(ArtifactRef(**report['render']['video']))
    results = run_ledger(rows, args.stage, identity, output, process, verify,
                         args.resume, args.retry_interrupted)
    if args.stage == 'execute':
        from .batch_edit import atomic_json
        counts = {outcome: sum(row['outcome'] == outcome for row in results)
                  for outcome in ('completed', 'terminated', 'truncated')}
        atomic_json(output / 'outcomes.json', {'expected': len(expected), 'with_rollout_and_video': len(results),
            'failed_or_missing': len(expected) - len(results), 'outcomes': counts,
            'instruction_success': 'unassessed', 'policy_sha256': policy.checkpoint.sha256,
            'protocol_id': protocol.identity})


if __name__ == '__main__':
    main()
