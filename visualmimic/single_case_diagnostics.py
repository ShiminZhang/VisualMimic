"""Human-layer diagnostics for one matched 3-D/video experiment, not retrieval accuracy."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .contracts import ArtifactRef, MotionSequence
from .storage import ArtifactStore, encode
from .backends.tmr_evaluator import TMRMotionEvaluator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    p.add_argument('--source-pair', required=True)
    p.add_argument('--video-pair', required=True)
    p.add_argument('--sample-id', required=True)
    p.add_argument('--repository', required=True)
    p.add_argument('--tmr-assets', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    status.write_bytes(encode({'status': 'running', 'arguments': vars(args)}))
    try:
        prepared = Path(args.prepared).resolve()
        pairs = {'a': Path(args.source_pair).resolve(), 'b': Path(args.video_pair).resolve()}
        results = {k: json.loads((v/'result.json').read_text()) for k, v in pairs.items()}
        for result in results.values():
            if result['input']['sample_id'] != args.sample_id:
                raise ValueError('Sample mismatch')
        if results['a']['editor'] != results['b']['editor'] or results['a']['generation'] != results['b']['generation']:
            raise ValueError('Editor or generation settings differ between branches')
        if results['a']['input']['instruction'] != results['b']['input']['instruction']:
            raise ValueError('Instruction mismatch')
        labels = json.loads((prepared/'labels/manifest.json').read_text())
        selected = [r for r in labels if r['key']['sample_id'] == args.sample_id]
        if len(selected) != 1:
            raise ValueError('Expected one official target for the sample')
        roots = [prepared/'labels/artifacts'] + [v/'artifacts' for v in pairs.values()]
        store = ArtifactStore(out/'artifacts', roots)
        target = store.read_motion(ArtifactRef(**selected[0]['target']))
        motions, provenance, retargeting = {}, {}, {}
        for kind, result in results.items():
            for branch in ('source', 'edited'):
                name = f'{kind}_{branch}'
                row = result['branches'][branch]
                provenance[name] = row['human']
                motions[name] = store.read_motion(ArtifactRef(**row['human']))
                diag = store.read_json(ArtifactRef(**row['diagnostics']))
                errors = np.asarray(diag['weighted_ik_errors'])
                if not np.isfinite(errors).all():
                    raise ValueError('Nonfinite IK diagnostics')
                retargeting[name] = {'weighted_ik_error_mean': errors.mean(axis=0).tolist(),
                    'max_joint_limit_violation_rad': diag['max_joint_limit_violation_rad'],
                    'ground_shift_z_m': diag['ground_shift_z_m']}
        original, reconstructed = motions['a_source'], motions['b_source']
        if original.spec != reconstructed.spec or not np.array_equal(original.timestamps_s, reconstructed.timestamps_s):
            raise ValueError('Reconstruction must retain source timing and representation')
        original_rotation = Rotation.from_rotvec(original.fields['rots'][:, 1:].reshape(-1, 3))
        predicted_rotation = Rotation.from_rotvec(reconstructed.fields['rots'][:, 1:].reshape(-1, 3))
        angles = (original_rotation.inv() * predicted_rotation).magnitude()
        evaluator = TMRMotionEvaluator(args.repository, args.tmr_assets, args.device)
        cropped = []
        for motion in motions.values():
            count = min(len(target.timestamps_s), len(motion.timestamps_s))
            cropped.append(MotionSequence(motion.spec, motion.timestamps_s[:count],
                {k: v[:count] for k, v in motion.fields.items()}, motion.valid_mask[:count]))
        embeddings = evaluator.encode([target] + cropped)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if not np.isfinite(embeddings).all() or np.any(norms == 0):
            raise ValueError('Invalid TMR embedding')
        embeddings = embeddings / norms
        scores = {name: float(embeddings[0] @ embeddings[i+1]) for i, name in enumerate(motions)}
        report = {'sample': selected[0]['key'], 'instruction': results['a']['input']['instruction'],
            'scope': 'single-case human and retargeting diagnostics; not retrieval accuracy or robot instruction success',
            'instruction_success': 'unassessed', 'evaluator': evaluator.identity,
            'target': selected[0]['target'], 'human_motions': provenance,
            'tmr_target_cosine': scores,
            'tmr_length_policy': 'generated cropped to its own target length; target unchanged',
            'reconstruction': {'frames': len(original.timestamps_s),
                'local_body_rotation_mean_geodesic_rad': float(angles.mean()),
                'local_body_rotation_p95_geodesic_rad': float(np.quantile(angles, .95)),
                'alignment': 'corresponding frames and local joint rotations; global heading excluded'},
            'retargeting': retargeting}
        (out/'report.json').write_bytes(encode(report))
        status.write_bytes(encode({'status': 'evaluated', 'instruction_success': 'unassessed'}))
        print(str(out/'report.json'))
    except Exception as exc:
        status.write_bytes(encode({'status': 'failed', 'error': str(exc)}))
        raise


if __name__ == '__main__':
    main()
