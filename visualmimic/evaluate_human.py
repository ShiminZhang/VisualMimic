"""Evaluate a fixed official split with the MotionFix TMR motion encoder.

Missing predictions are reported without reducing the retrieval candidate set.
The explicit unedited baseline is a diagnostic, never an edited-model result.
"""
import argparse
from pathlib import Path
import json
from .contracts import ArtifactRef, SampleKey
from .storage import ArtifactStore, encode, digest
from .backends.tmr_evaluator import TMRMotionEvaluator


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared', required=True)
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--predictions', help='JSON rows with key and motion ArtifactRef')
    inputs.add_argument('--unedited-baseline', action='store_true')
    p.add_argument('--prediction-artifacts')
    p.add_argument('--repository', required=True)
    p.add_argument('--tmr-assets', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve(); out.mkdir(parents=True,exist_ok=False)
    status = out/'status.json'; status.write_bytes(encode({'status':'running','arguments':vars(args)}))
    try:
        prepared = Path(args.prepared).resolve()
        metadata = json.loads((prepared/'preparation.json').read_text())
        sources = json.loads((prepared/'source/manifest.json').read_text())
        labels = json.loads((prepared/'labels/manifest.json').read_text())
        expected = [SampleKey(**r['key']) for r in labels]
        if [k.sample_id for k in expected] != metadata['ordered_ids'] or len(set(expected)) != len(expected):
            raise ValueError('Prepared candidate order mismatch')
        if args.unedited_baseline:
            predictions = [{'key':r['key'],'motion':r['source']} for r in sources]
        else:
            if not args.prediction_artifacts:
                raise ValueError('--prediction-artifacts is required with predictions')
            predictions = json.loads(Path(args.predictions).read_text())
        indexed = {}
        for row in predictions:
            if set(row) != {'key','motion'}:
                raise ValueError('Unexpected prediction fields')
            key = SampleKey(**row['key'])
            if key in indexed or key not in expected:
                raise ValueError('Duplicate or unexpected prediction')
            indexed[key] = ArtifactRef(**row['motion'])
        missing = [k for k in expected if k not in indexed]
        report = {'condition':'unedited-baseline' if args.unedited_baseline else 'generated',
                  'expected':len(expected),'available':len(indexed),'missing':missing,
                  'coverage':len(indexed)/len(expected),'candidate_keys':expected,
                  'candidate_id':digest(expected),'dataset_hashes':metadata['hashes'],
                  'original_cross_split_segments':metadata['cross_split_segment_count'],
                  'metrics':None}
        if not missing:
            roots = [prepared/'source/artifacts',prepared/'labels/artifacts']
            if args.prediction_artifacts: roots.append(args.prediction_artifacts)
            store = ArtifactStore(out/'artifacts',roots)
            targets = [store.read_motion(ArtifactRef(**r['target'])) for r in labels]
            generated = [store.read_motion(indexed[k]) for k in expected]
            evaluator = TMRMotionEvaluator(args.repository,args.tmr_assets,args.device)
            report['retrieval'] = evaluator.paired_retrieval([k.sample_id for k in expected],targets,generated)
            report['metrics'] = report['retrieval']['metrics']
            report['status'] = 'evaluated'
        else:
            report['status'] = 'incomplete_predictions'
            report['reason'] = 'Retrieval withheld to preserve the fixed candidate set'
        (out/'report.json').write_bytes(encode(report))
        status.write_bytes(encode({'status':report['status']}))
        print(str(out/'report.json'))
    except Exception as exc:
        status.write_bytes(encode({'status':'failed','error':str(exc)}))
        raise


if __name__ == '__main__':
    main()
