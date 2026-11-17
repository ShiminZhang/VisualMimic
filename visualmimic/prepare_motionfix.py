"""Import trusted official MotionFix data into separate source and label stores."""
import argparse
import json
from pathlib import Path
from hashlib import file_digest
from .storage import ArtifactStore, encode, digest
from .contracts import SampleKey
from .dataset import DatasetRecord, prepare
from .human_motion import from_motionfix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, help='Trusted official joblib archive')
    p.add_argument('--splits', required=True)
    p.add_argument('--annotations', required=True)
    p.add_argument('--split', choices=['train','val','test'], default='test')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = out / 'status.json'
    status.write_bytes(encode({'status':'running','arguments':vars(args)}))
    try:
        import joblib
        hashes = {}
        for label, path in [('dataset',args.dataset),('splits',args.splits),('annotations',args.annotations)]:
            with Path(path).open('rb') as stream:
                hashes[label] = file_digest(stream,'sha256').hexdigest()
        splits = json.loads(Path(args.splits).read_text())
        annotations = json.loads(Path(args.annotations).read_text())
        # Audit identical annotated source/target segments across all official splits,
        # including splits whose motion tensors were not downloaded.
        seen_ids, segments, missing_annotations = set(), {}, []
        for split, ids in splits.items():
            for sample_id in ids:
                if sample_id in seen_ids:
                    raise ValueError('Sample appears in multiple official splits')
                seen_ids.add(sample_id)
                if sample_id not in annotations:
                    missing_annotations.append({'sample_id':sample_id,'split':split})
                    if split == args.split:
                        raise ValueError(f'Missing annotation in selected split: {sample_id}')
                    continue
                row = annotations[sample_id]
                for branch in ('source','target'):
                    segment = digest([row['amass_path_'+branch],row['timestamp_'+branch]])
                    segments.setdefault(segment, []).append({'sample_id':sample_id,'split':split,'role':branch,
                        'amass_path':row['amass_path_'+branch],'timestamp':row['timestamp_'+branch]})
        overlaps = {k:v for k,v in segments.items() if len({r['split'] for r in v}) > 1}
        (out/'official_split_audit.json').write_bytes(encode({'cross_split_segment_count':len(overlaps),
            'cross_split_segments':overlaps,'missing_annotations':missing_annotations,
            'policy':'preserve official splits; do not claim motion-disjoint evaluation'}))
        data = joblib.load(args.dataset)
        expected = splits[args.split]
        if any(key not in data for key in expected):
            raise ValueError('Archive is missing samples in requested official split')
        source_store = ArtifactStore(out/'source/artifacts')
        label_store = ArtifactStore(out/'labels/artifacts', [source_store.root])
        version = 'motionfix-' + hashes['dataset'][:16]
        records = []
        for sample_id in expected:
            raw, row = data[sample_id], annotations[sample_id]
            if raw['text'].strip() != row['annotation'].strip():
                raise ValueError(f'Instruction metadata mismatch: {sample_id}')
            source = source_store.put_motion(from_motionfix(raw['motion_source']))
            target = label_store.put_motion(from_motionfix(raw['motion_target']))
            records.append(DatasetRecord(SampleKey(version,sample_id,args.split),raw['text'],
                digest([row['amass_path_source'],row['timestamp_source']]),
                digest([row['amass_path_target'],row['timestamp_target']]),source,target))
        prepared = prepare(records,source_store,label_store)
        (out/'source/manifest.json').write_bytes(source_store.read_bytes(prepared.source_manifest))
        (out/'tasks.json').write_bytes(encode([
            {'key': record.key, 'instruction': record.instruction} for record in records]))
        (out/'labels/manifest.json').write_bytes(label_store.read_bytes(prepared.label_manifest))
        (out/'preparation.json').write_bytes(encode({'prepared':prepared,'hashes':hashes,
            'ordered_ids':expected,'count':len(records),'split':args.split,
            'split_audit':'available annotation segments; tensor checksum audit on imported split only',
            'missing_annotations':missing_annotations,'cross_split_segment_count':len(overlaps)}))
        status.write_bytes(encode({'status':'prepared','count':len(records),'split':args.split}))
        print(str(out/'preparation.json'))
    except Exception as exc:
        status.write_bytes(encode({'status':'failed','error':str(exc)}))
        raise


if __name__ == '__main__':
    main()
