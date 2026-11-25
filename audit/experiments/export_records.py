"""Export existing experiment records without credentials, media, or API envelopes."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_gzip(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(gzip.compress(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode(), mtime=0))


def response_record(path):
    raw = path.read_bytes()
    obj = json.loads(raw)
    output = ''.join(c['text'] for item in obj.get('output', []) for c in item.get('content', []) if c.get('type') == 'output_text')
    return {'source_response_sha256': sha(raw), 'model': obj['model'], 'status': obj['status'], 'output_text': output, 'usage': obj.get('usage')}


def export_technical(root, out):
    files = {}

    def add(path, expected=None):
        path = Path(path).resolve()
        relative = str(path.relative_to(root))
        assert 'judge-secrets' not in path.parts
        if relative not in files:
            raw = path.read_bytes()
            json.loads(raw)
            files[relative] = {'source_path': relative, 'sha256': sha(raw), 'source_text': raw.decode()}
        item = files[relative]
        assert expected is None or item['sha256'] == expected
        return json.loads(item['source_text'])

    def add_ref(ref):
        return add(ref['uri'], ref['sha256'])

    def run_dir(ref):
        d = Path(ref['uri']).parent
        assert d.name.endswith('_artifacts')
        return d.with_name(d.name[:-10])

    campaigns = root / 'campaigns'
    rows = add(campaigns / 'abc_g1_retrieval307_v1/parent_snapshot/inputs.json')
    add(campaigns / 'abc_g1_retrieval307_v1/report.json')
    add(campaigns / 'luna_extend100_v1/combined_report.json')
    assert len(rows) == 307
    for row in rows:
        for branch in ('A', 'B'):
            result_ref = row[branch]['result']
            result = add_ref(result_ref)
            retarget = add_ref(result['retarget_report'])
            add_ref(result['rollout']['metadata'])
            editor_dir = run_dir(retarget['human'])
            for d in [editor_dir, run_dir(result['retarget_report']), run_dir(result_ref)]:
                add(d / 'run.json')
            key = sha(json.dumps(row['key'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode())
            add(editor_dir / 'items' / (key + '.json'))
    directories = [campaigns / n for n in ['luna_abc10_v1', 'luna_pregmr10_v1', 'luna_extend30_v1']]
    directories += [campaigns / 'luna_extend100_v1/shards' / str(i) for i in range(7)]
    responses = []
    for directory in directories:
        for name in ['protocol.json', 'tasks.json', 'report.json']:
            add(directory / name)
        for request_dir in sorted((directory / 'requests').iterdir()):
            result = add(request_dir / 'result.json')
            add(request_dir / 'request.json')
            response = response_record(request_dir / 'response.json')
            assert json.loads(response['output_text']) == result['assessment']
            responses.append({'key': result['key'], 'branch': result['branch'], **response})
    assert len(responses) == 400
    for name in ['luna_pregmr10_v1', 'luna_extend30_v1', 'luna_extend100_v1']:
        directory = campaigns / name
        for name in ['inputs.json', 'tasks.json', 'render/manifest.json']:
            add(directory / name)
        for p in sorted((directory / 'render_preparation').rglob('*.json')):
            add(p)
    write_gzip(out / 'c2_c4/source_records.json.gz', list(files.values()))
    write_gzip(out / 'c2_c4/response_records.json.gz', responses)
    return {'matched_pairs': len(rows), 'original_json_files': len(files), 'response_excerpts': len(responses)}


def export_r2(source, out, name):
    dest = out / name
    dest.mkdir(parents=True, exist_ok=True)
    results = json.loads((source / 'results.json').read_text())
    responses, requests, video_records = [], [], []
    for result in results:
        sid, condition = result['key']['sample_id'], result['condition']
        directory = source / 'judgments' / (sid + '_' + condition)
        response = response_record(directory / 'response.json')
        assert response['source_response_sha256'] == result['response_sha256']
        assert response['output_text'] == result['output_text']
        responses.append({'key': result['key'], 'condition': condition, **response})
        body = gzip.decompress((directory / 'request.json.gz').read_bytes())
        assert sha(body) == result['request_sha256']
        payload = json.loads(body)
        count = 0
        for message in payload['input']:
            for content in message['content']:
                if content['type'] == 'input_image':
                    url = content.pop('image_url')
                    prefix, encoded = url.split(',', 1)
                    image = base64.b64decode(encoded, validate=True)
                    content['omitted_image'] = {'mime_type': prefix[5:].split(';')[0], 'sha256': sha(image), 'bytes': len(image)}
                    count += 1
        assert count == len(result['source_frame_indices']) + len(result['candidate_frame_indices'])
        metadata = json.loads((directory / 'request_metadata.json').read_text())
        requests.append({'metadata': metadata, 'original_payload_sha256': sha(body), 'payload_without_image_bytes': payload})
        video_path = source / 'videos' / sid / (condition + '.json')
        raw = video_path.read_bytes()
        record = json.loads(raw)
        assert record['video']['sha256'] == result['candidate_video_sha256']
        video_records.append({'sample_id': sid, 'condition': condition, 'metadata_sha256': sha(raw), 'source_text': raw.decode()})
    (dest / 'results.json.gz').write_bytes(gzip.compress((source / 'results.json').read_bytes(), mtime=0))
    write_gzip(dest / 'response_records.json.gz', responses)
    write_gzip(dest / 'request_records.json.gz', requests)
    write_gzip(dest / 'video_records.json.gz', video_records)
    for path in sorted(source.glob('*.log')):
        (dest / 'logs').mkdir(exist_ok=True)
        shutil.copy2(path, dest / 'logs' / path.name)
    return {'cases': len(results) // 6, 'judgments': len(results), 'requests_verified_before_image_omission': len(requests), 'responses_verified_before_envelope_omission': len(responses), 'job_logs': len(list((dest / 'logs').glob('*.log')))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--project', type=Path, required=True)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--full', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {'c2_c4': export_technical(args.project.resolve(), args.output)}
    print('Technical records exported', flush=True)
    summary['r2_pilot'] = export_r2(args.pilot, args.output, 'r2_pilot')
    print('Pilot records exported', flush=True)
    summary['r2_full100'] = export_r2(args.full, args.output, 'r2_full100')
    summary['excluded'] = ['credentials and authentication headers', 'API account metadata and encrypted reasoning', 'image and video bytes', 'licensed motion/body-model assets and policy weights']
    (args.output / 'export_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
